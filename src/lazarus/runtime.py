"""One persistent interpreter, owned and supervised by the host process."""

import asyncio
import json
import os
from pathlib import Path
import signal
import sys

from kosong.tooling import ToolError, ToolOk, ToolReturnValue

DEFAULT_CELL_TIMEOUT = 300.0
CELL_INTERRUPT_GRACE = 10.0


class PythonRuntime:
    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None
        self._reader_transport: asyncio.ReadTransport | None = None
        self._lock = asyncio.Lock()
        self.cwd = os.getcwd()
        self.initial_cwd = self.cwd
        self.generation = 0

    async def run(
        self,
        code: str,
        timeout: float = DEFAULT_CELL_TIMEOUT,
        *,
        output_path: Path,
        cancel: asyncio.Event | None = None,
    ) -> ToolReturnValue:
        async with self._lock:
            if cancel is not None and cancel.is_set():
                return ToolError(
                    message="Cell cancelled before execution.", brief="Cancelled"
                )
            execution = asyncio.create_task(self._run_locked(code, output_path))
            cancellation = asyncio.create_task(cancel.wait()) if cancel else None
            try:
                tasks = {execution, cancellation} if cancellation else {execution}
                await asyncio.wait(
                    tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                )
                # Completion wins a simultaneous timeout/cancel race.
                if execution.done():
                    return execution.result()
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
                if cancellation is not None and cancellation.done():
                    return await self._interrupt_cell("Cell cancelled", "Cancelled")
                return await self._interrupt_cell(
                    f"Cell exceeded the {timeout:g}s timeout", "Cell timed out"
                )
            except asyncio.CancelledError:
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
                await self._interrupt_cell("Cell cancelled", "Cancelled")
                raise
            finally:
                if cancellation is not None:
                    cancellation.cancel()
                    await asyncio.gather(cancellation, return_exceptions=True)

    async def _run_locked(self, code: str, output_path: Path) -> ToolReturnValue:
        try:
            await self._ensure_worker()
            assert self._process is not None
            assert self._process.stdin is not None
            assert self._reader is not None

            request = (
                json.dumps(
                    {
                        "code": code,
                        "output_path": str(output_path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            self._process.stdin.write(request.encode())
            await self._process.stdin.drain()
            raw_response = await self._reader.readline()
            if not raw_response:
                await self._forget_worker()
                return ToolError(
                    message="The IPython worker exited; its in-memory state was lost.",
                    output="",
                    brief="Worker exited",
                )
            response = json.loads(raw_response)
        except (OSError, ValueError) as exc:
            await self._forget_worker()
            return ToolError(
                message=f"The IPython worker protocol failed: {exc}",
                output="",
                brief="Worker failed",
            )

        if isinstance(response.get("cwd"), str):
            self.cwd = response["cwd"]
        if response.get("ok"):
            return ToolOk(output="")
        return ToolError(
            message=str(response.get("error", "IPython cell failed")),
            brief="Cell failed",
        )

    async def _interrupt_cell(self, reason: str, brief: str) -> ToolReturnValue:
        process = self._process
        reader = self._reader
        if process is not None and process.returncode is None and reader is not None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGINT)
                else:
                    process.send_signal(signal.SIGINT)
                raw_response = await asyncio.wait_for(
                    reader.readline(), CELL_INTERRUPT_GRACE
                )
                if raw_response:
                    response = json.loads(raw_response)
                    await self._kill_worker_children(process.pid)
                    if isinstance(response.get("cwd"), str):
                        self.cwd = response["cwd"]
                    return ToolError(
                        message=(
                            f"{reason} and was interrupted; interpreter state was preserved. "
                            "Partial effects remain; inspect before retrying."
                        ),
                        brief=brief,
                    )
            except (
                OSError,
                TimeoutError,
                ConnectionResetError,
                ValueError,
                json.JSONDecodeError,
            ):
                pass

        await self._forget_worker()
        return ToolError(
            message=f"{reason}; interpreter state was lost. Partial effects may remain.",
            output="",
            brief=brief,
        )

    @staticmethod
    async def _kill_worker_children(group: int) -> None:
        if os.name != "posix":
            return
        # SIGINT can return control to IPython while a child ignores it. Kill
        # surviving members of our private process group, preserving IPython.
        probe = await asyncio.create_subprocess_exec(
            "ps",
            "-A",
            "-o",
            "pid=,pgid=",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            output, _ = await asyncio.wait_for(probe.communicate(), 2)
            if probe.returncode:
                raise OSError("Could not inspect the worker process group")
            for line in output.splitlines():
                pid, pgid = map(int, line.split())
                if pgid == group and pid != group:
                    try:
                        # Recheck ownership in case the process exited meanwhile.
                        if os.getpgid(pid) == group:
                            os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        finally:
            if probe.returncode is None:
                probe.kill()
                await probe.wait()

    async def _ensure_worker(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return

        await self._forget_worker()
        read_fd, write_fd = os.pipe()
        try:
            self._process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",
                "-m",
                "lazarus.python_worker",
                str(write_fd),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                pass_fds=(write_fd,),
                start_new_session=os.name == "posix",
                cwd=self.cwd,
            )
        except BaseException:
            os.close(read_fd)
            raise
        finally:
            os.close(write_fd)

        self.generation += 1
        # Only bounded status metadata crosses the pipe; output stays in the log.
        reader = asyncio.StreamReader(limit=65536)
        protocol = asyncio.StreamReaderProtocol(reader)
        loop = asyncio.get_running_loop()
        transport, _ = await loop.connect_read_pipe(
            lambda: protocol,
            os.fdopen(read_fd, "rb", buffering=0),
        )
        self._reader = reader
        self._reader_transport = transport

    async def _forget_worker(self) -> None:
        if self._reader_transport is not None:
            self._reader_transport.close()
        self._reader = None
        self._reader_transport = None

        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=0.5)
            except TimeoutError:
                self._signal_worker(process, force=False)
                try:
                    await asyncio.wait_for(process.wait(), timeout=0.5)
                except TimeoutError:
                    self._signal_worker(process, force=True)
                    await process.wait()
        if os.name == "posix":
            self._signal_worker(process, force=True)

    @staticmethod
    def _signal_worker(process: asyncio.subprocess.Process, *, force: bool) -> None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif force:
            process.kill()
        else:
            process.terminate()

    async def close(self) -> None:
        async with self._lock:
            await self._forget_worker()
