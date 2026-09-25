"""Host-side jobs: yielding never cancels execution or blocks observation."""

import asyncio
import codecs
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Callable
from uuid import uuid4

from kosong.tooling import ToolError, ToolOk, ToolReturnValue

from lazarus.runtime import PythonRuntime


DEFAULT_TOOL_OUTPUT_LIMIT_KIB = 48


@dataclass
class Job:
    id: str
    output_path: Path
    started: float = field(default_factory=time.monotonic)
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    status: str = "running"
    result: ToolReturnValue | None = None
    finished: float | None = None
    cursor: int = 0
    notified: bool = False


class Jobs:
    def __init__(
        self,
        runtime: PythonRuntime,
        directory: Path,
        record: Callable[..., None] | None = None,
        *,
        tool_output_limit_kib: int = DEFAULT_TOOL_OUTPUT_LIMIT_KIB,
    ) -> None:
        if tool_output_limit_kib <= 0:
            raise ValueError("tool output limit must be positive")
        self._output_limit_bytes = tool_output_limit_kib * 1024
        self.runtime = runtime
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.closed = False
        self.record = record

    @property
    def active(self) -> Job | None:
        return next(
            (job for job in self.jobs.values() if job.status == "running"), None
        )

    def busy(self) -> ToolReturnValue | None:
        if self.closed:
            return ToolError(message="Session is closing.", brief="Closed")
        if active := self.active:
            return ToolError(
                message=f"Interpreter busy with job {active.id}. Use job to wait or cancel; this cell was not executed.",
                brief="Busy",
            )
        return None

    async def submit(
        self,
        code: str,
        timeout: float,
        yield_after: float,
        *,
        wait_completion: bool = False,
    ) -> ToolReturnValue:
        if busy := self.busy():
            return busy
        # Only finished, already observed jobs can leave the in-memory registry.
        # Their logs and journal entries stay available on disk.
        for job in list(self.jobs.values()):
            if len(self.jobs) < 20:
                break
            if job.status != "running" and job.notified:
                del self.jobs[job.id]
        job_id = uuid4().hex[:12]
        job = Job(job_id, self.directory / f"{job_id}.log")
        job.output_path.touch(mode=0o600, exist_ok=False)
        if self.record:
            self.record(
                "job_started",
                id=job.id,
                code=code,
                timeout=timeout,
                output_path=str(job.output_path),
            )
        self.jobs[job.id] = job
        job.task = asyncio.create_task(self._execute(job, code, timeout))
        if wait_completion:
            await asyncio.shield(job.task)
        return await self.inspect(job.id, wait=yield_after)

    async def _execute(self, job: Job, code: str, timeout: float) -> None:
        try:
            job.result = await self.runtime.run(
                code,
                timeout=timeout,
                output_path=job.output_path,
                cancel=job.cancel,
            )
            job.status = {
                "Cancelled": "cancelled",
                "Cell timed out": "timed_out",
                "Worker exited": "lost",
                "Worker failed": "lost",
            }.get(job.result.brief, "failed" if job.result.is_error else "completed")
        except Exception as exc:
            job.result = ToolError(
                message=f"Job failed: {type(exc).__name__}: {exc}", brief="Job failed"
            )
            job.status = "failed"
        finally:
            job.finished = time.monotonic()
        if self.record:
            self.record(
                "job_finished",
                id=job.id,
                status=job.status,
                message=job.result.message,
                generation=self.runtime.generation,
                cwd=self.runtime.cwd,
            )

    async def inspect(
        self,
        job_id: str,
        *,
        wait: float = 0,
        cancel: bool = False,
        cursor: int | None = None,
    ) -> ToolReturnValue:
        job = self.jobs.get(job_id)
        if job is None:
            return ToolError(
                message="Unknown or expired job ID. Use job() to list retained jobs.",
                brief="Unknown job",
            )
        if cancel and job.status == "running":
            job.cancel.set()
        if wait and job.task is not None:
            # asyncio.wait does not cancel the job when observation times out.
            await asyncio.wait({job.task}, timeout=wait)
        if job.task is not None and job.task.done():
            job.task.result()  # Surface journal failures instead of hiding them.
        data = self.snapshot(job, cursor)
        if job.status != "running":
            job.notified = True
        output = json.dumps(data, ensure_ascii=False)
        if job.result is not None and job.result.is_error:
            return ToolError(
                message=job.result.message, output=output, brief=job.result.brief
            )
        return ToolOk(output=output)

    def snapshot(self, job: Job, cursor: int | None = None) -> dict[str, object]:
        start = job.cursor if cursor is None else cursor
        limit = self._output_limit_bytes
        with job.output_path.open("rb") as log:
            end = log.seek(0, 2)
            if start > end:
                raise ValueError(f"cursor {start} exceeds log size {end}")
            log.seek(start)
            if end - start > limit:
                head_size = limit // 3
                head = log.read(head_size).decode(errors="replace")
                log.seek(end - (limit - head_size))
                tail = log.read(limit - head_size).decode(errors="replace")
                output = f"{head}\n... {end - start - limit:,} output bytes omitted; full log: {job.output_path} ...\n{tail}"
            else:
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                output = decoder.decode(
                    log.read(end - start), final=job.status != "running"
                )
                end -= len(decoder.getstate()[0])
        if cursor is None:
            job.cursor = end
        data: dict[str, object] = {
            "job_id": job.id,
            "status": job.status,
            "elapsed": round((job.finished or time.monotonic()) - job.started, 3),
            "output": output,
            "cursor": end,
            "output_path": str(job.output_path),
            "cwd": self.runtime.cwd,
            "generation": self.runtime.generation,
            "interpreter_alive": self.runtime._process is not None
            and self.runtime._process.returncode is None,
        }
        if job.cancel.is_set() and job.status == "running":
            data["cancel_requested"] = True
        if job.result is not None and job.result.message:
            data["message"] = job.result.message
        return data

    def listing(self) -> str:
        return json.dumps(
            [
                {
                    "job_id": job.id,
                    "status": job.status,
                    "output_path": str(job.output_path),
                }
                for job in self.jobs.values()
            ]
        )

    def notifications(self) -> list[str]:
        notices = []
        for job in self.jobs.values():
            if job.status != "running" and not job.notified:
                if job.task is not None and job.task.done():
                    job.task.result()
                notices.append(json.dumps(self.snapshot(job), ensure_ascii=False))
                job.notified = True
        return notices

    async def close(self) -> None:
        self.closed = True
        await self.interrupt()

    async def interrupt(self) -> None:
        for job in self.jobs.values():
            if job.status == "running":
                job.cancel.set()
        await asyncio.gather(
            *(job.task for job in self.jobs.values() if job.task),
            return_exceptions=True,
        )
