import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import BinaryIO, NotRequired, TypedDict

from IPython.core.interactiveshell import InteractiveShell
from traitlets.config import Config


os.environ.setdefault("IPYTHONDIR", os.path.join(tempfile.gettempdir(), "lazarus-ipython"))


MAX_OUTPUT_BYTES = 48 * 1024
MAX_SAVED_OUTPUTS = 20


class CellResult(TypedDict):
    ok: bool
    stdout: str
    stderr: str
    error: NotRequired[str]
    cwd: NotRequired[str]
    output_path: NotRequired[str]


def create_shell() -> InteractiveShell:
    config = Config()
    config.HistoryManager.hist_file = ":memory:"
    config.InteractiveShell.colors = "nocolor"
    return InteractiveShell.instance(config=config)


def _read_stream(file: BinaryIO, limit: int) -> str:
    file.flush()
    size = file.seek(0, os.SEEK_END)
    file.seek(0)
    if size <= limit:
        data = file.read()
    else:
        head_size = limit // 3
        tail_size = limit - head_size
        head = file.read(head_size)
        file.seek(-tail_size, os.SEEK_END)
        tail = file.read(tail_size)
        omitted = size - limit
        marker = f"\n... {omitted:,} output bytes omitted ...\n".encode()
        data = head + marker + tail
    return data.decode(errors="replace")


def _stream_limits(
    stdout_size: int, stderr_size: int, max_output_bytes: int = MAX_OUTPUT_BYTES
) -> tuple[int, int]:
    stdout_limit = min(stdout_size, max_output_bytes // 2)
    stderr_limit = min(stderr_size, max_output_bytes // 2)
    remaining = max_output_bytes - stdout_limit - stderr_limit
    stdout_limit += min(remaining, stdout_size - stdout_limit)
    remaining = max_output_bytes - stdout_limit - stderr_limit
    stderr_limit += min(remaining, stderr_size - stderr_limit)
    return stdout_limit, stderr_limit


def _save_output(
    stdout_file: BinaryIO,
    stderr_file: BinaryIO,
    output_dir: str,
    output_index: int,
) -> str:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"output-{os.getpid()}-{output_index:06d}.log"
    with path.open("wb") as target:
        target.write(b"[stdout]\n")
        stdout_file.seek(0)
        shutil.copyfileobj(stdout_file, target)
        target.write(b"\n[stderr]\n")
        stderr_file.seek(0)
        shutil.copyfileobj(stderr_file, target)

    saved = sorted(
        directory.glob("output-*.log"),
        key=lambda item: (item.stat().st_mtime_ns, item.name),
    )
    for old_path in saved[:-MAX_SAVED_OUTPUTS]:
        old_path.unlink()
    return str(path)


def _flush(stream: object) -> None:
    flush = getattr(stream, "flush", None)
    if not callable(flush):
        return
    try:
        flush()
    except Exception:
        pass


def execute_cell(
    shell: InteractiveShell,
    code: str,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
    output_dir: str | None = None,
    output_index: int = 0,
) -> CellResult:
    base_stdout = sys.__stdout__
    base_stderr = sys.__stderr__

    with (
        tempfile.TemporaryFile(mode="w+b") as stdout_file,
        tempfile.TemporaryFile(mode="w+b") as stderr_file,
    ):
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        _flush(sys.stdout)
        _flush(sys.stderr)
        os.dup2(stdout_file.fileno(), 1)
        os.dup2(stderr_file.fileno(), 2)
        sys.stdout = base_stdout
        sys.stderr = base_stderr

        result = None
        infrastructure_error: BaseException | None = None
        try:
            result = shell.run_cell(code, store_history=False, silent=False)
        except BaseException as exc:
            infrastructure_error = exc
        finally:
            _flush(sys.stdout)
            _flush(sys.stderr)
            _flush(base_stdout)
            _flush(base_stderr)
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            sys.stdout = base_stdout
            sys.stderr = base_stderr

        stdout_size = stdout_file.seek(0, os.SEEK_END)
        stderr_size = stderr_file.seek(0, os.SEEK_END)
        output_path = None
        if output_dir is not None and stdout_size + stderr_size > max_output_bytes:
            try:
                output_path = _save_output(
                    stdout_file, stderr_file, output_dir, output_index
                )
            except OSError:
                pass
        stdout_limit, stderr_limit = _stream_limits(
            stdout_size, stderr_size, max_output_bytes
        )
        stdout = _read_stream(stdout_file, stdout_limit)
        stderr = _read_stream(stderr_file, stderr_limit)

    error = infrastructure_error
    if error is None and result is not None:
        error = result.error_before_exec or result.error_in_exec

    response: CellResult = {
        "ok": error is None,
        "stdout": stdout,
        "stderr": stderr,
    }
    if error is not None:
        response["error"] = f"{type(error).__name__}: {error}"
    if output_path is not None:
        response["output_path"] = output_path
    try:
        response["cwd"] = os.getcwd()
    except OSError:
        pass
    return response


def _protocol_input() -> BinaryIO:
    request_fd = os.dup(0)
    devnull_fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull_fd, 0)
    os.close(devnull_fd)
    sys.stdin = open(0, encoding="utf-8", closefd=False)
    return os.fdopen(request_fd, "rb", buffering=0)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: python_worker RESPONSE_FD MAX_OUTPUT_BYTES OUTPUT_DIR")

    requests = _protocol_input()
    responses = os.fdopen(int(sys.argv[1]), "w", encoding="utf-8", buffering=1)
    max_output_bytes = int(sys.argv[2])
    if max_output_bytes <= 0:
        raise SystemExit("MAX_OUTPUT_BYTES must be positive")
    output_dir = sys.argv[3]
    shell = create_shell()

    for output_index, raw_line in enumerate(requests, start=1):
        try:
            request = json.loads(raw_line)
            response = execute_cell(
                shell,
                request["code"],
                max_output_bytes,
                output_dir,
                output_index,
            )
        except BaseException as exc:
            response = {
                "ok": False,
                "stdout": "",
                "stderr": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        responses.write(json.dumps(response, ensure_ascii=False) + "\n")
        responses.flush()


if __name__ == "__main__":
    main()
