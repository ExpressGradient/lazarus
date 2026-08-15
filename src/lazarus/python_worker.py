import json
import os
import sys
import tempfile
from typing import BinaryIO, NotRequired, TypedDict

from IPython.core.interactiveshell import InteractiveShell
from traitlets.config import Config


os.environ.setdefault("IPYTHONDIR", os.path.join(tempfile.gettempdir(), "lazarus-ipython"))


MAX_STREAM_BYTES = 100_000


class CellResult(TypedDict):
    ok: bool
    stdout: str
    stderr: str
    error: NotRequired[str]
    cwd: NotRequired[str]


def create_shell() -> InteractiveShell:
    config = Config()
    config.HistoryManager.hist_file = ":memory:"
    config.InteractiveShell.colors = "nocolor"
    return InteractiveShell.instance(config=config)


def _read_stream(file: BinaryIO) -> str:
    file.flush()
    size = file.seek(0, os.SEEK_END)
    file.seek(0)
    if size <= MAX_STREAM_BYTES:
        data = file.read()
    else:
        head_size = MAX_STREAM_BYTES * 3 // 4
        tail_size = MAX_STREAM_BYTES - head_size
        head = file.read(head_size)
        file.seek(-tail_size, os.SEEK_END)
        tail = file.read(tail_size)
        omitted = size - MAX_STREAM_BYTES
        marker = f"\n... {omitted:,} output bytes omitted ...\n".encode()
        data = head + marker + tail
    return data.decode(errors="replace")


def _flush(stream: object) -> None:
    flush = getattr(stream, "flush", None)
    if not callable(flush):
        return
    try:
        flush()
    except Exception:
        pass


def execute_cell(shell: InteractiveShell, code: str) -> CellResult:
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

        stdout = _read_stream(stdout_file)
        stderr = _read_stream(stderr_file)

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
    if len(sys.argv) != 2:
        raise SystemExit("usage: python_worker RESPONSE_FD")

    requests = _protocol_input()
    responses = os.fdopen(int(sys.argv[1]), "w", encoding="utf-8", buffering=1)
    shell = create_shell()

    for raw_line in requests:
        try:
            request = json.loads(raw_line)
            response = execute_cell(shell, request["code"])
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
