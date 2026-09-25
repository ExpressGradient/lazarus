import json
import os
import signal
import sys
import tempfile
from typing import BinaryIO, NotRequired, TypedDict

from IPython.core.interactiveshell import InteractiveShell
from traitlets.config import Config


os.environ.setdefault(
    "IPYTHONDIR", os.path.join(tempfile.gettempdir(), "lazarus-ipython")
)


class CellResult(TypedDict):
    ok: bool
    error: NotRequired[str]
    cwd: NotRequired[str]


def create_shell() -> InteractiveShell:
    config = Config()
    config.HistoryManager.hist_file = ":memory:"
    config.InteractiveShell.colors = "nocolor"
    return InteractiveShell.instance(config=config)


def _flush(stream: object) -> None:
    flush = getattr(stream, "flush", None)
    if not callable(flush):
        return
    try:
        flush()
    except Exception:
        pass


def execute_cell(shell: InteractiveShell, code: str, output_path: str) -> CellResult:
    base_stdout = sys.__stdout__
    base_stderr = sys.__stderr__

    # The host owns this single log. The worker never reads or truncates it.
    with open(output_path, "ab", buffering=0) as output_file:
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        _flush(sys.stdout)
        _flush(sys.stderr)
        os.dup2(output_file.fileno(), 1)
        os.dup2(output_file.fileno(), 2)
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

    error = infrastructure_error
    if error is None and result is not None:
        error = result.error_before_exec or result.error_in_exec

    response: CellResult = {"ok": error is None}
    if error is not None:
        response["error"] = f"{type(error).__name__}: {error}"[:4096]
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
    executing = False

    def interrupt(signum: int, frame: object) -> None:
        # A deadline can race with a just-written response. SIGINT must not
        # kill an idle interpreter after the supervisor receives that response.
        if executing:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)

    for raw_line in requests:
        try:
            executing = True
            request = json.loads(raw_line)
            response = execute_cell(shell, request["code"], request["output_path"])
        except BaseException as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:4096]}
        finally:
            executing = False
        responses.write(json.dumps(response, ensure_ascii=False) + "\n")
        responses.flush()


if __name__ == "__main__":
    main()
