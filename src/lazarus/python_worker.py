import builtins
import contextlib
import json
import os
import sys
import tempfile
import time
import traceback
from collections.abc import Iterator
from typing import BinaryIO

PROTECTED_BUILTIN_NAMES = (
    "print",
    "open",
    "input",
    "exec",
    "eval",
    "compile",
    "__import__",
    "breakpoint",
)
ORIGINAL_BUILTINS = {name: getattr(builtins, name) for name in PROTECTED_BUILTIN_NAMES}


def restore_builtins(namespace: dict[str, object]) -> None:
    namespace["__builtins__"] = builtins
    for name, value in ORIGINAL_BUILTINS.items():
        setattr(builtins, name, value)
        namespace.pop(name, None)


@contextlib.contextmanager
def captured_fds() -> Iterator[tuple[BinaryIO, BinaryIO]]:
    with (
        tempfile.TemporaryFile(mode="w+b") as stdout_file,
        tempfile.TemporaryFile(mode="w+b") as stderr_file,
    ):
        saved_stdout = os.dup(sys.stdout.fileno())
        saved_stderr = os.dup(sys.stderr.fileno())

        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_file.fileno(), sys.stdout.fileno())
            os.dup2(stderr_file.fileno(), sys.stderr.fileno())
            yield stdout_file, stderr_file
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, sys.stdout.fileno())
            os.dup2(saved_stderr, sys.stderr.fileno())
            os.close(saved_stdout)
            os.close(saved_stderr)


def read_file(file: BinaryIO) -> str:
    file.flush()
    file.seek(0)
    return file.read().decode(errors="replace")


def execute(code: str, namespace: dict[str, object]) -> dict[str, object]:
    start = time.perf_counter()
    stdout = ""
    stderr = ""

    try:
        restore_builtins(namespace)
        with captured_fds() as (stdout_file, stderr_file):
            try:
                exec(code, namespace, namespace)
            finally:
                stdout = read_file(stdout_file)
                stderr = read_file(stderr_file)
        restore_builtins(namespace)
        return {
            "ok": True,
            "stdout": stdout,
            "stderr": stderr,
            "duration": time.perf_counter() - start,
        }
    except Exception:
        restore_builtins(namespace)
        return {
            "ok": False,
            "stdout": stdout,
            "stderr": stderr,
            "error": traceback.format_exc(),
            "duration": time.perf_counter() - start,
        }


def main() -> None:
    namespace: dict[str, object] = {"__name__": "__main__"}
    protocol = (
        os.fdopen(int(sys.argv[1]), "w", buffering=1)
        if len(sys.argv) > 1
        else sys.stdout
    )

    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = execute(request["code"], namespace)
        except Exception:
            response = {
                "ok": False,
                "stdout": "",
                "stderr": "",
                "error": traceback.format_exc(),
                "duration": 0,
            }

        print(json.dumps(response), file=protocol, flush=True)


if __name__ == "__main__":
    main()
