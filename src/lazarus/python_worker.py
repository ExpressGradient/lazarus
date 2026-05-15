import builtins
import contextlib
import io
import json
import sys
import time
import traceback


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
ORIGINAL_BUILTINS = {
    name: getattr(builtins, name) for name in PROTECTED_BUILTIN_NAMES
}


def restore_protected_builtins(namespace: dict[str, object]) -> None:
    namespace["__builtins__"] = builtins
    for name, value in ORIGINAL_BUILTINS.items():
        setattr(builtins, name, value)
        namespace.pop(name, None)


def main() -> None:
    namespace = {"__name__": "__main__"}

    for line in sys.stdin:
        start = time.perf_counter()
        stdout = io.StringIO()
        stderr = io.StringIO()

        try:
            request = json.loads(line)
            code = request["code"]
            restore_protected_builtins(namespace)

            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(code, namespace, namespace)

            restore_protected_builtins(namespace)

            response = {
                "ok": True,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "duration": time.perf_counter() - start,
            }
        except Exception:
            restore_protected_builtins(namespace)

            response = {
                "ok": False,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "error": traceback.format_exc(),
                "duration": time.perf_counter() - start,
            }

        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
