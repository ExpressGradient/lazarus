import contextlib
import io
import json
import sys
import time
import traceback


def main() -> None:
    namespace = {"__name__": "__main__"}

    for line in sys.stdin:
        start = time.perf_counter()
        stdout = io.StringIO()
        stderr = io.StringIO()

        try:
            request = json.loads(line)
            code = request["code"]

            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(code, namespace, namespace)

            response = {
                "ok": True,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "duration": time.perf_counter() - start,
            }
        except Exception:
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
