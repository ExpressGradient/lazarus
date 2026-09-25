"""Plain terminal status; full tool results still go to the model and journal."""

import json

from kosong.tooling import ToolReturnValue


def result_text(value: ToolReturnValue) -> str:
    parts = [str(value.output)] if value.output else []
    if value.message:
        parts.append(value.message)
    return "\n".join(parts).rstrip() or "(no output)"


class ToolDisplay:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._states: dict[str, tuple[str, bool]] = {}

    def cell(self, name: str, code: str) -> None:
        if self.verbose:
            print(f"\n[{name}]\n{code}", flush=True)

    def result(self, value: ToolReturnValue) -> None:
        if self.verbose:
            label = "error" if value.is_error else "output"
            print(f"\n[{label}]\n{result_text(value)}", flush=True)
            return
        try:
            data = json.loads(str(value.output))
        except (ValueError, TypeError):
            data = None
        if isinstance(data, dict) and "job_id" in data:
            job_id = str(data["job_id"])
            status = str(data.get("status", "unknown"))
            cancelling = bool(data.get("cancel_requested"))
            state = (status, cancelling)
            if self._states.get(job_id) == state:
                return
            self._states[job_id] = state
            label = "cancelling" if cancelling and status == "running" else status
            elapsed = data.get("elapsed")
            timing = f" · {elapsed:.1f}s" if isinstance(elapsed, (int, float)) else ""
            detail = data.get("message") or value.message
            # Errors remain visible without dumping tracebacks or full output.
            error = (
                f" · {self._brief(detail)}" if detail and status != "completed" else ""
            )
            print(f"  Python {job_id} · {label}{timing}{error}", flush=True)
        elif value.is_error:
            print(
                f"  Tool error · {self._brief(value.message or value.brief)}",
                flush=True,
            )

    @staticmethod
    def _brief(value: object) -> str:
        return " ".join(str(value).split())[:160]
