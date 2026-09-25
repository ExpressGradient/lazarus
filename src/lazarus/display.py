"""Plain terminal status; full tool results still go to the model and journal."""

import json
import re
import shutil

from kosong.tooling import ToolReturnValue


# Strip terminal escape sequences before rendering untrusted code or output.
_ESCAPES = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]")


def result_text(value: ToolReturnValue) -> str:
    parts = [str(value.output)] if value.output else []
    if value.message:
        parts.append(value.message)
    return "\n".join(parts).rstrip() or "(no output)"


class ToolDisplay:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._states: dict[str, tuple[str, bool]] = {}
        self._labels: dict[str, str] = {}
        self._cursors: dict[str, int] = {}

    def cell(self, name: str, code: str, description: str = "") -> None:
        if self.verbose:
            print(f"\n[{name}]\n{code}", flush=True)
        else:
            print(
                f"  Python · {self._brief(description.strip() or code, 100)}",
                flush=True,
            )

    def result(self, value: ToolReturnValue, *, label: str | None = None) -> None:
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
            changed = self._states.get(job_id) != state
            self._states[job_id] = state
            if label:
                self._labels[job_id] = self._brief(label, 100)
            purpose = self._labels.get(job_id, job_id)
            status_label = (
                "cancelling" if cancelling and status == "running" else status
            )
            elapsed = data.get("elapsed")
            timing = f" · {elapsed:.1f}s" if isinstance(elapsed, (int, float)) else ""
            detail = data.get("message") or value.message
            # Errors remain visible without dumping tracebacks or full output.
            error = (
                f" · {self._brief(detail)}" if detail and status != "completed" else ""
            )
            if changed and not (label is not None and status == "running"):
                print(
                    f"  Python · {purpose} · {status_label}{timing}{error}", flush=True
                )
            cursor = data.get("cursor")
            seen = self._cursors.get(job_id, 0)
            output = data.get("output")
            if isinstance(cursor, int) and cursor > seen:
                self._cursors[job_id] = cursor
                if output and not value.is_error:
                    self._preview(str(output))
            elif changed and status == "completed" and not seen and not output:
                print("    (no output)", flush=True)
        elif value.is_error:
            print(
                f"  Tool error · {self._brief(value.message or value.brief)}",
                flush=True,
            )

    @staticmethod
    def _brief(value: object, limit: int = 160) -> str:
        text = " ".join(_ESCAPES.sub("", str(value)).split())
        text = "".join(char for char in text if char.isprintable())
        return text if len(text) <= limit else text[: limit - 1] + "…"

    def _preview(self, output: str) -> None:
        width = max(20, min(120, shutil.get_terminal_size().columns - 4))
        lines = _ESCAPES.sub("", output).strip().splitlines()
        for line in lines[:3]:
            print(f"    {self._brief(line, width)}", flush=True)
        if len(lines) > 3:
            print("    …", flush=True)
