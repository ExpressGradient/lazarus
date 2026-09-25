"""Append-only evidence and conversation recovery; never replay Python cells."""

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from collections.abc import Iterator
from uuid import uuid4

from kosong.message import Message, ToolCall


def pending_tool_calls(history: list[Message]) -> dict[str, ToolCall]:
    pending: dict[str, ToolCall] = {}
    for message in history:
        for call in message.tool_calls or []:
            pending[call.id] = call
        if message.role == "tool" and message.tool_call_id is not None:
            pending.pop(message.tool_call_id, None)
    return pending


class Session:
    def __init__(self, directory: str | None, *, resume: bool = False) -> None:
        if directory is None:
            name = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
            )
            directory = str(Path.home() / ".local/state/lazarus/sessions" / name)
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.directory / "journal.jsonl"
        self._file = path.open("r+b" if resume else "x+b")
        try:
            fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.chmod(path, 0o600)
            self._file.seek(0, 2)
        except BaseException:
            self._file.close()
            raise

    def record(self, event: str, **fields: object) -> None:
        entry = {"event": event, **fields}
        self._file.seek(0, 2)
        self._file.write((json.dumps(entry, ensure_ascii=False) + "\n").encode())
        self._file.flush()
        os.fsync(self._file.fileno())

    def message(self, message: Message) -> None:
        self.record("message", message=message.model_dump(mode="json"))

    def _events(self) -> Iterator[dict]:
        self._file.seek(0)
        try:
            while True:
                start = self._file.tell()
                line = self._file.readline()
                if not line:
                    break
                # Only a torn final record can be discarded. Invalid complete
                # records must fail visibly rather than hide lost history.
                if not line.endswith(b"\n"):
                    self._file.truncate(start)
                    self._file.flush()
                    os.fsync(self._file.fileno())
                    break
                yield json.loads(line)
        finally:
            self._file.seek(0, 2)

    def restore(self) -> tuple[list[Message], str, str, str]:
        history: list[Message] = []
        task = ""
        system_prompt = ""
        cwd = ""
        for event in self._events():
            match event["event"]:
                case "session":
                    system_prompt, cwd = event["system_prompt"], event["cwd"]
                case "request":
                    task = event["task"]
                case "message":
                    message = Message.model_validate(event["message"])
                    history.append(message)
                    # Older journals only recorded cwd in tool results.
                    if message.role == "tool":
                        try:
                            data = json.loads(message.extract_text())
                        except ValueError:
                            continue
                        if isinstance(data, dict) and isinstance(data.get("cwd"), str):
                            cwd = data["cwd"]
                case "job_finished":
                    if isinstance(event.get("cwd"), str):
                        cwd = event["cwd"]
                case "reset":
                    history = [
                        Message.model_validate(item) for item in event["history"]
                    ]
        if not task or not system_prompt or not cwd:
            raise ValueError("Session has no recoverable task.")
        # A crash may leave an assistant tool call without its matching result.
        # Close that exchange explicitly; execution outcome is unknown.
        for call_id in pending_tool_calls(history):
            message = Message(
                role="tool",
                tool_call_id=call_id,
                content="Session interrupted; result unknown. Inspect effects before retrying. Code was not replayed.",
            )
            history.append(message)
            self.message(message)
        recovery = Message(
            role="user",
            content=(
                "Session resumed from its journal. This is a fresh interpreter: Python names, "
                "objects, and old job handles are lost. Files and job logs remain in "
                f"{self.directory}. Prior subprocesses may still exist; inspect them and any "
                "partial effects before retrying. No code has been replayed. Continue the task."
            ),
        )
        history.append(recovery)
        self.message(recovery)
        return history, task, system_prompt, cwd

    def close(self) -> None:
        self._file.close()
