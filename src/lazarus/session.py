"""Append-only evidence and conversation recovery; never replay Python cells."""

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from uuid import uuid4

from kosong.message import Message


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
            # Only repair an interrupted final write. Malformed complete records
            # are an error, rather than silently losing evidence.
            raw = self._file.read()
            if raw and not raw.endswith(b"\n"):
                raw = raw[: raw.rfind(b"\n") + 1]
                self._file.truncate(len(raw))
            self.events = [json.loads(line) for line in raw.splitlines()]
            self._file.seek(0, 2)
        except BaseException:
            self._file.close()
            raise

    def record(self, event: str, **fields: object) -> None:
        entry = {"event": event, **fields}
        self._file.write((json.dumps(entry, ensure_ascii=False) + "\n").encode())
        self._file.flush()
        os.fsync(self._file.fileno())

    def message(self, message: Message) -> None:
        self.record("message", message=message.model_dump(mode="json"))

    def restore(self) -> tuple[list[Message], str, str, str]:
        history: list[Message] = []
        task = ""
        system_prompt = ""
        cwd = ""
        for event in self.events:
            match event["event"]:
                case "session":
                    system_prompt, cwd = event["system_prompt"], event["cwd"]
                case "request":
                    task = event["task"]
                case "message":
                    history.append(Message.model_validate(event["message"]))
                case "reset":
                    history = [
                        Message.model_validate(item) for item in event["history"]
                    ]
        if not task or not system_prompt or not cwd:
            raise ValueError("Session has no recoverable task.")
        # A crash may leave an assistant tool call without its matching result.
        # Close that exchange explicitly; execution outcome is unknown.
        pending = {}
        for message in history:
            for call in message.tool_calls or []:
                pending[call.id] = call
            if message.role == "tool":
                pending.pop(message.tool_call_id, None)
        for call_id in pending:
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
