import asyncio
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import sys
import tempfile
import time
import tracemalloc
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kosong.chat_provider import ChatProviderError
from kosong.message import Message, TextPart, ToolCall
from kosong.tooling.simple import SimpleToolset

from lazarus.cli import CellTool, TokenTotals, run_request
from lazarus.jobs import Jobs
from lazarus.runtime import PythonRuntime
from lazarus.session import Session, pending_tool_calls


class DispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_dropped_stream_never_dispatches_partial_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = PythonRuntime()
            jobs = Jobs(runtime, root / "jobs")
            marker = root / "side-effect"

            class Stream:
                id = "fixture"
                usage = None

                async def __aiter__(self):
                    yield ToolCall(
                        id="write",
                        function=ToolCall.FunctionBody(
                            name="python",
                            arguments=json.dumps(
                                {"code": f"open({str(marker)!r}, 'w').write('done')"}
                            ),
                        ),
                    )
                    yield TextPart(text="next part")
                    await asyncio.sleep(0.05)
                    raise ChatProviderError("connection dropped")

            class Provider:
                async def generate(self, *args):
                    return Stream()

            try:
                with self.assertRaisesRegex(ChatProviderError, "connection dropped"):
                    await run_request(
                        Provider(),
                        SimpleToolset([CellTool(jobs, "python", "run")]),
                        runtime,
                        [],
                        "task",
                        TokenTotals(),
                        150000,
                        jobs=jobs,
                        system_prompt="fixture",
                    )
                self.assertFalse(jobs.jobs)
                self.assertFalse(marker.exists())
            finally:
                await jobs.close()
                await runtime.close()

    async def test_dispatch_is_durable_and_resume_restores_cwd_without_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "nested"
            target.mkdir()
            marker = root / "effects"
            session = Session(str(root / "session"))
            journal = session.directory / "journal.jsonl"
            runtime = PythonRuntime()
            jobs = Jobs(runtime, session.directory / "jobs", session.record)
            session.record("session", system_prompt="fixed", cwd=runtime.cwd)
            code = (
                "import json, os\n"
                f"events = [json.loads(s) for s in open({str(journal)!r})]\n"
                "assert any(e['event'] == 'message' and e['message'].get('tool_calls') for e in events)\n"
                f"os.chdir({str(target)!r})\n"
                f"with open({str(marker)!r}, 'a') as f: f.write('once')\n"
            )
            call = ToolCall(
                id="change",
                function=ToolCall.FunctionBody(
                    name="python",
                    arguments=json.dumps({"code": code, "yield_after": 5}),
                ),
            )
            replies = [
                SimpleNamespace(
                    usage=None,
                    message=Message(role="assistant", content=[], tool_calls=[call]),
                ),
                SimpleNamespace(
                    usage=None, message=Message(role="assistant", content="done")
                ),
            ]
            try:
                with (
                    patch("lazarus.cli.kosong.generate", side_effect=replies),
                    redirect_stdout(StringIO()),
                ):
                    await run_request(
                        object(),
                        SimpleToolset([CellTool(jobs, "python", "run")]),
                        runtime,
                        [],
                        "task",
                        TokenTotals(),
                        150000,
                        jobs=jobs,
                        session=session,
                        system_prompt="fixed",
                    )
            finally:
                await jobs.close()
                await runtime.close()
                session.close()
            self.assertEqual(marker.read_text(), "once")
            resumed = Session(str(root / "session"), resume=True)
            try:
                history, _, prompt, cwd = resumed.restore()
                self.assertEqual(cwd, str(target.resolve()))
                self.assertEqual(prompt, "fixed")
                self.assertFalse(pending_tool_calls(history))
                self.assertEqual(marker.read_text(), "once")
            finally:
                resumed.close()

    async def test_dispatch_failure_leaves_recoverable_call(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Session(directory)
            session.record("session", system_prompt="fixed", cwd=directory)
            call = ToolCall(
                id="pending",
                function=ToolCall.FunctionBody(
                    name="python", arguments='{"code":"pass"}'
                ),
            )
            reply = SimpleNamespace(
                usage=None,
                message=Message(role="assistant", content=[], tool_calls=[call]),
            )

            def fail(call):
                raise OSError("dispatch unavailable")

            try:
                with (
                    patch("lazarus.cli.kosong.generate", return_value=reply),
                    redirect_stdout(StringIO()),
                ):
                    with self.assertRaisesRegex(OSError, "dispatch unavailable"):
                        await run_request(
                            object(),
                            SimpleNamespace(tools=[], handle=fail),
                            PythonRuntime(),
                            [],
                            "task",
                            TokenTotals(),
                            150000,
                            session=session,
                            system_prompt="fixed",
                        )
            finally:
                session.close()
            resumed = Session(directory, resume=True)
            try:
                history, _, _, _ = resumed.restore()
                self.assertFalse(pending_tool_calls(history))
                self.assertIn("not replayed", history[-2].extract_text())
            finally:
                resumed.close()


class JournalTests(unittest.TestCase):
    def test_large_journal_streams_resets_and_preserves_latest_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            with path.open("w") as file:

                def write(**entry):
                    file.write(json.dumps(entry) + "\n")

                write(event="session", cwd=directory, system_prompt="fixed")
                write(event="request", task="task")
                for _ in range(1000):
                    write(
                        event="message",
                        message=Message(
                            role="assistant", content="x" * 16000
                        ).model_dump(mode="json"),
                    )
                    write(event="reset", history=[])
                # Legacy tool results and current job metadata both restore cwd.
                write(
                    event="message",
                    message=Message(
                        role="tool",
                        tool_call_id="old",
                        content=json.dumps({"cwd": "/old"}),
                    ).model_dump(mode="json"),
                )
                write(event="job_finished", cwd=directory)
                write(
                    event="reset",
                    history=[
                        Message(role="user", content="latest").model_dump(mode="json")
                    ],
                )
                file.write('{"event":"torn')
            size = path.stat().st_size
            tracemalloc.start()
            session = Session(directory, resume=True)
            try:
                history, _, prompt, cwd = session.restore()
                _, peak = tracemalloc.get_traced_memory()
            finally:
                session.close()
                tracemalloc.stop()
            self.assertLess(peak, size // 4)
            self.assertEqual((cwd, prompt), (directory, "fixed"))
            self.assertEqual(history[0].extract_text(), "latest")
            self.assertEqual(len(history), 2)
            with path.open() as file:
                for line in file:
                    json.loads(line)

    def test_complete_corruption_fails_without_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            raw = b'{"event":"broken"\n'
            path.write_bytes(raw)
            session = Session(directory, resume=True)
            try:
                with self.assertRaises(json.JSONDecodeError):
                    session.restore()
            finally:
                session.close()
            self.assertEqual(path.read_bytes(), raw)


# Real CLI, signals, and worker processes; only model replies are deterministic.
INTERACTIVE_FIXTURE = r"""
import asyncio, json, sys
from types import SimpleNamespace
from kosong.message import Message, ToolCall
import lazarus.cli as cli
import lazarus.runtime as runtime
runtime.CELL_INTERRUPT_GRACE = .2
cli._system_prompt = lambda cwd: "fixed"
codes = {
    "slow": "import time; answer=42; print('READY_SLOW', flush=True); time.sleep(60)",
    "check": "print('VALUE', answer)",
    "stubborn": "import time, signal; signal.signal(signal.SIGINT, signal.SIG_IGN); print('READY_STUBBORN', flush=True); time.sleep(60)",
    "fresh": "print('FRESH', 'answer' in globals())",
}
seen = set()
async def generate(**kwargs):
    text = next(m.extract_text() for m in reversed(kwargs['history']) if m.role == 'user' and m.extract_text() in {*codes, 'thinking'})
    calls = []
    if text not in seen:
        seen.add(text)
        if text == 'thinking':
            print('MODEL_WAIT', flush=True)
            await asyncio.sleep(60)
        else:
            calls = [ToolCall(id=text, function=ToolCall.FunctionBody(name='python', arguments=json.dumps({'code':codes[text], 'description':text, 'yield_after':60})))]
    return SimpleNamespace(usage=None, message=Message(role='assistant', content='' if calls else 'done', tool_calls=calls))
cli.kosong.generate = generate
asyncio.run(cli.run(SimpleNamespace(name='fixture', model_name='fixture'), None, 150000, 48, session_dir=sys.argv[1]))
"""


class InteractiveTests(unittest.TestCase):
    def test_ctrl_c_preserves_session_and_reports_forced_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session"
            master, slave = pty.openpty()
            process = subprocess.Popen(
                [sys.executable, "-c", INTERACTIVE_FIXTURE, str(session)],
                stdin=slave,
                stdout=slave,
                stderr=slave,
                start_new_session=True,
            )
            os.close(slave)
            transcript = bytearray()

            def wait_for(text=None, log_marker=None):
                before = len(transcript)
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if select.select([master], [], [], 0.05)[0]:
                        try:
                            chunk = os.read(master, 65536)
                        except OSError:
                            chunk = b""
                        if not chunk:
                            self.fail(transcript.decode(errors="replace"))
                        transcript.extend(chunk)
                    if text and text.encode() in transcript[before:]:
                        return transcript[before:].decode()
                    if log_marker and any(
                        log_marker in p.read_text(errors="replace")
                        for p in (session / "jobs").glob("*.log")
                    ):
                        return ""
                self.fail("Timed out: " + transcript.decode(errors="replace"))

            def send(text):
                os.write(master, (text + "\n").encode())

            try:
                wait_for("You: ")
                send("slow")
                wait_for(log_marker="READY_SLOW")
                process.send_signal(signal.SIGINT)
                self.assertIn("state was preserved", wait_for("You: "))
                send("check")
                self.assertIn("VALUE 42", wait_for("You: "))
                send("thinking")
                wait_for("MODEL_WAIT")
                process.send_signal(signal.SIGINT)
                self.assertIn("state was preserved", wait_for("You: "))
                send("stubborn")
                wait_for(log_marker="READY_STUBBORN")
                process.send_signal(signal.SIGINT)
                self.assertIn("state was lost", wait_for("You: "))
                send("fresh")
                self.assertIn("FRESH False", wait_for("You: "))
                process.send_signal(signal.SIGINT)
                wait_for("You: ")
                send("/quit")
                self.assertEqual(process.wait(timeout=5), 0)
                messages = [
                    Message.model_validate(e["message"])
                    for e in (
                        json.loads(line)
                        for line in (session / "journal.jsonl").read_text().splitlines()
                    )
                    if e["event"] == "message"
                ]
                self.assertFalse(pending_tool_calls(messages))
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                os.close(master)
