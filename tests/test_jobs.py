"""Integration checks against real worker processes and the agent boundary."""

import asyncio
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kosong.message import Message, ToolCall
from kosong.tooling.simple import SimpleToolset
from pydantic import ValidationError

from lazarus.cli import (
    CellParams,
    CellTool,
    JobParams,
    JobTool,
    TokenTotals,
    run_request,
)
from lazarus.jobs import Job, Jobs
from lazarus.runtime import PythonRuntime


class JobIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime = PythonRuntime()
        self.jobs = Jobs(self.runtime, self.root / "jobs")

    async def asyncTearDown(self):
        await self.jobs.close()
        await self.runtime.close()
        self.temp.cleanup()

    async def start(self, code, timeout=10):
        result = await self.jobs.submit(code, timeout, 0)
        self.assertFalse(result.is_error, result)
        return json.loads(result.output)

    async def wait_for_log(self, job_id, marker):
        path = self.jobs.jobs[job_id].output_path
        async with asyncio.timeout(5):
            while marker not in path.read_text(errors="replace"):
                await asyncio.sleep(0.01)

    async def test_immediate_live_logs_cursor_wait_and_cancel(self):
        data = await self.start(
            "import time\nanswer = 42\nprint('ready', flush=True)\ntime.sleep(30)"
        )
        job_id = data["job_id"]
        self.assertEqual(data["status"], "running")
        await self.wait_for_log(job_id, "ready")
        first = json.loads((await self.jobs.inspect(job_id)).output)
        self.assertIn("ready", first["output"])
        second = json.loads((await self.jobs.inspect(job_id, wait=0.02)).output)
        self.assertEqual(second["output"], "")
        self.assertEqual(second["status"], "running")
        reread = json.loads((await self.jobs.inspect(job_id, cursor=0)).output)
        self.assertIn("ready", reread["output"])
        blocked = await self.jobs.submit("answer = 0", 10, 0)
        self.assertTrue(blocked.is_error)
        cancelled = await self.jobs.inspect(job_id, cancel=True, wait=5)
        self.assertEqual(json.loads(cancelled.output)["status"], "cancelled")
        self.assertIn("preserved", cancelled.message)
        recovered = await self.jobs.submit("print(answer)", 10, 5)
        self.assertIn("42", json.loads(recovered.output)["output"])
        self.assertFalse(self.jobs.notifications())

    async def test_cancel_before_start_never_executes(self):
        path = self.root / "side-effect"
        data = await self.start(f"open({str(path)!r}, 'w').write('oops')")
        result = await self.jobs.inspect(data["job_id"], cancel=True, wait=5)
        self.assertEqual(json.loads(result.output)["status"], "cancelled")
        self.assertFalse(path.exists())

    async def test_crash_partial_output_survives_and_worker_recovers(self):
        data = await self.start(
            "import os\nprint('last words', flush=True)\nos._exit(17)"
        )
        result = await self.jobs.inspect(data["job_id"], wait=5)
        state = json.loads(result.output)
        self.assertEqual(state["status"], "lost")
        self.assertIn("last words", state["output"])
        self.assertFalse(state["interpreter_alive"])
        result = await self.jobs.submit("print(6 * 7)", 10, 5)
        self.assertFalse(result.is_error, result)
        self.assertIn("42", json.loads(result.output)["output"])
        self.assertEqual(self.runtime.generation, 2)

    async def test_timeout_ignoring_interrupt_forces_restart(self):
        with patch("lazarus.runtime.CELL_INTERRUPT_GRACE", 0.1):
            result = await self.jobs.submit(
                "import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nprint('ready', flush=True)\ntime.sleep(30)",
                1,
                5,
            )
        self.assertEqual(json.loads(result.output)["status"], "timed_out")
        self.assertIn("state was lost", result.message)
        result = await self.jobs.submit("print(42)", 10, 5)
        self.assertFalse(result.is_error)

    async def test_large_output_and_protocol_above_64k(self):
        result = await self.jobs.submit(
            "import os\nos.write(1, b'h' * 1000000)\nos.write(2, b'tail')", 10, 5
        )
        data = json.loads(result.output)
        self.assertFalse(result.is_error, result)
        self.assertIn("output bytes omitted", data["output"])
        self.assertIn("tail", data["output"])
        self.assertGreater(Path(data["output_path"]).stat().st_size, 1000000)
        self.assertLess(len(data["output"]), 50000)
        runtime = PythonRuntime()
        large_jobs = Jobs(runtime, self.root / "large-jobs", tool_output_limit_kib=128)
        try:
            result = await large_jobs.submit("print('x' * 150000)", 10, 5)
            self.assertFalse(result.is_error, result)
            self.assertGreater(len(result.output), 65536)
        finally:
            await large_jobs.close()
            await runtime.close()

    async def test_job_failures_and_invalid_observations_are_tool_errors(self):
        with patch.object(self.runtime, "run", side_effect=RuntimeError("boom")):
            failed = await self.jobs.submit("pass", 10, 5)
        state = json.loads(failed.output)
        self.assertTrue(failed.is_error)
        self.assertEqual("failed", state["status"])
        self.assertIn("RuntimeError: boom", failed.message)

        invalid = await self.jobs.inspect(state["job_id"], cursor=1000)
        unknown = await self.jobs.inspect("missing")
        missing_id = await JobTool(self.jobs)(JobParams(wait=1))
        self.assertEqual("Invalid cursor", invalid.brief)
        self.assertEqual("Unknown job", unknown.brief)
        self.assertEqual("Missing job ID", missing_id.brief)

    def test_live_snapshot_waits_for_complete_utf8(self):
        output = self.root / "partial.log"
        output.write_bytes(b"\xe2\x82")
        job = Job("partial", output)

        first = self.jobs.snapshot(job)
        self.assertEqual(("", 0), (first["output"], first["cursor"]))

        with output.open("ab") as stream:
            stream.write(b"\xac\n")
        second = self.jobs.snapshot(job)
        self.assertEqual(("€\n", 4), (second["output"], second["cursor"]))

    async def test_observer_cancellation_does_not_cancel_job(self):
        data = await self.start(
            "import time\nprint('ready', flush=True)\ntime.sleep(.4)\nprint('done')"
        )
        await self.wait_for_log(data["job_id"], "ready")
        observer = asyncio.create_task(self.jobs.inspect(data["job_id"], wait=5))
        await asyncio.sleep(0.01)
        observer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await observer
        result = await self.jobs.inspect(data["job_id"], wait=5)
        self.assertEqual(json.loads(result.output)["status"], "completed")

    async def test_handoff_rejects_busy_then_preserves_state(self):
        data = await self.start("import time\ntime.sleep(.2)\nanswer = 42")
        tool = CellTool(self.jobs, "start_new_loop", "handoff")
        with redirect_stdout(StringIO()):
            blocked = await tool(CellParams(code="answer = 0"))
            self.assertTrue(blocked.is_error)
            await self.jobs.inspect(data["job_id"], wait=5)
            handoff = await tool(CellParams(code="print(answer)"))
        self.assertFalse(handoff.is_error)
        self.assertIn("42", handoff.output)

    async def test_agent_appends_completion_keeps_prefix_and_waits_before_exit(self):
        tools = SimpleToolset(
            [CellTool(self.jobs, "python", "run"), JobTool(self.jobs)]
        )
        histories = []
        prompts = []
        calls = 0

        async def generate(**kwargs):
            nonlocal calls
            calls += 1
            histories.append([m.model_dump(mode="json") for m in kwargs["history"]])
            prompts.append(kwargs["system_prompt"])
            tool_calls = []
            if calls == 1:
                call = ToolCall(
                    id="run",
                    function=ToolCall.FunctionBody(
                        name="python",
                        arguments=json.dumps(
                            {
                                "code": "import time, os\nanswer = 42\nprint('started', flush=True)\ntime.sleep(.3)\nos.chdir('/tmp')\nprint(answer)"
                            }
                        ),
                    ),
                )
                tool_calls = [call]
            return SimpleNamespace(
                usage=None,
                message=Message(
                    role="assistant",
                    content="done" if calls > 1 else "working",
                    tool_calls=tool_calls,
                ),
            )

        with (
            patch("lazarus.cli.kosong.generate", side_effect=generate),
            redirect_stdout(StringIO()),
        ):
            history = await run_request(
                object(),
                tools,
                self.runtime,
                [],
                "Task",
                TokenTotals(),
                150000,
                jobs=self.jobs,
            )
        self.assertEqual(calls, 3)
        self.assertEqual(len(set(prompts)), 1)
        for previous, current in zip(histories, histories[1:]):
            self.assertEqual(current[: len(previous)], previous)
        self.assertEqual(
            sum("[Job completion]" in m.extract_text() for m in history), 1
        )
        self.assertIn("42", str(history))
        self.assertIsNone(self.jobs.active)

    async def test_idle_interrupt_does_not_kill_completed_worker(self):
        await self.jobs.submit("answer = 42", 10, 5)
        os.kill(self.runtime._process.pid, signal.SIGINT)
        await asyncio.sleep(0.02)
        result = await self.jobs.submit("print(answer)", 10, 5)
        self.assertFalse(result.is_error, result)
        self.assertIn("42", result.output)

    async def test_cancel_kills_stubborn_child_and_preserves_state(self):
        child = "import os, signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); signal.signal(signal.SIGTERM, signal.SIG_IGN); print(os.getpid(), flush=True); time.sleep(60)"
        code = f"import subprocess, sys, time\nanswer=42\np = subprocess.Popen([sys.executable, '-u', '-c', {child!r}])\nprint('parent-ready', flush=True)\ntime.sleep(60)"
        data = await self.start(code)
        await self.wait_for_log(data["job_id"], "parent-ready")
        async with asyncio.timeout(5):
            while len(Path(data["output_path"]).read_text().splitlines()) < 2:
                await asyncio.sleep(0.01)
        result = await self.jobs.inspect(data["job_id"], cancel=True, wait=5)
        self.assertEqual(json.loads(result.output)["status"], "cancelled")
        result = await self.jobs.submit("print(p.wait(timeout=2), answer)", 10, 5)
        self.assertFalse(result.is_error, result)
        self.assertIn("-9 42", result.output)

    async def test_failed_process_inspection_falls_back_to_group_shutdown(self):
        data = await self.start(
            "import time\nprint('ready', flush=True)\ntime.sleep(30)"
        )
        await self.wait_for_log(data["job_id"], "ready")
        with patch.object(
            self.runtime, "_kill_worker_children", side_effect=PermissionError("denied")
        ):
            result = await self.jobs.inspect(data["job_id"], cancel=True, wait=5)
        self.assertIn("state was lost", result.message)
        self.assertIsNone(self.runtime._process)

    async def test_parallel_submission_executes_only_one_cell(self):
        results = await asyncio.gather(
            *(self.jobs.submit("import time; time.sleep(.1)", 10, 0) for _ in range(8))
        )
        self.assertEqual(sum(not r.is_error for r in results), 1)
        self.assertEqual(len(self.jobs.jobs), 1)

    async def test_finished_jobs_evicted_but_logs_retained(self):
        for i in range(22):
            result = await self.jobs.submit(f"print({i})", 10, 5)
            self.assertFalse(result.is_error, result)
        self.assertEqual(len(self.jobs.jobs), 20)
        self.assertEqual(len(list((self.root / "jobs").glob("*.log"))), 22)

    def test_invalid_deadlines_rejected(self):
        for value in [float("nan"), float("inf"), -1, 0]:
            with self.assertRaises(ValidationError):
                CellParams(code="pass", timeout=value)
        for value in [float("nan"), float("inf"), -1, 61]:
            with self.assertRaises(ValidationError):
                JobParams(wait=value)
