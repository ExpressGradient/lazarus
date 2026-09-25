import asyncio
import json
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kosong.message import Message, ToolCall
from kosong.chat_provider import TokenUsage
from kosong.tooling import ToolError, ToolOk, ToolResult

from lazarus.cli import (
    PythonRuntime,
    TokenTotals,
    _new_loop_history,
    _system_prompt,
    build_parser,
    run_request,
)
from lazarus.jobs import Jobs

MAX_OUTPUT_BYTES = 48 * 1024


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.runtime = PythonRuntime()
        self.temp = tempfile.TemporaryDirectory()
        self.jobs = Jobs(self.runtime, Path(self.temp.name) / "jobs")

    async def run_cell(self, code, timeout=10):
        return await self.jobs.submit(code, timeout, 5)

    async def asyncTearDown(self) -> None:
        await self.jobs.close()
        await self.runtime.close()
        self.temp.cleanup()

    async def test_worker_protocol_preserves_state(self) -> None:
        first = await self.run_cell("answer = 40")
        second = await self.run_cell("answer + 2")

        self.assertFalse(first.is_error)
        self.assertFalse(second.is_error)
        self.assertIn("42", second.output)

    async def test_worker_input_does_not_consume_protocol(self) -> None:
        failed = await self.run_cell("input()")
        recovered = await self.run_cell("6 * 7")

        self.assertTrue(failed.is_error)
        self.assertIn("EOFError", failed.message)
        self.assertFalse(recovered.is_error)
        self.assertIn("42", recovered.output)

    async def test_worker_truncates_combined_output(self) -> None:
        result = await self.run_cell(
            "import os; "
            f"os.write(1, b'A' * {MAX_OUTPUT_BYTES}); "
            f"os.write(2, b'B' * {MAX_OUTPUT_BYTES})"
        )

        self.assertFalse(result.is_error)
        data = json.loads(result.output)
        self.assertEqual(1, data["output"].count("output bytes omitted"))
        self.assertLess(len(data["output"].encode()), MAX_OUTPUT_BYTES + 500)
        saved = Path(data["output_path"])
        full_output = saved.read_bytes()
        self.assertIn(b"A" * MAX_OUTPUT_BYTES, full_output)
        self.assertIn(b"B" * MAX_OUTPUT_BYTES, full_output)
        await self.runtime.close()
        self.assertTrue(saved.exists())

    def test_tool_output_limit_is_configurable(self) -> None:
        args = build_parser().parse_args(["--tool-output-limit-kib", "64"])

        self.assertEqual(64, args.tool_output_limit_kib)

    async def test_timeout_interrupts_worker_and_preserves_state(self) -> None:
        await self.run_cell("answer = 42")
        timed_out = await self.run_cell("import time; time.sleep(10)", timeout=0.01)
        recovered = await self.run_cell("'answer' in globals()")

        self.assertTrue(timed_out.is_error)
        self.assertIn("timeout", timed_out.message)
        self.assertIn("state was preserved", timed_out.message)
        self.assertFalse(recovered.is_error)
        self.assertIn("True", recovered.output)

    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
    async def test_timeout_kills_worker_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ready_path = Path(temp_dir) / "child-ready"
            child_code = (
                "import os, signal, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                f"open({str(ready_path)!r}, 'w').write(str(os.getpid()))\n"
                "time.sleep(60)\n"
            )
            cell_code = (
                "import subprocess, sys\n"
                f"subprocess.run([sys.executable, '-c', {child_code!r}])"
            )

            timed_out = await self.run_cell(cell_code, timeout=0.5)
            child_pid = int(ready_path.read_text())
            try:
                for _ in range(20):
                    if not self._process_exists(child_pid):
                        break
                    await asyncio.sleep(0.05)
                self.assertFalse(self._process_exists(child_pid))
            finally:
                if self._process_exists(child_pid):
                    os.kill(child_pid, signal.SIGKILL)

            self.assertTrue(timed_out.is_error)
            self.assertIn("timeout", timed_out.message)

    @staticmethod
    def _process_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    def test_loop_context_counts_latest_request_once(self) -> None:
        totals = TokenTotals()
        totals.add(TokenUsage(input_other=10, input_cache_read=90, output=5))
        totals.add(TokenUsage(input_other=5, input_cache_read=120, output=10))

        self.assertEqual(135, totals.loop_context_tokens)
        self.assertEqual(240, totals.total)

    def test_system_prompt_does_not_include_loop_count(self) -> None:
        prompt = _system_prompt("/workspace")

        self.assertNotIn("loop resets so far", prompt)

    def test_new_loop_keeps_only_the_handoff_call_and_result(self) -> None:
        python_call = ToolCall(
            id="python-call",
            function=ToolCall.FunctionBody(name="python", arguments='{"code":"1 + 1"}'),
        )
        handoff_call = ToolCall(
            id="handoff-call",
            function=ToolCall.FunctionBody(
                name="start_new_loop",
                arguments='{"code":"next_step = \'run tests\'"}',
            ),
        )
        results = [
            ToolResult(tool_call_id="python-call", return_value=ToolOk(output="2")),
            ToolResult(tool_call_id="handoff-call", return_value=ToolOk(output="done")),
        ]

        history = _new_loop_history(
            "Original task", [python_call, handoff_call], results
        )
        if history is None:
            self.fail("Successful handoff did not start a new loop")

        self.assertEqual(3, len(history))
        self.assertEqual("Original task", history[0].extract_text())
        self.assertEqual([handoff_call], history[1].tool_calls)
        self.assertEqual("handoff-call", history[2].tool_call_id)
        self.assertIn("Context reset complete", history[2].extract_text())
        self.assertIn("done", history[2].extract_text())

    def test_failed_handoff_does_not_start_a_new_loop(self) -> None:
        handoff_call = ToolCall(
            id="handoff-call",
            function=ToolCall.FunctionBody(name="start_new_loop", arguments="{}"),
        )
        result = ToolResult(
            tool_call_id="handoff-call",
            return_value=ToolError(message="failed", brief="failed"),
        )

        self.assertIsNone(_new_loop_history("Original task", [handoff_call], [result]))

    async def test_assistant_text_prints_before_tool_execution(self) -> None:
        call = ToolCall(
            id="python-call",
            function=ToolCall.FunctionBody(name="python", arguments='{"code":"1 + 1"}'),
        )
        result = ToolResult(tool_call_id=call.id, return_value=ToolOk(output="2"))

        def handle(call):
            print("[tool executed]")
            return result

        steps = [
            SimpleNamespace(
                usage=TokenUsage(input_other=0, output=0),
                message=Message(
                    role="assistant", content="Checking.", tool_calls=[call]
                ),
                tool_calls=[call],
            ),
            SimpleNamespace(
                usage=TokenUsage(input_other=0, output=0),
                message=Message(role="assistant", content="Done."),
                tool_calls=[],
            ),
        ]

        output = StringIO()
        with patch("lazarus.cli.kosong.generate", side_effect=steps):
            with redirect_stdout(output):
                await run_request(
                    object(),
                    SimpleNamespace(tools=[], handle=handle),
                    SimpleNamespace(cwd="/workspace"),
                    [],
                    "Task",
                    TokenTotals(),
                    250_000,
                )

        text = output.getvalue()
        self.assertLess(text.index("Checking."), text.index("[tool executed]"))
        self.assertEqual(1, text.count("Checking."))
        self.assertEqual(1, text.count("Done."))
