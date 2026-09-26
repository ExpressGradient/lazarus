from contextlib import redirect_stdout
from datetime import date
from io import StringIO
from importlib.metadata import version
from pathlib import Path
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

    async def test_worker_input_does_not_consume_protocol(self) -> None:
        failed = await self.run_cell("input()")
        recovered = await self.run_cell("6 * 7")

        self.assertTrue(failed.is_error)
        self.assertIn("EOFError", failed.message)
        self.assertFalse(recovered.is_error)
        self.assertIn("42", recovered.output)

    def test_version(self) -> None:
        output = StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            build_parser().parse_args(["--version"])

        self.assertEqual(0, raised.exception.code)
        self.assertEqual(
            f"{build_parser().prog} {version('lazarus')}\n", output.getvalue()
        )

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

    def test_system_prompt_includes_current_date(self) -> None:
        prompt = _system_prompt("/workspace", current_date=date(2026, 9, 26))

        self.assertIn("Current date: 2026-09-26.", prompt)

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

    async def test_context_limit_adds_one_handoff_request(self) -> None:
        call = ToolCall(
            id="python-call",
            function=ToolCall.FunctionBody(name="python", arguments='{"code":"pass"}'),
        )
        result = ToolResult(tool_call_id=call.id, return_value=ToolOk(output="done"))
        histories = []

        async def generate(**kwargs):
            histories.append(list(kwargs["history"]))
            tool_calls = [call] if len(histories) == 1 else []
            return SimpleNamespace(
                usage=TokenUsage(input_other=10, output=1),
                message=Message(role="assistant", content="", tool_calls=tool_calls),
            )

        totals = TokenTotals()
        with (
            patch("lazarus.cli.kosong.generate", side_effect=generate),
            redirect_stdout(StringIO()),
        ):
            await run_request(
                object(),
                SimpleNamespace(tools=[], handle=lambda _: result),
                SimpleNamespace(cwd="/workspace"),
                [],
                "Task",
                totals,
                5,
            )

        steer = [
            message.extract_text()
            for message in histories[1]
            if "start_new_loop" in message.extract_text()
        ]
        self.assertEqual(1, len(steer))
        self.assertIn("at least 5 tokens", steer[0])
        self.assertTrue(totals.loop_steer_sent)

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
