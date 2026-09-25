from contextlib import redirect_stdout
from io import StringIO
import json
from os import terminal_size
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kosong.chat_provider import TokenUsage
from kosong.tooling import ToolOk

from lazarus.cli import (
    CellParams,
    CellTool,
    JobParams,
    JobTool,
    TokenTotals,
    _print_token_usage,
)
from lazarus.display import ToolDisplay
from lazarus.jobs import Jobs
from lazarus.runtime import PythonRuntime


class DisplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_compact_status_keeps_full_results_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = PythonRuntime()
            jobs = Jobs(runtime, Path(tmp) / "jobs")
            display = ToolDisplay()
            cell = CellTool(jobs, "python", "Run a cell", display)
            observer = JobTool(jobs, display)
            screen = StringIO()
            try:
                with redirect_stdout(screen):
                    result = await cell(
                        CellParams(
                            code="import time\nprint('First result\\n' + 'x'*500 + '\\nThird result\\nFULL_RESULT_MARKER')\ntime.sleep(.2)",
                            description="Read the browser instructions",
                        )
                    )
                    job_id = json.loads(result.output)["job_id"]
                    result = await observer(JobParams(id=job_id, wait=10))
                    await observer(JobParams(id=job_id))
                data = json.loads(result.output)
                self.assertIn("FULL_RESULT_MARKER", data["output"])
                self.assertIn(
                    "FULL_RESULT_MARKER", Path(data["output_path"]).read_text()
                )
                self.assertNotIn("FULL_RESULT_MARKER", screen.getvalue())
                self.assertIn("Read the browser instructions", screen.getvalue())
                self.assertIn("First result", screen.getvalue())
                self.assertIn("Third result", screen.getvalue())
                self.assertIn("…", screen.getvalue())
                self.assertNotIn("x" * 121, screen.getvalue())
                self.assertNotIn("import time", screen.getvalue())
                self.assertNotIn('"job_id"', screen.getvalue())
                self.assertEqual(screen.getvalue().count("completed"), 1)
                with redirect_stdout(screen):
                    failed = await cell(
                        CellParams(
                            code="raise ValueError('visible failure')",
                            description="Check failure reporting",
                            yield_after=5,
                        )
                    )
                self.assertTrue(failed.is_error)
                self.assertIn("visible failure", screen.getvalue())
                self.assertNotIn("Traceback", screen.getvalue())
                verbose = StringIO()
                with redirect_stdout(verbose):
                    full = ToolDisplay(verbose=True)
                    full.cell("python", "print('FULL_RESULT_MARKER')")
                    full.result(result)
                self.assertIn("[python]", verbose.getvalue())
                self.assertIn("FULL_RESULT_MARKER", verbose.getvalue())
                self.assertIn('"job_id"', verbose.getvalue())
            finally:
                await jobs.close()
                await runtime.close()

    def test_incremental_preview_deduplicates_reads_and_sanitizes_terminal_text(self):
        display = ToolDisplay()
        screen = StringIO()

        def result(status, output, cursor):
            return ToolOk(
                output=json.dumps(
                    dict(job_id="abc", status=status, output=output, cursor=cursor)
                )
            )

        with (
            redirect_stdout(screen),
            patch(
                "lazarus.display.shutil.get_terminal_size",
                return_value=terminal_size((60, 24)),
            ),
        ):
            display.cell("python", "print('fallback code')")
            display.result(result("running", "", 0), label="Read\ninstructions")
            display.result(
                result("running", "\x1b[31mfirst\x1b[0m\x00\n" + "x" * 200, 220)
            )
            display.result(
                result("running", "\x1b[31mfirst\x1b[0m\x00\n" + "x" * 200, 220)
            )
            display.result(result("running", "second", 226))
            display.result(result("completed", "", 226))
            display.result(result("completed", "", 226))
        rendered = screen.getvalue()
        self.assertIn("print('fallback code')", rendered)
        self.assertIn("Read instructions · completed", rendered)
        self.assertEqual(rendered.count("first"), 1)
        self.assertEqual(rendered.count("second"), 1)
        self.assertEqual(rendered.count("completed"), 1)
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\x00", rendered)
        self.assertIn("…", rendered)
        self.assertTrue(all(len(line) <= 60 for line in rendered.splitlines()))

    def test_latest_context_is_not_cumulative_usage(self):
        totals = TokenTotals()
        totals.add(TokenUsage(input_other=100, input_cache_read=200, output=10))
        totals.add(TokenUsage(input_other=50, input_cache_read=400, output=20))
        self.assertEqual(totals.context_tokens, 470)
        self.assertEqual(totals.input, 750)
        self.assertEqual(totals.input_cache_read, 600)
        screen = StringIO()
        with redirect_stdout(screen):
            _print_token_usage(totals, interactive=True)
        self.assertIn("470 tokens (last call)", screen.getvalue())
        self.assertIn("750 in (600 cached)", screen.getvalue())
        totals.add(None)
        self.assertIsNone(totals.context_tokens)


if __name__ == "__main__":
    unittest.main()
