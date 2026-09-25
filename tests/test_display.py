from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from kosong.chat_provider import TokenUsage

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
                            code="import time\nprint('FULL_RESULT_MARKER')\ntime.sleep(.2)"
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
                self.assertNotIn('"job_id"', screen.getvalue())
                self.assertEqual(screen.getvalue().count("completed"), 1)
                with redirect_stdout(screen):
                    failed = await cell(
                        CellParams(
                            code="raise ValueError('visible failure')", yield_after=5
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
