from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from lazarus.python_worker import create_shell, execute_cell


class PythonWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shell = create_shell()

    @classmethod
    def tearDownClass(cls) -> None:
        from IPython.core.async_helpers import get_asyncio_loop

        get_asyncio_loop().close()
        cls.shell.clear_instance()

    def execute(self, code):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cell.log"
            result = execute_cell(self.shell, code, str(output))
            self.assertNotIn("stdout", result)
            return {**result, "stdout": output.read_text()}

    def test_state_and_last_expression_persist(self) -> None:
        name = f"value_{uuid4().hex}"
        first = self.execute(f"{name} = 40")
        second = self.execute(f"{name} + 2")

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertIn("42", second["stdout"])

    def test_history_stays_in_memory(self) -> None:
        history = self.shell.history_manager
        if history is None:
            self.fail("IPython history manager is missing")
        self.assertEqual(":memory:", history.hist_file)

    def test_cell_history_does_not_grow(self) -> None:
        before = len(self.shell.user_ns["_ih"])
        self.execute("'not retained in input history'")
        self.assertEqual(before, len(self.shell.user_ns["_ih"]))

    def test_captures_python_and_subprocess_output(self) -> None:
        code = (
            "import subprocess, sys\n"
            "print('python-output')\n"
            "subprocess.run([sys.executable, '-c', \"print('child-output')\"])"
        )
        result = self.execute(code)

        self.assertTrue(result["ok"])
        self.assertIn("python-output", result["stdout"])
        self.assertIn("child-output", result["stdout"])

    def test_top_level_await(self) -> None:
        result = self.execute(
            "import asyncio\nawait asyncio.sleep(0)\n6 * 7",
        )

        self.assertTrue(result["ok"])
        self.assertIn("42", result["stdout"])

    def test_errors_do_not_kill_the_shell(self) -> None:
        failed = self.execute("raise RuntimeError('boom')")
        recovered = self.execute("21 * 2")

        self.assertFalse(failed["ok"])
        self.assertIn("RuntimeError", failed.get("error", ""))
        self.assertTrue(recovered["ok"])
        self.assertIn("42", recovered["stdout"])

    def test_large_output_stays_in_log_and_error_metadata_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cell.log"
            result = execute_cell(
                self.shell,
                "print('x' * 100000); raise ValueError('e' * 100000)",
                str(output),
            )
            self.assertFalse(result["ok"])
            self.assertNotIn("stdout", result)
            self.assertLessEqual(len(result["error"]), 4096)
            self.assertGreater(output.stat().st_size, 200000)
