import asyncio
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from kosong.chat_provider import TokenUsage
from kosong.message import Message, ToolCall

from lazarus.cli import run
from lazarus.skills import discover_skills, skills_prompt


class SkillTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.cwd = self.repo / "nested"
        self.cwd.mkdir(parents=True)
        (self.repo / ".git").write_text("gitdir: elsewhere")
        self.home = self.root / "home"
        self.home.mkdir()
        self.home_patch = patch("lazarus.skills.Path.home", return_value=self.home)
        self.home_patch.start()
        self.addCleanup(self.home_patch.stop)

    def skill(self, root, name, description="Useful workflow", extra=""):
        path = root / ".agents/skills" / name / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"---\nname: {name}\ndescription: {description}\n{extra}---\nBODY_MARKER"
        )
        return path

    def test_precedence_symlinks_and_repository_boundary(self):
        nearest = self.skill(self.cwd, "shared", "Nearest definition")
        self.skill(self.repo, "shared", "Parent definition")
        self.skill(self.home, "shared", "Global definition")
        self.skill(self.root, "outside-repo")
        self.skill(self.repo, "parent-only")
        global_path = self.skill(self.home, "global-only")
        links = self.cwd / ".agents/skills"
        (links / "linked").symlink_to(global_path.parent, target_is_directory=True)
        (links / "cycle").symlink_to(links, target_is_directory=True)
        skills, warnings = discover_skills(self.cwd)
        by_name = {s.name: s for s in skills}
        self.assertEqual(set(by_name), {"shared", "parent-only", "global-only"})
        self.assertEqual(by_name["shared"].path, nearest.resolve())
        self.assertEqual(len(warnings), 2)

    def test_yaml_errors_hidden_overrides_and_metadata_only(self):
        self.skill(self.cwd, "disabled", extra="disable-model-invocation: true\n")
        self.skill(self.home, "disabled", "Must not override disabled project skill")
        self.skill(self.cwd, "folded", ">\n  Use for <xml> &\n  multiline tasks.")
        malformed = self.skill(self.cwd, "broken")
        malformed.write_text("---\nname: [unterminated\n---\n")
        self.skill(self.cwd, "invalid-type", "42")
        loop = self.cwd / ".agents/skills/bad-link/SKILL.md"
        loop.parent.mkdir()
        loop.symlink_to(loop)
        body, warnings = skills_prompt(self.cwd)
        self.assertIn('name="folded"', body)
        self.assertIn("&lt;xml&gt; &amp; multiline", body)
        self.assertNotIn('name="disabled"', body)
        self.assertNotIn("BODY_MARKER", body)
        self.assertEqual(len(warnings), 4)
        with patch("lazarus.skills.MAX_CATALOG_CHARS", 1):
            body, warnings = skills_prompt(self.cwd)
        self.assertEqual(body, "")
        self.assertTrue(any("budget" in w for w in warnings))

    def test_catalog_stays_fixed_across_turns_reset_and_resume(self):
        path = self.skill(self.cwd, "stable", "Original description")
        calls = []

        async def generate(**kwargs):
            calls.append(kwargs["system_prompt"])
            tool_calls = []
            if len(calls) == 1:
                path.write_text("broken after session starts")
                tool_calls = [
                    ToolCall(
                        id="reset",
                        function=ToolCall.FunctionBody(
                            name="start_new_loop", arguments='{"code":"pass"}'
                        ),
                    )
                ]
            return SimpleNamespace(
                message=Message(
                    role="assistant", content="done", tool_calls=tool_calls
                ),
                tool_calls=tool_calls,
                usage=TokenUsage(input_other=100, input_cache_read=200, output=10),
            )

        chat = SimpleNamespace(name="test", model_name="test")
        session = str(self.root / "session")
        with (
            patch("os.getcwd", return_value=str(self.cwd)),
            patch("lazarus.cli.kosong.generate", side_effect=generate),
            patch("builtins.input", side_effect=["first", "second", "/quit"]),
            redirect_stdout(StringIO()),
        ):
            asyncio.run(run(chat, None, 150_000, 48, session_dir=session))
            with patch(
                "lazarus.cli._system_prompt", side_effect=AssertionError("rescan")
            ):
                asyncio.run(run(chat, "third", 150_000, 48, resume=session))
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(set(calls)), 1)
        self.assertIn("Original description", calls[0])
        events = [
            json.loads(s)
            for s in (Path(session) / "journal.jsonl").read_text().splitlines()
        ]
        self.assertEqual(sum(e["event"] == "reset" for e in events), 1)
        self.assertEqual(sum(e["event"] == "usage" for e in events), 4)


if __name__ == "__main__":
    unittest.main()
