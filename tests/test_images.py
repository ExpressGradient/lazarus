"""Vision output without network calls: artifacts, jobs, history, and recovery."""

import asyncio
import base64
from contextlib import redirect_stdout
from contextvars import copy_context
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from kosong.message import ImageURLPart, Message, ToolCall
from kosong.tooling import ToolError, ToolOk, ToolResult
from kosong.tooling.simple import SimpleToolset

from lazarus.cli import (
    CellTool,
    JobTool,
    TokenTotals,
    _image_message,
    _new_loop_history,
    run_request,
)
from lazarus.display import ToolDisplay, result_text
from lazarus.images import MAX_IMAGES, image_output, load_images, show_image
from lazarus.jobs import Jobs
from lazarus.runtime import PythonRuntime
from lazarus.session import Session


def png(color="red", size=(20, 10)):
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def image_parts(value):
    return (
        [part for part in value.output if isinstance(part, ImageURLPart)]
        if isinstance(value.output, list)
        else []
    )


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / "cell.log"

    def emit(self, value):
        with image_output(str(self.log)), redirect_stdout(StringIO()) as terminal:
            show_image(value)
        return load_images(self.log), terminal.getvalue()

    def test_paths_bytes_pil_and_figure(self):
        source = self.root / "source.png"
        source.write_bytes(png())

        class Figure:
            def get_size_inches(self):
                return (2, 1)

            def savefig(self, file, **kwargs):
                self.kwargs = kwargs
                file.write(png())

        figure = Figure()
        with image_output(str(self.log)), redirect_stdout(StringIO()):
            for value in [
                str(source),
                source,
                png(),
                bytearray(png()),
                memoryview(png()),
                Image.new("RGB", (20, 10), "red"),
                figure,
            ]:
                show_image(value)
        images = load_images(self.log)
        self.assertEqual(7, len(images))
        self.assertEqual({"format": "png", "dpi": 100}, figure.kwargs)
        for image in images:
            path = Path(image.image_url.id)
            self.assertEqual(0o600, path.stat().st_mode & 0o777)
            pixels = base64.b64decode(image.image_url.url.split(",", 1)[1])
            with Image.open(BytesIO(pixels)) as decoded:
                self.assertEqual((20, 10), decoded.size)
                self.assertEqual((255, 0, 0), decoded.getpixel((0, 0)))
        self.assertEqual(0o700, self.log.with_suffix(".images").stat().st_mode & 0o777)

    def test_snapshot_is_independent_and_terminal_has_no_pixels(self):
        source = self.root / "source.png"
        source.write_bytes(png())
        images, text = self.emit(source)
        source.write_bytes(png("blue"))
        self.assertEqual(images, load_images(self.log))
        self.assertIn("Image shown:", text)
        self.assertIn("20×10", text)
        self.assertNotIn("base64", text)

    def test_resize_orientation_and_byte_limit(self):
        source = Image.new("RGB", (3000, 1000))
        images, _ = self.emit(source)
        with Image.open(Path(images[0].image_url.id)) as image:
            self.assertEqual(2048, image.width)
            self.assertAlmostEqual(3, image.width / image.height, places=2)
        # EXIF orientation must not leave a sideways screenshot.
        source = Image.new("RGB", (20, 10))
        source.getexif()[274] = 6
        self.log = self.root / "oriented.log"
        images, _ = self.emit(source)
        with Image.open(Path(images[0].image_url.id)) as image:
            self.assertEqual((10, 20), image.size)
        self.log = self.root / "bounded.log"
        with patch("lazarus.images.MAX_IMAGE_BYTES", 100):
            images, _ = self.emit(Image.new("RGB", (500, 500)))
        self.assertLessEqual(Path(images[0].image_url.id).stat().st_size, 100)

    def test_invalid_inputs_limits_and_inactive_context(self):
        with self.assertRaisesRegex(RuntimeError, "during a Python cell"):
            show_image(png())
        with image_output(str(self.log)), redirect_stdout(StringIO()):
            for value, error in [
                (object(), TypeError),
                (b"not an image", OSError),
                (self.root / "missing", FileNotFoundError),
            ]:
                with self.subTest(value=value), self.assertRaises(error):
                    show_image(value)
            with (
                patch("lazarus.images.MAX_INPUT_BYTES", 10),
                self.assertRaisesRegex(ValueError, "20 MiB"),
            ):
                show_image(png())
            with (
                patch("lazarus.images.MAX_PIXELS", 1),
                self.assertRaisesRegex(ValueError, "pixels"),
            ):
                show_image(Image.new("RGB", (20, 10)))
            for _ in range(MAX_IMAGES):
                show_image(png())
            with self.assertRaisesRegex(ValueError, "At most"):
                show_image(png())
            inherited = copy_context()
        with self.assertRaises(RuntimeError):
            inherited.run(show_image, png())
        self.assertEqual(MAX_IMAGES, len(load_images(self.log)))

    def test_display_and_handoff_preserve_only_explicit_images(self):
        images, _ = self.emit(png())
        data = {
            "job_id": "job",
            "status": "failed",
            "output": "",
            "cursor": 0,
            "images": [images[0].image_url.id],
        }
        result = ToolError(
            output=[Message(role="user", content=json.dumps(data)).content[0], *images],
            message="boom",
            brief="Failed",
        )
        with redirect_stdout(StringIO()) as terminal:
            ToolDisplay().result(result)
            ToolDisplay(verbose=True).result(result)
        self.assertIn("failed", terminal.getvalue())
        self.assertIn("Image ·", terminal.getvalue())
        self.assertNotIn("data:image", terminal.getvalue())
        self.assertNotIn("base64", result_text(result))
        call = ToolCall(
            id="handoff",
            function=ToolCall.FunctionBody(name="start_new_loop", arguments="{}"),
        )
        history = _new_loop_history(
            "task",
            [call],
            [
                ToolResult(
                    tool_call_id=call.id, return_value=ToolOk(output=result.output)
                )
            ],
        )
        self.assertEqual(4, len(history))
        self.assertIsInstance(history[-1].content[-1], ImageURLPart)
        self.assertIsNone(history[-1].content[-1].image_url.id)


class ImageJobTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime = PythonRuntime()
        self.jobs = Jobs(self.runtime, self.root / "jobs", tool_output_limit_kib=1)
        self.source = self.root / "source.png"
        self.source.write_bytes(png())
        self.code = f"show_image({str(self.source)!r})"

    async def asyncTearDown(self):
        await self.jobs.close()
        await self.runtime.close()
        self.temp.cleanup()

    async def test_worker_artifacts_delivered_once_despite_log_truncation(self):
        result = await self.jobs.submit(self.code + "\nprint('x' * 10000)", 10, 5)
        self.assertFalse(result.is_error, result)
        images = image_parts(result)
        self.assertEqual(1, len(images))
        data = json.loads(result_text(result))
        self.assertIn("omitted", data["output"])
        self.assertEqual([images[0].image_url.id], data["images"])
        for cursor in (None, 0):
            reread = await self.jobs.inspect(data["job_id"], cursor=cursor)
            self.assertFalse(image_parts(reread))
        self.assertFalse(self.jobs.notifications())
        # Starting another cell resets the per-cell limit and output destination.
        second = await self.jobs.submit(self.code, 10, 5)
        self.assertEqual(1, len(image_parts(second)))
        self.assertNotEqual(images[0].image_url.id, image_parts(second)[0].image_url.id)

    async def test_images_survive_failure_cancellation_and_worker_crash(self):
        for suffix, status in [
            ("raise ValueError('boom')", "failed"),
            ("import os; os._exit(1)", "lost"),
        ]:
            with self.subTest(status=status):
                result = await self.jobs.submit(self.code + "\n" + suffix, 10, 5)
                self.assertTrue(result.is_error)
                self.assertEqual(1, len(image_parts(result)))
                self.assertEqual(
                    status,
                    json.loads(result_text(result, include_message=False))["status"],
                )
        started = await self.jobs.submit(
            self.code + "\nimport time; time.sleep(30)", 10, 0
        )
        job_id = json.loads(started.output)["job_id"]
        async with asyncio.timeout(5):
            while "Image shown:" not in self.jobs.jobs[job_id].output_path.read_text():
                await asyncio.sleep(0.01)
        self.assertFalse(image_parts(await self.jobs.inspect(job_id)))
        result = await self.jobs.inspect(job_id, cancel=True, wait=5)
        self.assertEqual(1, len(image_parts(result)))
        self.assertTrue(result.is_error)

    async def test_background_notifications_deliver_images_once(self):
        started = await self.jobs.submit(self.code, 10, 0)
        job_id = json.loads(started.output)["job_id"]
        await self.jobs.jobs[job_id].task
        notices = self.jobs.notifications()
        self.assertEqual(1, len(image_parts(notices[0])))
        self.assertFalse(self.jobs.notifications())
        self.assertFalse(image_parts(await self.jobs.inspect(job_id)))

    async def test_agent_receives_images_after_all_tool_results_and_resume(self):
        for delay in (False, True):
            with self.subTest(background=delay):
                histories = []

                async def generate(**kwargs):
                    histories.append(list(kwargs["history"]))
                    calls = []
                    if len(histories) == 1:
                        calls = [
                            ToolCall(
                                id="image",
                                function=ToolCall.FunctionBody(
                                    name="python",
                                    arguments=json.dumps(
                                        {
                                            "code": self.code
                                            + (
                                                "\nimport time; time.sleep(.1)"
                                                if delay
                                                else ""
                                            ),
                                            "yield_after": 0 if delay else 5,
                                        }
                                    ),
                                ),
                            ),
                            ToolCall(
                                id="list",
                                function=ToolCall.FunctionBody(
                                    name="job", arguments="{}"
                                ),
                            ),
                        ]
                    return SimpleNamespace(
                        usage=None,
                        message=Message(
                            role="assistant", content="done", tool_calls=calls
                        ),
                    )

                directory = self.root / f"session-{delay}"
                session = Session(str(directory))
                session.record("session", system_prompt="fixture", cwd=str(self.root))
                tools = SimpleToolset(
                    [CellTool(self.jobs, "python", "run"), JobTool(self.jobs)]
                )
                try:
                    with (
                        patch("lazarus.cli.kosong.generate", side_effect=generate),
                        redirect_stdout(StringIO()),
                    ):
                        history = await run_request(
                            object(),
                            tools,
                            self.runtime,
                            [],
                            "Inspect the screenshot",
                            TokenTotals(),
                            150000,
                            jobs=self.jobs,
                            session=session,
                        )
                finally:
                    session.close()
                images = [
                    (i, m)
                    for i, m in enumerate(history)
                    if any(isinstance(p, ImageURLPart) for p in m.content)
                ]
                self.assertEqual(1, len(images))
                index, image_message = images[0]
                self.assertEqual("user", image_message.role)
                self.assertGreater(
                    index, max(i for i, m in enumerate(history) if m.role == "tool")
                )
                self.assertTrue(
                    any(
                        any(isinstance(p, ImageURLPart) for p in m.content)
                        for m in histories[-1]
                    )
                )
                self.source.unlink()
                resumed = Session(str(directory), resume=True)
                try:
                    restored, *_ = resumed.restore()
                finally:
                    resumed.close()
                restored_images = [
                    m
                    for m in restored
                    if any(isinstance(p, ImageURLPart) for p in m.content)
                ]
                self.assertEqual([image_message], restored_images)
                self.source.write_bytes(png())

    async def test_openai_request_contains_native_image_content(self):
        from openai import AsyncOpenAI
        import httpx
        from lazarus.chatgpt import ChatGPT

        result = await self.jobs.submit(self.code, 10, 5)
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text='data: {"type":"response.completed","response":{"id":"r","usage":null}}\n\n',
            )

        provider = ChatGPT(model="fixture")
        await provider.close()
        provider.auth = SimpleNamespace(access_token=lambda: "fixture-token")
        provider._client = AsyncOpenAI(
            api_key="unused",
            base_url="https://api.openai.com/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        try:
            stream = await provider.generate("fixture", [], [_image_message([result])])
            _ = [part async for part in stream]
        finally:
            await provider.close()
        content = requests[0]["input"][0]["content"]
        self.assertEqual("input_image", content[-1]["type"])
        self.assertTrue(content[-1]["image_url"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
