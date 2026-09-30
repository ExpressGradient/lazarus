"""Explicit image output for Python cells; pixels never travel through stdout."""

import base64
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from io import BytesIO
import os
import math
from pathlib import Path
import tempfile

from PIL import Image, ImageOps
from kosong.message import ImageURLPart

MAX_IMAGES = 8
MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_PIXELS = 25_000_000
MAX_EDGE = 2048


@dataclass
class _Output:
    directory: Path | None
    count: int = 0


_output: ContextVar[_Output | None] = ContextVar("image_output", default=None)


@contextmanager
def image_output(output_path: str):
    output = _Output(Path(output_path).resolve().with_suffix(".images"))
    token = _output.set(output)
    try:
        yield
    finally:
        output.directory = None  # Also disable inherited contexts from old tasks.
        _output.reset(token)


def _check_size(width: float, height: float) -> None:
    if (
        not math.isfinite(width * height)
        or width <= 0
        or height <= 0
        or width * height > MAX_PIXELS
    ):
        raise ValueError(f"Image must contain between 1 and {MAX_PIXELS:,} pixels.")


def show_image(value) -> None:
    """Show a local path, encoded bytes, PIL image, or matplotlib figure to the model."""
    output = _output.get()
    if output is None or output.directory is None:
        raise RuntimeError("show_image() is only available during a Python cell.")
    if output.count >= MAX_IMAGES:
        raise ValueError(f"At most {MAX_IMAGES} images can be shown per cell.")

    if isinstance(value, Image.Image):
        _check_size(*value.size)
        image = ImageOps.exif_transpose(value)
    else:
        if isinstance(value, (str, os.PathLike)):
            path = Path(value).expanduser()
            with path.open("rb") as file:
                data = file.read(MAX_INPUT_BYTES + 1)
        elif isinstance(value, (bytes, bytearray, memoryview)):
            if len(value) > MAX_INPUT_BYTES:
                raise ValueError("Image input exceeds 20 MiB.")
            data = bytes(value)
        elif callable(getattr(value, "savefig", None)) and callable(
            getattr(value, "get_size_inches", None)
        ):
            width, height = value.get_size_inches()
            _check_size(width * 100, height * 100)
            buffer = BytesIO()
            value.savefig(buffer, format="png", dpi=100)
            data = buffer.getvalue()
        else:
            raise TypeError(
                "Expected an image path, image bytes, PIL image, or matplotlib figure."
            )
        if len(data) > MAX_INPUT_BYTES:
            raise ValueError("Image input exceeds 20 MiB.")
        with Image.open(BytesIO(data)) as source:
            _check_size(*source.size)
            image = ImageOps.exif_transpose(source)
            image.load()

    image = image.convert(
        "RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB"
    )
    image.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
    while True:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        if len(data) <= MAX_IMAGE_BYTES:
            break
        image = image.resize(
            (max(1, image.width * 3 // 4), max(1, image.height * 3 // 4)),
            Image.Resampling.LANCZOS,
        )

    directory = output.directory
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"{output.count + 1:02d}.png"
    fd, temporary = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    output.count += 1
    print(f"Image shown: {path} · {image.width}×{image.height}", flush=True)


def load_images(output_path: Path) -> list[ImageURLPart]:
    """Load completed artifacts once, including those from interrupted cells."""
    images = []
    paths = sorted(output_path.with_suffix(".images").glob("*.png"))
    if len(paths) > MAX_IMAGES:
        raise ValueError("Too many image artifacts.")
    for path in paths:
        with path.open("rb") as file:
            data = file.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f"Image artifact exceeds 4 MiB: {path}")
        images.append(
            ImageURLPart(
                image_url=ImageURLPart.ImageURL(
                    url="data:image/png;base64," + base64.b64encode(data).decode(),
                    id=str(path),
                )
            )
        )
    return images
