
import cv2
import numpy as np

from PIL import Image
from rembg import remove, new_session


def preprocess_sketch(
    input_path: str,
    output_path: str,
    size: int = 512,
    threshold: int = 180
) -> Image.Image:
    """
    Preprocess a 2D sketch for Scribble ControlNet.

    - Converts sketch to grayscale.
    - Resizes to 512x512.
    - Preserves black outlines on white background.
    """

    sketch = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if sketch is None:
        raise FileNotFoundError(
            f"Cannot read sketch: {input_path}"
        )

    sketch = cv2.resize(
        sketch,
        (size, size),
        interpolation=cv2.INTER_AREA
    )

    _, binary = cv2.threshold(
        sketch,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    processed = Image.fromarray(binary).convert("RGB")

    processed.save(output_path)

    return processed


def remove_background(
    input_path: str,
    output_path: str,
    foreground_scale: float = 0.85,
    model_name: str = "u2net"
) -> Image.Image:
    """
    Remove background using U2Net.

    Returns a centered 512x512 RGBA image
    with transparent background.
    """

    image = Image.open(input_path).convert("RGB")

    session = new_session(
        model_name,
        providers=["CPUExecutionProvider"]
    )

    result = remove(
        image,
        session=session
    ).convert("RGBA")

    bbox = result.getchannel("A").getbbox()

    if bbox is None:
        raise ValueError("No foreground object detected.")

    foreground = result.crop(bbox)

    canvas_size = 512
    max_size = int(canvas_size * foreground_scale)

    foreground.thumbnail(
        (max_size, max_size),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "RGBA",
        (canvas_size, canvas_size),
        (0, 0, 0, 0)
    )

    x = (canvas_size - foreground.width) // 2
    y = (canvas_size - foreground.height) // 2

    canvas.alpha_composite(
        foreground,
        (x, y)
    )

    canvas.save(output_path)

    return canvas
