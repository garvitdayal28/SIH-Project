"""
Image preprocessing.

This module is the single definition of "how an image becomes model input", and
it deliberately depends only on Pillow and numpy -- not TensorFlow -- so that
predict.py can run on a machine with just tflite-runtime installed.

The steps here mirror what the ESP32-CAM firmware will do to a camera frame:

    full frame  ->  centre-crop to a square  ->  resize to 96x96  ->  raw RGB888

Note what is *not* here: there is no mean subtraction or /127.5 normalisation.
That is baked into the model itself as a Rescaling layer, so the quantized
model's input tensor takes raw pixel bytes. On the ESP32 that means the firmware
can hand the camera buffer almost straight to the interpreter, which keeps the
device-side code small and removes a whole class of "the desktop and the board
disagree" bugs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def center_crop_to_square(image: Image.Image) -> Image.Image:
    """Crop the largest centred square out of an image.

    Centre-cropping rather than squashing matters: the camera sits above the
    tray looking at one item, so the centre of the frame is the subject and the
    edges are the tray. Squashing a 4:3 frame to a square would distort every
    shape the model relies on.
    """
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def load_image(path: str | Path, size: int) -> np.ndarray:
    """Load an image file and return it as a uint8 RGB array of (size, size, 3).

    Raises FileNotFoundError if the path does not exist, and PIL's
    UnidentifiedImageError if the file is not a readable image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such image: {path}")

    with Image.open(path) as image:
        # Phone photos carry an EXIF orientation flag; without this a portrait
        # photo arrives rotated 90 degrees and the model sees nonsense.
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = center_crop_to_square(image)
        image = image.resize((size, size), Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def to_model_input(pixels: np.ndarray, input_details: dict) -> np.ndarray:
    """Convert a uint8 HWC image into the batched tensor a TFLite model expects.

    Handles both the int8 model that ships to the ESP32 and a float32 model, so
    the same prediction code works before and after quantization.

    For the int8 model the conversion is just `pixel + zero_point` with a scale
    of ~1.0 -- in practice `pixel - 128`. That single subtraction is the entire
    preprocessing step the ESP32 firmware has to perform.
    """
    dtype = input_details["dtype"]
    batched = np.expand_dims(pixels, axis=0)

    if dtype == np.uint8:
        return batched.astype(np.uint8)

    if dtype == np.int8:
        scale, zero_point = input_details["quantization"]
        if scale == 0:
            # Unquantized int8 input should not happen, but fall back to the
            # plain offset rather than dividing by zero.
            return (batched.astype(np.int32) - 128).astype(np.int8)
        quantized = np.round(batched.astype(np.float32) / scale) + zero_point
        return np.clip(quantized, -128, 127).astype(np.int8)

    # float32 model: the Rescaling layer inside the model still does the
    # normalisation, so we only need to widen the type.
    return batched.astype(np.float32)


def dequantize_output(raw: np.ndarray, output_details: dict) -> np.ndarray:
    """Turn a model's raw output tensor into float probabilities."""
    dtype = output_details["dtype"]
    if dtype in (np.int8, np.uint8):
        scale, zero_point = output_details["quantization"]
        return (raw.astype(np.float32) - zero_point) * scale
    return raw.astype(np.float32)
