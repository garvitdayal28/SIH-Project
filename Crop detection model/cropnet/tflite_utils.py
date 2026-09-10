"""
TFLite interpreter loading and top-k prediction.

Kept free of any TensorFlow import at module level so that predict.py stays
usable on a machine that only has the much smaller tflite-runtime installed --
which is the realistic situation if you ever want to run this on a Raspberry Pi
or a second laptop without a 600 MB TensorFlow install.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .preprocess import dequantize_output, load_image, to_model_input


def load_interpreter(model_path: str | Path):
    """Return a TFLite Interpreter, trying the lightest runtime available first."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run scripts/train.py then scripts/export_tflite.py first."
        )

    last_error = None
    for import_path in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter"):
        try:
            module = __import__(import_path, fromlist=["Interpreter"])
            return module.Interpreter(model_path=str(model_path))
        except ImportError as exc:
            last_error = exc

    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=str(model_path))
    except ImportError as exc:
        raise ImportError(
            "No TFLite runtime available. Install one of: tensorflow, "
            "ai-edge-litert, or tflite-runtime."
        ) from (last_error or exc)


class CropClassifier:
    """Thin wrapper that turns an image path into ranked (label, confidence)."""

    def __init__(self, model_path: str | Path, labels: list[str]):
        self.labels = labels
        self.interpreter = load_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        _, height, width, _ = self.input_details["shape"]
        self.image_size = int(height)
        if height != width:
            raise ValueError(f"Expected a square input, got {height}x{width}")

        num_outputs = int(self.output_details["shape"][-1])
        if num_outputs != len(labels):
            raise ValueError(
                f"Model outputs {num_outputs} classes but labels.txt lists "
                f"{len(labels)}. These must match -- the label file is what "
                f"maps an output index to a crop_id."
            )

    def predict_array(self, pixels: np.ndarray) -> np.ndarray:
        """Run inference on a uint8 HWC image, returning a probability vector."""
        tensor = to_model_input(pixels, self.input_details)
        self.interpreter.set_tensor(self.input_details["index"], tensor)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output_details["index"])
        return dequantize_output(raw, self.output_details)[0]

    def predict_file(self, image_path: str | Path) -> np.ndarray:
        pixels = load_image(image_path, self.image_size)
        return self.predict_array(pixels)

    def top_k(self, probabilities: np.ndarray, k: int) -> list[tuple[str, float]]:
        k = min(k, len(self.labels))
        # argsort ascending, take the tail, reverse -- cheaper than a full sort
        # of a vector we only need the top of.
        indices = np.argsort(probabilities)[-k:][::-1]
        return [(self.labels[i], float(probabilities[i])) for i in indices]
