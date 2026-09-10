"""
Quantize the trained model to int8 TFLite for the ESP32.

    python scripts/export_tflite.py

Full integer quantization -- weights *and* activations become int8, and the
input/output tensors are int8 too. This is not an optimisation, it is a
requirement: TFLite Micro on an ESP32 has no float acceleration, and a float
model would be roughly 4x the flash and many times slower.

Two things this script does that are easy to get wrong:

1. It strips the augmentation layers. RandomFlip and friends exist only for
   training; leaving them in the graph either bloats it or breaks conversion.

2. It feeds the converter a representative dataset of *raw* pixel values in
   [0, 255], matching the Rescaling layer that lives inside the model. That is
   what makes the exported input tensor take raw camera bytes, so the ESP32
   firmware's entire preprocessing step is `pixel - 128`.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from cropnet.preprocess import load_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPRESENTATIVE_SAMPLES = 300


def build_inference_model(tf, trained):
    """Rebuild the trained model without its training-only layers.

    Walks the functional graph in order and re-applies every layer except the
    augmentation block, reusing the trained weights.
    """
    inputs = tf.keras.Input(shape=config.INPUT_SHAPE, name="image")
    x = inputs

    for layer in trained.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if layer.name == "augmentation":
            continue
        if isinstance(layer, tf.keras.layers.Dropout):
            # Identity at inference; dropping it keeps the graph smaller.
            continue
        x = layer(x)

    model = tf.keras.Model(inputs, x, name="farmfrost_crop_classifier_inference")
    print(f"Inference graph: {len(model.layers)} layers "
          f"(stripped augmentation and dropout)")
    return model


def representative_image_paths(rng: random.Random) -> list[Path]:
    """Sample training images evenly across classes.

    Even sampling matters here. The converter derives every activation's scale
    and zero-point from what it sees; a sample dominated by one class would
    calibrate the ranges around that class and cost accuracy on the others.
    """
    train_dir = config.DATA_PREPARED / "train"
    if not train_dir.exists():
        raise SystemExit(f"Missing {train_dir}. Run scripts/prepare_data.py first.")

    per_class = {}
    for class_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        images = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if images:
            rng.shuffle(images)
            per_class[class_dir.name] = images

    if not per_class:
        raise SystemExit(f"No images found under {train_dir}.")

    quota = max(1, REPRESENTATIVE_SAMPLES // len(per_class))
    chosen = []
    for images in per_class.values():
        chosen.extend(images[:quota])

    rng.shuffle(chosen)
    return chosen


def make_representative_dataset(paths: list[Path]):
    def generator():
        for path in paths:
            pixels = load_image(path, config.IMAGE_SIZE)
            # float32 in [0, 255]: the model's own Rescaling layer normalises.
            yield [np.expand_dims(pixels.astype(np.float32), axis=0)]
    return generator


def describe_tensor(interpreter, details, role: str) -> None:
    scale, zero_point = details["quantization"]
    print(f"  {role:<7} name={details['name']!r}")
    print(f"          shape={list(details['shape'])} dtype={np.dtype(details['dtype']).name}")
    print(f"          scale={scale:.8f} zero_point={zero_point}")


def main() -> None:
    import tensorflow as tf

    rng = random.Random(config.SEED)

    if not config.KERAS_MODEL.exists():
        raise SystemExit(
            f"Missing {config.KERAS_MODEL}. Run scripts/train.py first."
        )

    print(f"Loading {config.KERAS_MODEL}")
    trained = tf.keras.models.load_model(config.KERAS_MODEL)
    inference_model = build_inference_model(tf, trained)

    paths = representative_image_paths(rng)
    print(f"Calibrating on {len(paths)} representative training images")

    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(paths)

    # Refuse to fall back to float ops. If the converter cannot express
    # something in int8 we want a loud failure here, on the desktop, rather than
    # an unsupported-op crash on the ESP32 at 3 AM before the demo.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # The converter quantizes Dense/FullyConnected weights per-channel by
    # default. TFLite Micro's fully_connected kernel only implements per-tensor
    # for the filter -- fully_connected_common.cpp asserts scale->size == 1 --
    # so a per-channel head makes AllocateTensors() fail on the board with:
    #
    #   FullyConnected per-channel quantization not yet supported.
    #   Node FULLY_CONNECTED (number 30f) failed to prepare with status 1
    #   [CV] AllocateTensors failed -- raise CROP_ARENA_BYTES
    #
    # That last line is a red herring; the arena is fine. Conv2D and
    # DepthwiseConv2D keep per-channel either way -- TFLM supports those -- so
    # this only affects the 9-class classifier head, where the per-channel
    # scales span about 1.3x and collapsing them costs no measurable accuracy.
    converter._experimental_disable_per_channel_quantization_for_dense_layers = True

    print("Converting...")
    tflite_bytes = converter.convert()

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.TFLITE_MODEL.write_bytes(tflite_bytes)

    size = len(tflite_bytes)
    print(f"\nWrote {config.TFLITE_MODEL}")
    print(f"Size: {size:,} bytes ({size / 1024:.1f} KB)")

    budget = config.MAX_MODEL_BYTES
    if size > budget:
        print(
            f"\nWARNING: over the {budget / 1024:.0f} KB flash budget.\n"
            "  Set BACKBONE='mobilenet_v1' and ALPHA=0.25 in config.py and retrain\n"
            "  for a roughly 220 KB model."
        )
    else:
        print(f"Within the {budget / 1024:.0f} KB flash budget "
              f"({100 * size / budget:.0f}% used).")

    # Report the exact tensor quantization parameters. The firmware needs the
    # input scale/zero_point to convert camera bytes, and the output ones to
    # turn the result back into a confidence percentage.
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    print("\nTensor details -- the firmware needs these:")
    describe_tensor(interpreter, interpreter.get_input_details()[0], "input")
    describe_tensor(interpreter, interpreter.get_output_details()[0], "output")

    input_details = interpreter.get_input_details()[0]
    scale, zero_point = input_details["quantization"]
    if scale and abs(scale - 1.0) < 0.01 and zero_point == -128:
        print("\n  Input is raw pixels offset by -128, so the ESP32 preprocessing is:")
        print("      input[i] = (int8_t)(rgb_byte[i] - 128);")
    else:
        print("\n  Input needs: int8 = round(pixel / scale) + zero_point, "
              "using the values above.")

    # An upper bound on the arena, not the real figure: this sums every
    # intermediate tensor, whereas TFLite Micro reuses buffers whose lifetimes
    # do not overlap and typically needs a fraction of this.
    upper_bound = sum(
        int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize
        for t in interpreter.get_tensor_details()
        if len(t["shape"]) > 0
    )
    print(f"\nTensor-arena upper bound: {upper_bound / 1024:.0f} KB "
          f"(actual need is much lower -- buffers are reused).")
    print("  Start the ESP32 arena at 300 KB in PSRAM, then call "
          "arena_used_bytes()\n  after AllocateTensors() and shrink to the "
          "reported figure plus headroom.")

    print("\nDone. Next: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
