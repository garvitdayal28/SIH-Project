"""
Evaluate the quantized model on the held-out test set.

    python scripts/evaluate.py

This runs the actual .tflite file -- the same bytes that go on the ESP32 -- not
the Keras model. That distinction is the whole point of this script: int8
quantization always costs some accuracy, and the only number worth quoting is
the one measured after it.

Reports top-1, top-5, per-class accuracy, a confusion matrix, and how often the
model would fall below the firmware's confidence threshold.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from cropnet.labels import read_labels
from cropnet.preprocess import load_image
from cropnet.tflite_utils import CropClassifier

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_test_set(labels: list[str]) -> list[tuple[Path, int]]:
    test_dir = config.DATA_PREPARED / "test"
    if not test_dir.exists():
        raise SystemExit(f"Missing {test_dir}. Run scripts/prepare_data.py first.")

    samples = []
    for index, label in enumerate(labels):
        class_dir = test_dir / label
        if not class_dir.exists():
            print(f"  warning: no test folder for class {label!r}")
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, index))

    if not samples:
        raise SystemExit(f"No test images found under {test_dir}.")
    return samples


def print_confusion(matrix: np.ndarray, labels: list[str]) -> None:
    """Print the confusion matrix as text.

    Rows are the true class, columns the prediction, so anything off the
    diagonal in row i is "what crop i gets mistaken for". That is the map you
    use to decide which class needs more training images.
    """
    short = [label[:6] for label in labels]
    width = max(len(label) for label in labels)

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    header = " " * (width + 2) + " ".join(f"{s:>6}" for s in short)
    print(header)
    for i, label in enumerate(labels):
        cells = " ".join(f"{matrix[i, j]:>6}" for j in range(len(labels)))
        print(f"{label:<{width}}  {cells}")


def main() -> None:
    labels = read_labels(config.LABELS_FILE)
    classifier = CropClassifier(config.TFLITE_MODEL, labels)
    samples = collect_test_set(labels)

    print(f"Evaluating {config.TFLITE_MODEL.name} on {len(samples)} test images")
    print(f"Model input: {classifier.image_size}x{classifier.image_size}, "
          f"{len(labels)} classes\n")

    num_classes = len(labels)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)

    top1_correct = 0
    top5_correct = 0
    below_threshold = 0
    confident_and_wrong = 0

    for position, (path, true_index) in enumerate(samples, start=1):
        probabilities = classifier.predict_file(path)
        ranking = np.argsort(probabilities)[::-1]

        predicted = int(ranking[0])
        confidence = float(probabilities[predicted])

        confusion[true_index, predicted] += 1
        per_class_total[true_index] += 1

        if predicted == true_index:
            top1_correct += 1
            per_class_correct[true_index] += 1
        elif confidence >= config.CONFIDENCE_THRESHOLD:
            # The dangerous case: wrong, but confident enough that the firmware
            # would act on it and change fan speed for the wrong crop.
            confident_and_wrong += 1

        if true_index in ranking[:config.TOP_K]:
            top5_correct += 1

        if confidence < config.CONFIDENCE_THRESHOLD:
            below_threshold += 1

        if position % 100 == 0:
            print(f"  {position}/{len(samples)}...")

    total = len(samples)
    top1 = top1_correct / total
    top5 = top5_correct / total

    print(f"\n{'=' * 58}")
    print(f"Top-1 accuracy          {top1:6.2%}  ({top1_correct}/{total})")
    print(f"Top-{config.TOP_K} accuracy          {top5:6.2%}  ({top5_correct}/{total})")
    print(f"{'=' * 58}")

    print("\nPer-class accuracy:")
    width = max(len(label) for label in labels)
    for index, label in enumerate(labels):
        count = per_class_total[index]
        if count == 0:
            print(f"  {label:<{width}}  (no test images)")
            continue
        accuracy = per_class_correct[index] / count
        bar = "#" * int(accuracy * 30)
        print(f"  {label:<{width}}  {accuracy:6.2%}  ({per_class_correct[index]:>3}/{count:<3}) {bar}")

    print_confusion(confusion, labels)

    print(f"\nAgainst the firmware confidence threshold "
          f"({config.CONFIDENCE_THRESHOLD:.0%}):")
    print(f"  below threshold, ignored by the Main ESP32   "
          f"{below_threshold:>4} ({below_threshold / total:.1%})")
    print(f"  wrong AND above threshold, would act on it   "
          f"{confident_and_wrong:>4} ({confident_and_wrong / total:.1%})")
    print("\n  The second number is the one that matters. It is how often the fan\n"
          "  would be set for the wrong crop. Raise CONFIDENCE_THRESHOLD in\n"
          "  config.py to trade responsiveness for safety.")

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "model": str(config.TFLITE_MODEL.name),
        "test_images": total,
        "top1_accuracy": top1,
        f"top{config.TOP_K}_accuracy": top5,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "below_threshold": below_threshold,
        "confident_and_wrong": confident_and_wrong,
        "labels": labels,
        "confusion_matrix": confusion.tolist(),
        "per_class_accuracy": {
            label: (per_class_correct[i] / per_class_total[i]) if per_class_total[i] else None
            for i, label in enumerate(labels)
        },
    }
    path = config.OUTPUTS_DIR / "int8_test_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
