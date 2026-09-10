"""
Crop detection on an image file -- the stand-in for the ESP32-CAM.

    python predict.py path\\to\\image.jpg
    python predict.py path\\to\\folder            (every image in it)
    python predict.py image.jpg --json            (machine-readable)
    python predict.py image.jpg --esp-now         (show the packet that would be sent)

This runs models/model_int8.tflite -- byte for byte the model that will run on
the ESP32-CAM -- through the same centre-crop-and-resize preprocessing the
firmware will use. The numbers printed here are the numbers the board will
produce, give or take the difference between a JPEG file and an OV2640 frame.
"""

from __future__ import annotations

import os
import warnings

# Set before anything can import TensorFlow. The interpreter is loaded lazily in
# cropnet.tflite_utils, so this still lands in time -- and without it the tool
# prints half a screen of oneDNN and XNNPACK notices before every result.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from cropnet.labels import read_labels
from cropnet.tflite_utils import CropClassifier

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_images(target: Path) -> list[Path]:
    if target.is_dir():
        images = sorted(
            p for p in target.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise SystemExit(f"No images found in {target}")
        return images

    if not target.exists():
        raise SystemExit(f"No such file: {target}")
    return [target]


def confidence_bar(value: float, width: int = 28) -> str:
    filled = int(round(value * width))
    return "#" * filled + "." * (width - filled)


def print_human(path: Path, ranked: list[tuple[str, float]], labels: list[str]) -> None:
    top_label, top_confidence = ranked[0]
    accepted = top_confidence >= config.CONFIDENCE_THRESHOLD

    print(f"\n{path.name}")
    print("-" * 60)

    name_width = max(len(label) for label, _ in ranked)
    for rank, (label, confidence) in enumerate(ranked, start=1):
        marker = ">" if rank == 1 else " "
        print(f" {marker} {rank}. {label:<{name_width}}  {confidence:6.2%}  "
              f"{confidence_bar(confidence)}")

    print("-" * 60)
    print(f"   Detected : {top_label.upper()}")
    print(f"   Confidence: {top_confidence:.1%}")

    if not accepted:
        print(f"   REJECTED  : below the {config.CONFIDENCE_THRESHOLD:.0%} threshold.")
        print("               The Main ESP32 would keep its previous fan state.")
    elif top_label == config.UNKNOWN_CLASS:
        print("   REJECTED  : classified as unknown, so no crop to act on.")
        print("               The Main ESP32 would keep its previous fan state.")
    else:
        print(f"   ACCEPTED  : crop_id {labels.index(top_label)} would be sent.")


def print_esp_now(ranked: list[tuple[str, float]], labels: list[str]) -> None:
    """Show the ESP-NOW packet this detection would produce.

    Only top-1 travels over the wire -- the V1 spec keeps the packet to two
    bytes. The top-5 above is a desktop debugging aid, not something the Main
    ESP32 ever sees.
    """
    top_label, top_confidence = ranked[0]
    crop_id = labels.index(top_label)
    confidence_byte = int(round(top_confidence * 100))

    usable = (
        top_confidence >= config.CONFIDENCE_THRESHOLD
        and top_label != config.UNKNOWN_CLASS
    )

    print("\n   ESP-NOW packet (struct CropResult):")
    print(f"     crop_id    = {crop_id:<3}  // {top_label}")
    print(f"     confidence = {confidence_byte:<3}  // percent")
    print(f"     raw bytes  = {crop_id:02X} {confidence_byte:02X}")
    if not usable:
        print("     -> would NOT be transmitted; nothing actionable to send.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a crop image using the ESP32-bound int8 model.",
    )
    parser.add_argument("image", type=Path,
                        help="path to an image file, or a folder of images")
    parser.add_argument("--model", type=Path, default=config.TFLITE_MODEL,
                        help="path to a .tflite model (default: models/model_int8.tflite)")
    parser.add_argument("--labels", type=Path, default=config.LABELS_FILE,
                        help="path to labels.txt")
    parser.add_argument("--top", type=int, default=config.TOP_K,
                        help=f"how many matches to show (default: {config.TOP_K})")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of the human-readable report")
    parser.add_argument("--esp-now", action="store_true",
                        help="also show the ESP-NOW packet that would be sent")
    args = parser.parse_args()

    # A missing model or label file is the normal state before training, not a
    # crash worth a traceback -- the messages carry the fix.
    try:
        labels = read_labels(args.labels)
        classifier = CropClassifier(args.model, labels)
    except (FileNotFoundError, ValueError, ImportError) as error:
        raise SystemExit(str(error))

    images = resolve_images(args.image)

    results = []
    for path in images:
        probabilities = classifier.predict_file(path)
        ranked = classifier.top_k(probabilities, args.top)

        if args.json:
            top_label, top_confidence = ranked[0]
            results.append({
                "image": str(path),
                "detected": top_label,
                "crop_id": labels.index(top_label),
                "confidence": round(top_confidence, 4),
                "accepted": bool(
                    top_confidence >= config.CONFIDENCE_THRESHOLD
                    and top_label != config.UNKNOWN_CLASS
                ),
                "top_k": [
                    {"rank": i, "crop": label, "crop_id": labels.index(label),
                     "confidence": round(float(value), 4)}
                    for i, (label, value) in enumerate(ranked, start=1)
                ],
            })
        else:
            print_human(path, ranked, labels)
            if args.esp_now:
                print_esp_now(ranked, labels)

    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    elif len(images) > 1:
        print(f"\n{len(images)} images classified.")


if __name__ == "__main__":
    main()
