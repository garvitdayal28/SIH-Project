"""
Label handling.

The label order is not cosmetic: the index of a class in this list becomes the
`crop_id` byte sent over ESP-NOW, and the Main ESP32 looks up fan speed by that
id. If the order ever changes without the firmware's crop table changing with
it, tomato silently becomes potato. So the order is written to labels.txt at
training time and every consumer reads it from there rather than recomputing it.
"""

from __future__ import annotations

from pathlib import Path


def write_labels(path: str | Path, names: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def read_labels(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {path}\n"
            "Train and export the model first (see README.md)."
        )
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [name for name in names if name]
