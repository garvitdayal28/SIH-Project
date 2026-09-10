"""
Generate rig-domain training images by compositing produce into the real box.

    python scripts/make_rig_composites.py

Why this exists
---------------
The training set had a shortcut in it that made the model useless on the actual
hardware. Photographs taken inside the box existed in exactly one class --
`unknown`, built from crops of the empty-box photo -- while all 8 crop classes
held only Kaggle web photos. The cheapest rule that separates those classes is
"white thermocol texture -> unknown", so that is what the model learned.

Measured on the real rig photos before this script existed:

    lemon in the box    -> unknown 90.2%      (same lemon as a web photo: 99.6% lemon)
    tomato in the box   -> unknown 79.7%      (same tomato as a web photo: TOMATO)
    pomegranate in box  -> potato  79.3%      accepted, and wrong

The fix is not to remove the empty-box images -- the model genuinely needs to
recognise an empty box. It is to give EVERY class images that share the box's
background statistics, so background stops being a usable cue and the model has
to look at the object.

What it produces
----------------
For each of the 8 crops, and for the produce negatives that feed `unknown`, a
set of images showing that item sitting in the box: random wall/floor
background, plausible position and scale, matched to how the item actually
appears in `my images/`.

The composites are synthetic and look it up close. That does not matter. Their
job is to remove background as a shortcut, and for that they only have to share
the box's colour and texture statistics, which they do exactly -- the
background pixels are the real photograph.

Output: data/raw/rig_composite/<class>/*.jpg, wired in through
config.EXTRA_TRAIN_DIRS.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

BOX_PHOTO = config.ROOT / "my images" / "environment" / "empty.jpeg"
DEST = config.DATA_RAW / "rig_composite"
OUT_SIZE = 256
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# How much of the frame the item fills, and where it sits.
#
# Measured off the real photos, and they disagree, so the range spans both:
#
#   ESP32-CAM, mounted   item fills roughly half the frame   (pomegranate-*.jpeg)
#   phone, held back     item fills under a fifth            (lemon/tomato.jpeg)
#
# The mounted camera is the deployment and the wide end matters most, but the
# narrow end is worth covering because at 96x96 an item at 0.18 of frame width
# is only about 17 pixels across -- if the model has never seen one that small
# it has no chance on a frame like that.
#
# The item always rests on the floor of the box, so it sits low, never high.
SCALE_RANGE = (0.15, 0.62)
CENTRE_X_RANGE = (0.34, 0.66)
CENTRE_Y_RANGE = (0.45, 0.80)

PER_CROP = 160
PER_NEGATIVE_CLASS = 16


def load_box(rng: random.Random) -> Image.Image:
    """Return the empty-box photograph, cached on the function."""
    if not hasattr(load_box, "_cache"):
        if not BOX_PHOTO.exists():
            raise SystemExit(
                f"Missing {BOX_PHOTO}.\n"
                "This script needs the photograph of the empty box."
            )
        with Image.open(BOX_PHOTO) as im:
            load_box._cache = im.convert("RGB").copy()
    return load_box._cache


def box_background(rng: random.Random) -> Image.Image:
    """A random square region of the empty box, as the composite's background.

    Restricted to the lower two thirds of the photo, which is where the box
    interior is -- the top of the frame is the room behind it.
    """
    box = load_box(rng)
    W, H = box.size
    top_limit = int(H * 0.18)
    side = rng.randint(int(min(W, H * 0.55) * 0.75), min(W, int(H * 0.62)))
    left = rng.randint(0, max(0, W - side))
    top = rng.randint(top_limit, max(top_limit, H - side))
    crop = box.crop((left, top, left + side, top + side))
    return crop.resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)


def object_mask(rgb: np.ndarray) -> np.ndarray | None:
    """Segment the subject from a light, uniform background.

    Returns a float mask in [0, 1], or None when the background is not
    separable enough to trust -- many Kaggle photographs are shot on wooden
    tables or in bowls, where this would cut the object in half.
    """
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    median = np.median(border, axis=0)
    if median.mean() < 140:
        return None

    spread = np.linalg.norm(border - median, axis=1)
    if np.percentile(spread, 85) > 70:
        return None

    distance = np.linalg.norm(rgb - median, axis=2)
    mask = (distance > 55).astype(np.float32)

    covered = mask.mean()
    if not (0.04 < covered < 0.88):
        return None

    # Drop specks and fill pinholes, then soften the edge so the paste does not
    # show a hard cut line.
    m = Image.fromarray((mask * 255).astype(np.uint8))
    m = m.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    m = m.filter(ImageFilter.GaussianBlur(1.6))
    return np.asarray(m, dtype=np.float32) / 255.0


def fallback_mask(size: int, rng: random.Random) -> np.ndarray:
    """A soft rounded mask for images whose background cannot be segmented.

    Keeps the middle of the source photo and feathers away the edges. Some of
    the original background survives in the centre, which is imperfect but
    harmless: the point is that everything AROUND the object becomes real box
    pixels, which is what removes the shortcut.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = (size - 1) / 2
    rx = size * rng.uniform(0.34, 0.44)
    ry = size * rng.uniform(0.34, 0.44)
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    mask = np.clip(1.6 - 1.6 * d, 0.0, 1.0)
    return mask


def composite(source: Path, rng: random.Random) -> Image.Image | None:
    try:
        with Image.open(source) as im:
            item = im.convert("RGB")
    except Exception:
        return None

    # square-crop the source so the object keeps its aspect ratio
    W, H = item.size
    side = min(W, H)
    item = item.crop(((W - side) // 2, (H - side) // 2,
                      (W - side) // 2 + side, (H - side) // 2 + side))
    item = item.resize((192, 192), Image.BILINEAR)
    rgb = np.asarray(item, dtype=np.float32)

    mask = object_mask(rgb)
    if mask is None:
        mask = fallback_mask(192, rng)

    scale = rng.uniform(*SCALE_RANGE)
    target = max(24, int(OUT_SIZE * scale))
    item_small = item.resize((target, target), Image.BILINEAR)
    mask_small = np.asarray(
        Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (target, target), Image.BILINEAR),
        dtype=np.float32) / 255.0

    background = box_background(rng)
    bg = np.asarray(background, dtype=np.float32)
    fg = np.asarray(item_small, dtype=np.float32)

    cx = int(OUT_SIZE * rng.uniform(*CENTRE_X_RANGE))
    cy = int(OUT_SIZE * rng.uniform(*CENTRE_Y_RANGE))
    x0 = int(np.clip(cx - target // 2, 0, OUT_SIZE - target))
    y0 = int(np.clip(cy - target // 2, 0, OUT_SIZE - target))

    region = bg[y0:y0 + target, x0:x0 + target]
    a = mask_small[..., None]

    # Nudge the item towards the background's overall brightness, so it reads as
    # lit by the same scene rather than pasted from a different photograph.
    #
    # Clamped hard. The box is near-white, so an unclamped ratio blows a dark
    # potato out to a pale blob and destroys the colour that identifies it --
    # the gain is meant to match the lighting, not erase the subject.
    gain = float(bg.mean()) / max(float(fg.mean()), 1e-3)
    gain = float(np.clip(0.80 + 0.20 * gain, 0.88, 1.18))
    fg = np.clip(fg * gain, 0, 255)

    bg[y0:y0 + target, x0:x0 + target] = region * (1 - a) + fg * a
    out = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))

    # A touch of blur, because the camera is cheap and close.
    if rng.random() < 0.6:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))
    return out


def source_images(class_folder: str) -> list[Path]:
    found: list[Path] = []
    for split in ("train", "validation", "test"):
        d = config.DATA_RAW / split / class_folder
        if d.exists():
            found += [p for p in sorted(d.iterdir())
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return found


def generate(label: str, folders: list[str], count: int, rng: random.Random) -> int:
    out_dir = DEST / label
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    pool: list[Path] = []
    for f in folders:
        pool += source_images(f)
    if not pool:
        print(f"  {label:8} no source images found for {folders}")
        return 0

    rng.shuffle(pool)
    made = 0
    i = 0
    while made < count and i < count * 4:
        src = pool[i % len(pool)]
        i += 1
        img = composite(src, rng)
        if img is None:
            continue
        img.save(out_dir / f"rig_{label}_{made:04d}.jpg", quality=92)
        made += 1
    print(f"  {label:8} {made:>4} composites from {len(pool)} source photos")
    return made


def main() -> None:
    rng = random.Random(config.SEED)
    print(f"Compositing produce into {BOX_PHOTO.name}\n")

    total = 0
    for crop, folder in sorted(config.TARGET_CROPS.items()):
        total += generate(crop, [folder], PER_CROP, rng)

    # The negatives matter as much as the crops. Without them, "something in the
    # box that is not one of the 8" has no rig-domain example, and a pomegranate
    # in the box comes out as potato -- which is exactly what happened.
    negatives = list(getattr(config, "UNKNOWN_SOURCE_CLASSES", []))
    if negatives and config.INCLUDE_UNKNOWN:
        count = PER_NEGATIVE_CLASS * len(negatives)
        total += generate(config.UNKNOWN_CLASS, negatives, count, rng)

    print(f"\n{total} composites written to {DEST}")
    print("Next: python scripts/prepare_data.py, then train.py")


if __name__ == "__main__":
    main()
