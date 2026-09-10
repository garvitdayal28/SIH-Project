"""
Build the `unknown` class out of photographs of the empty box.

    python scripts/make_background.py                 # default: 120 images
    python scripts/make_background.py --count 200
    python scripts/make_background.py --sheet         # also write a contact sheet
    python scripts/make_background.py --clean         # wipe and rebuild

Reads:  my images/environment/     photographs of the empty rig
Writes: data/raw/background/       square crops, folded into `unknown` by
                                   scripts/prepare_data.py

What this adds to the unknown class
===================================

`unknown` means "not one of the 8 crops", and that covers two different
situations which need two different sources:

    some other produce   config.UNKNOWN_SOURCE_CLASSES, the Kaggle set's other
                         27 classes -- a pomegranate is not a crop we handle
    nothing at all       this script -- the tray is empty

This script supplies the second. Before it existed the unknown class was produce
negatives only, so the model had never seen a bare tray, and an emptied tray came
out as whichever crop the thermocol looked most like. CONFIDENCE_THRESHOLD cannot
paper over that on its own: a model with no empty class must put its probability
mass somewhere, and on a plain bright background it often does so confidently.

The two halves do not substitute for each other, which was measured rather than
assumed. A model trained on empty-box images alone reported a real pomegranate as
an apple at 85% confidence -- it knew "crop" from "empty tray" and nothing about
which crop.

Keep this the minority share of the class. The count here is added on top of
config.UNKNOWN_SIZE_MULTIPLIER's budget rather than inside it, so a large --count
tilts `unknown` towards bare tray and away from the produce negatives that do the
harder job.

How the crops are chosen
========================

The available source is a phone photograph of the empty box taken from outside
it, so the frame contains the box interior along with a sofa and a patterned
carpet. Only the interior is wanted.

Rather than hardcode a rectangle, candidate square crops are sampled at random
and each is tested against what the interior is and the surroundings are not:

    bright        the interior is white thermocol under a lamp
    desaturated   white and grey, where the sofa is brown and the tape is blue
    low-texture   a wall or floor panel, where the carpet is striped

This works here because the separation is wide and the source is a single known
image -- it is a threshold on one photograph, not a general-purpose filter. That
distinction matters: a similar-looking composition filter was tried on the Open
Images training data and abandoned, because there the classes were not separable
this way and the thresholds did not transfer between them.

Every accepted crop then gets photometric jitter baked in -- exposure, contrast,
a warm/cool white-balance shift, blur, and JPEG artefacts -- to approximate the
range an OV2640 produces under a domestic lamp. Baking it in matters because
train.py's augmentation only touches the training split, so without this the
validation and test backgrounds would all be near-identical crops of one
photograph.

Honest limitation
=================

One source photograph is not enough. Every image this produces is a crop of the
same box under the same lamp, so:

  - the model learns *this* box, not "an empty tray" in general;
  - the background part of the unknown row in the confusion matrix will look
    excellent and will not mean much, because the test crops are near-duplicates
    of the train crops.

The fix is cheap and worth doing before any demo: capture 20-30 frames of the
empty box through the ESP32-CAM itself, at the real mounting distance, with the
lamp on and off and the door open and shut, and drop them in
my images/environment/ (or straight into data/raw/background/). Then re-run this.
The script takes any number of source images and spreads the requested count
across them, so nothing needs changing when they arrive.
"""

from __future__ import annotations

import argparse
import io
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Default source: the rig photographs that ship with the project.
DEFAULT_SOURCE = config.ROOT / "my images" / "environment"

# Crops are written at this size. Larger than IMAGE_SIZE on purpose --
# prepare_data.py downscales to 96 through the same Pillow path predict.py
# uses, so writing 96 here would mean two successive resizes.
OUTPUT_SIZE = 256

# --------------------------------------------------------------------------
# What counts as "inside the empty box".
#
# Measured against the one available source photograph, whose interior is a
# white thermocol wall and a speckled white floor, and whose surroundings are a
# brown sofa and a black-and-cream striped carpet. The gap between those is
# wide, so these thresholds are not delicate -- but they are calibrated to this
# photograph, and --report exists to re-check them when new captures arrive.
# --------------------------------------------------------------------------

MIN_BRIGHTNESS = 120.0      # mean grey level; the sofa and carpet sit below
MAX_SATURATION = 42.0       # mean channel range; rejects the blue tape and sofa
MAX_TEXTURE = 26.0          # mean gradient magnitude; rejects the striped carpet
MAX_BRIGHTNESS_SPREAD = 62.0  # std of grey; rejects crops straddling the rim


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_upright(path: Path) -> Image.Image:
    """Open an image, honouring EXIF rotation.

    Without exif_transpose a portrait phone photo arrives rotated 90 degrees and
    every crop position computed below would refer to the wrong part of the
    scene. cropnet.preprocess.load_image does the same thing for the same reason.
    """
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def crop_statistics(crop: Image.Image) -> dict[str, float]:
    """Brightness, saturation, texture and spread for a candidate crop."""
    small = crop.resize((96, 96), Image.BILINEAR)
    rgb = np.asarray(small, dtype=np.float32)

    gray = rgb.mean(axis=2)
    dy, dx = np.gradient(gray)

    return {
        "brightness": float(gray.mean()),
        "spread": float(gray.std()),
        # Cheap saturation: per-pixel channel range. Enough to separate white
        # thermocol from a brown sofa without a colour-space conversion.
        "saturation": float((rgb.max(axis=2) - rgb.min(axis=2)).mean()),
        "texture": float(np.hypot(dx, dy).mean()),
    }


def is_interior(stats: dict[str, float]) -> bool:
    return (
        stats["brightness"] >= MIN_BRIGHTNESS
        and stats["saturation"] <= MAX_SATURATION
        and stats["texture"] <= MAX_TEXTURE
        and stats["spread"] <= MAX_BRIGHTNESS_SPREAD
    )


def sample_interior_crop(
    image: Image.Image, rng: random.Random, attempts: int = 400
) -> Image.Image | None:
    """Find one square crop that looks like the inside of the empty box.

    Sizes are sampled from a wide range so the resulting set spans "camera close
    to the wall" through "camera seeing a whole panel", which is the variation a
    slightly different mounting position would produce.
    """
    width, height = image.size
    largest = min(width, height)
    smallest = max(48, largest // 6)

    for _ in range(attempts):
        side = rng.randint(smallest, largest)
        left = rng.randint(0, width - side)
        top = rng.randint(0, height - side)
        crop = image.crop((left, top, left + side, top + side))
        if is_interior(crop_statistics(crop)):
            return crop
    return None


def jitter(crop: Image.Image, rng: random.Random) -> Image.Image:
    """Approximate the range of frames an OV2640 gives under a domestic lamp.

    Order matters and mirrors a real imaging chain: geometry, then optics
    (blur), then sensor response (exposure/contrast/white balance), then the
    JPEG encoder the ESP32-CAM actually ships frames through.
    """
    if rng.random() < 0.5:
        crop = crop.transpose(Image.FLIP_LEFT_RIGHT)

    # Rotate, then cut back to the largest square that is still entirely inside
    # the rotated image. Without that second step Image.rotate leaves black
    # triangles in the corners, and since only the unknown class is generated
    # here, those triangles would be a feature present in unknown and in no crop
    # class -- the model would learn "black corners mean empty" and score well on
    # the test split for a reason that does not exist on the device.
    angle = rng.uniform(-8, 8)
    if abs(angle) > 0.1:
        side = crop.size[0]
        crop = crop.rotate(angle, resample=Image.BILINEAR, expand=False)
        radians = math.radians(abs(angle))
        inner = int(side / (math.cos(radians) + math.sin(radians)))
        offset = (side - inner) // 2
        crop = crop.crop((offset, offset, offset + inner, offset + inner))

    crop = crop.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)

    # The OV2640 through a cheap lens is never quite in focus.
    if rng.random() < 0.7:
        crop = crop.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.4)))

    crop = ImageEnhance.Brightness(crop).enhance(rng.uniform(0.72, 1.28))
    crop = ImageEnhance.Contrast(crop).enhance(rng.uniform(0.70, 1.20))

    # White balance. The ESPCAM frames in my images/ run warm and washed out,
    # but the lamp could as easily be cool, so the shift goes both ways.
    red, green, blue = crop.split()
    warmth = rng.uniform(-0.10, 0.16)
    if abs(warmth) > 0.01:
        red = red.point(lambda v: min(255, int(v * (1.0 + warmth))))
        blue = blue.point(lambda v: min(255, int(v * (1.0 - warmth))))
        crop = Image.merge("RGB", (red, green, blue))

    # Round-trip through JPEG at the sort of quality the camera uses.
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG", quality=rng.randint(55, 88))
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def report_sources(sources: list[Path]) -> None:
    """Print the statistics behind the thresholds, for recalibration."""
    print("Per-source crop acceptance (200 random candidates each):\n")
    print(f"  {'source':<34} {'accepted':>9}  "
          f"{'bright':>7} {'sat':>6} {'tex':>6} {'spread':>7}")
    rng = random.Random(config.SEED)
    for path in sources:
        image = load_upright(path)
        width, height = image.size
        largest = min(width, height)
        accepted = 0
        rows = []
        for _ in range(200):
            side = rng.randint(max(48, largest // 6), largest)
            left = rng.randint(0, width - side)
            top = rng.randint(0, height - side)
            stats = crop_statistics(image.crop((left, top, left + side, top + side)))
            rows.append(stats)
            if is_interior(stats):
                accepted += 1
        means = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        name = path.name[:32]
        print(f"  {name:<34} {accepted:>4}/200  "
              f"{means['brightness']:>7.1f} {means['saturation']:>6.1f} "
              f"{means['texture']:>6.1f} {means['spread']:>7.1f}")
    print("\nThresholds in scripts/make_background.py:")
    print(f"  brightness >= {MIN_BRIGHTNESS}, saturation <= {MAX_SATURATION}, "
          f"texture <= {MAX_TEXTURE}, spread <= {MAX_BRIGHTNESS_SPREAD}")


def write_sheet(paths: list[Path], out: Path, cols: int = 12, cell: int = 84) -> None:
    if not paths:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = (len(paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * cell), (24, 24, 24))
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((cell - 2, cell - 2))
            canvas.paste(thumb, ((i % cols) * cell + (cell - thumb.width) // 2,
                                 (i // cols) * cell + (cell - thumb.height) // 2))
    canvas.save(out)
    print(f"Contact sheet: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the unknown class from photographs of the empty rig.",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help=f"folder of empty-box photographs "
                             f"(default: {DEFAULT_SOURCE.relative_to(config.ROOT)})")
    parser.add_argument("--count", type=int, default=60,
                        help="how many background images to write (default 60, "
                             "which splits 70/15/15 into roughly 42 train). "
                             "These sit alongside the produce negatives in the "
                             "unknown class, so this is a minority share on "
                             "purpose -- see the module docstring")
    parser.add_argument("--clean", action="store_true",
                        help="delete existing generated images first")
    parser.add_argument("--sheet", action="store_true",
                        help="write a contact sheet to outputs/ to check by eye")
    parser.add_argument("--report", action="store_true",
                        help="print crop statistics per source and exit")
    args = parser.parse_args()

    sources = list_images(args.source)
    if not sources:
        raise SystemExit(
            f"No images in {args.source}.\n\n"
            "Put photographs of the EMPTY box there -- the walls, the floor, the\n"
            "corners, with the lamp on and off. 20-30 frames captured through the\n"
            "ESP32-CAM itself are worth far more than one phone photo."
        )

    if args.report:
        report_sources(sources)
        return

    destination = config.BACKGROUND_DIR
    destination.mkdir(parents=True, exist_ok=True)

    if args.clean:
        removed = 0
        for path in destination.glob("bg_*.png"):
            path.unlink()
            removed += 1
        if removed:
            print(f"Removed {removed} previously generated image(s).")

    existing = {p.name for p in list_images(destination)}
    if existing:
        print(f"Note: {len(existing)} image(s) already in {destination}; "
              f"they are kept and will also be used.")

    print(f"Sources: {len(sources)} photograph(s) in {args.source}")
    print(f"Writing {args.count} crops to {destination}\n")

    rng = random.Random(config.SEED)
    loaded = [(path, load_upright(path)) for path in sources]

    written: list[Path] = []
    failures = 0
    index = 0
    while len(written) < args.count:
        source_path, image = loaded[index % len(loaded)]
        index += 1
        if index > args.count * 8 + 200:
            break

        crop = sample_interior_crop(image, rng)
        if crop is None:
            failures += 1
            # Every source is retried, but one that never yields is worth
            # naming -- it is usually a photo of something other than the box.
            if failures % max(1, len(loaded)) == 0:
                pass
            continue

        out = destination / f"bg_{len(written):05d}.png"
        jitter(crop, rng).save(out)
        written.append(out)

    if not written:
        raise SystemExit(
            "No crop in any source image looked like the inside of the box.\n\n"
            "Run with --report to see the measured statistics, then adjust the\n"
            "thresholds near the top of this script. They were calibrated on a\n"
            "white thermocol interior under a warm lamp; a darker or more\n"
            "colourful rig will need different numbers."
        )

    print(f"Wrote {len(written)} background image(s) at "
          f"{OUTPUT_SIZE}x{OUTPUT_SIZE}.")
    print("  These are added on top of config.UNKNOWN_SIZE_MULTIPLIER's budget,\n"
          "  not inside it, so a large --count skews the whole class towards\n"
          "  bare tray and away from the produce negatives.")
    if len(written) < args.count:
        print(f"  Asked for {args.count}; the sources did not yield more. "
              f"Run --report to see why.")

    per_split = (
        int(len(written) * config.SPLIT_RATIOS[0]),
        int(len(written) * config.SPLIT_RATIOS[1]),
    )
    print(f"\nprepare_data.py will split these {config.SPLIT_RATIOS} into roughly")
    print(f"  {per_split[0]} train / {per_split[1]} val / "
          f"{len(written) - sum(per_split)} test  for the unknown class.")

    if args.sheet:
        write_sheet(written[:120], config.OUTPUTS_DIR / "background_sheet.png")

    print("\nThese are crops of a single photograph, so the unknown class is")
    print("thinner than its image count suggests. See the module docstring.")
    print("\nNext: python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
