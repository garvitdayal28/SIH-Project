"""
Build the training set from the real rig photographs.

    python scripts/build_rig_dataset.py

This replaces the Kaggle-based pipeline for this project. The rig is a fixed
camera looking into a white thermocol box that holds exactly one item, and the
item is always one of four: banana, lemon, onion, tomato. Nothing else is ever
placed in it. So the model does not need to know what a pomegranate is, or what
an apple looks like on a tree -- it needs to tell four objects apart against one
background, and say "empty" when the box is bare.

Where the data comes from
-------------------------
    my images/from phone camera/<class>/   real photographs in the box
    my images/environment/empty.jpeg       the empty box
    data/raw/train/banana/                 Kaggle, only because there are no
                                           real banana photographs yet

Held-out val and test are real photographs only. Composites go into train and
nowhere else, so the reported accuracy is measured on genuine frames.

Banana is the exception and it is worth being clear about: with no real photos,
its val and test images are synthetic too, so its score is optimistic. Photograph
a banana in the box and this stops being true.

How train gets expanded
-----------------------
Seventeen photographs per class is not enough. Each real photo is cut apart --
the object separated from the white box -- and the object is pasted back onto
box backgrounds at other positions, scales and orientations. Everything stays
inside the real domain: real object pixels, real background pixels, only the
arrangement is new.

Finally each training image is pushed through a rough imitation of the ESP32-CAM
(smaller sensor, softer lens, noisier, warmer) so the model does not depend on
phone-camera sharpness that the real device will never deliver.
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from cropnet.labels import write_labels
from cropnet.preprocess import center_crop_to_square

RIG_DIR = config.ROOT / "my images" / "from phone camera"
EMPTY_PHOTO = config.ROOT / "my images" / "environment" / "empty.jpeg"
KAGGLE_FALLBACK = {"banana": config.DATA_RAW / "train" / "banana"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WORK = 256                      # working resolution before the final resize
TRAIN_PER_CLASS = 420           # after expansion
REAL_VARIANTS = 9               # camera simulations kept per real photograph
VAL_FRACTION, TEST_FRACTION = 0.17, 0.17


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------

def object_mask(rgb: np.ndarray) -> np.ndarray | None:
    """Separate the item from the white box.

    Easy here in a way it never was for web photographs: the background is a
    single near-white material and the item is strongly coloured, so distance
    from the border colour is enough. Returns None if that assumption fails.
    """
    h, w, _ = rgb.shape
    band = max(2, h // 12)
    border = np.concatenate([
        rgb[:band].reshape(-1, 3), rgb[-band:].reshape(-1, 3),
        rgb[:, :band].reshape(-1, 3), rgb[:, -band:].reshape(-1, 3),
    ])
    median = np.median(border, axis=0)
    if median.mean() < 110:
        return None

    distance = np.linalg.norm(rgb - median, axis=2)
    mask = (distance > 58).astype(np.uint8) * 255

    m = Image.fromarray(mask)
    m = m.filter(ImageFilter.MedianFilter(5))
    m = m.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    arr = np.asarray(m, dtype=np.float32) / 255.0

    covered = arr.mean()
    if not (0.01 < covered < 0.70):
        return None

    # Keep only the largest blob, so a shadow in a corner does not travel with
    # the object.
    arr = largest_blob(arr > 0.5).astype(np.float32)
    if arr.mean() < 0.008:
        return None

    soft = Image.fromarray((arr * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(1.5))
    return np.asarray(soft, dtype=np.float32) / 255.0


def largest_blob(binary: np.ndarray) -> np.ndarray:
    """Connected-component keep-largest, via iterative dilation from a seed."""
    from collections import deque
    h, w = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    best = np.zeros_like(binary, dtype=bool)
    best_n = 0
    for sy in range(0, h, 4):
        for sx in range(0, w, 4):
            if not binary[sy, sx] or seen[sy, sx]:
                continue
            q = deque([(sy, sx)])
            comp = []
            seen[sy, sx] = True
            while q:
                y, x = q.popleft()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(comp) > best_n:
                best_n = len(comp)
                best = np.zeros_like(binary, dtype=bool)
                for y, x in comp:
                    best[y, x] = True
    return best


def bounding_box(mask: np.ndarray):
    ys, xs = np.where(mask > 0.4)
    if len(ys) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


# ---------------------------------------------------------------------------
# camera simulation
# ---------------------------------------------------------------------------

def simulate_espcam(img: Image.Image, rng: random.Random) -> Image.Image:
    """Approximate what an OV2640 does to a scene a phone renders cleanly.

    Not a calibrated model, just the differences that matter: less resolution,
    a softer lens, sensor noise, and a colour cast from automatic white balance
    that has only white thermocol to lock onto.
    """
    small = rng.randint(96, 190)
    img = img.resize((small, small), Image.BILINEAR).resize((WORK, WORK), Image.BILINEAR)

    if rng.random() < 0.8:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.5)))

    a = np.asarray(img, dtype=np.float32)
    # Gentle. The first pass used +-14% per channel and turned the white box
    # pink and purple, which the real camera never does -- it has a whole wall
    # of white thermocol to balance against, so its cast is mild.
    gain = np.array([rng.uniform(0.95, 1.05) for _ in range(3)], dtype=np.float32)
    a *= gain
    a *= rng.uniform(0.82, 1.08)                       # exposure
    mean = a.mean()
    a = (a - mean) * rng.uniform(0.90, 1.12) + mean     # contrast
    a += np.random.normal(0, rng.uniform(1.5, 7.0), a.shape).astype(np.float32)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_square(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as im:
            return center_crop_to_square(im.convert("RGB")).resize(
                (WORK, WORK), Image.BILINEAR)
    except Exception:
        return None


def is_empty_region(img: Image.Image) -> bool:
    """True when a candidate plate contains no object.

    This check is not optional. Cropping "the top part of a rig photo" and
    assuming the item is below it is wrong often enough to poison the whole
    dataset: the first build put tomatoes and onions into the `empty` class,
    and pasted lemons on top of plates that already held an onion, so several
    classes were training on each other's objects.

    Anything with a coloured blob in it is rejected, whatever its position.
    """
    a = np.asarray(img.resize((96, 96), Image.BILINEAR), dtype=np.float32)
    median = np.median(a.reshape(-1, 3), axis=0)
    if median.mean() < 105:
        return False
    distance = np.linalg.norm(a - median, axis=2)
    # a bare box is uniform; a few percent covers seams, shadow and the blue rim
    return float((distance > 58).mean()) < 0.045


def box_backgrounds(rng: random.Random) -> list[Image.Image]:
    """Background plates: regions of the box verified to contain no item."""
    plates: list[Image.Image] = []

    if EMPTY_PHOTO.exists():
        with Image.open(EMPTY_PHOTO) as im:
            empty = im.convert("RGB")
        W, H = empty.size
        for _ in range(120):
            side = rng.randint(int(min(W, H) * 0.40), min(W, int(H * 0.55)))
            left = rng.randint(0, W - side)
            top = rng.randint(int(H * 0.15), max(int(H * 0.15), H - side))
            plate = empty.crop((left, top, left + side, top + side)).resize(
                (WORK, WORK), Image.BILINEAR)
            if is_empty_region(plate):
                plates.append(plate)

    # Item-free regions of the rig photos, for their lighting variety -- but
    # only where the emptiness check agrees.
    for cls in sorted(p.name for p in RIG_DIR.iterdir() if p.is_dir()):
        for p in list_images(RIG_DIR / cls):
            try:
                with Image.open(p) as im:
                    full = im.convert("RGB")
            except Exception:
                continue
            W, H = full.size
            side = int(W * 0.7)
            for top_frac in (0.06, 0.16, 0.26):
                top = int(H * top_frac)
                if top + side > H:
                    continue
                plate = full.crop((int(W * 0.15), top, int(W * 0.15) + side, top + side)
                                  ).resize((WORK, WORK), Image.BILINEAR)
                if is_empty_region(plate):
                    plates.append(plate)

    if not plates:
        raise SystemExit("No empty background plates could be verified.")
    return plates


def cutouts(paths: list[Path]) -> list[tuple[Image.Image, np.ndarray]]:
    """Object image plus its mask, cropped tight, for every photo we can cut."""
    out = []
    for p in paths:
        img = load_square(p)
        if img is None:
            continue
        mask = object_mask(np.asarray(img, dtype=np.float32))
        if mask is None:
            continue
        bb = bounding_box(mask)
        if bb is None:
            continue
        x0, y0, x1, y1 = bb
        if (x1 - x0) < 18 or (y1 - y0) < 18:
            continue

        sub_mask = mask[y0:y1, x0:x1]
        sub_img = img.crop((x0, y0, x1, y1))

        # Reject wall seams and shadows that survived segmentation. Two tells,
        # both seen in the first build as grey rectangles pasted into the
        # lemon, onion and tomato classes:
        #
        #   a real item is roundish, so it fills well under all of its bounding
        #   box, while a seam fills nearly the whole thing;
        #   a real item is coloured, while thermocol and shadow are grey.
        fill = float((sub_mask > 0.5).mean())
        if fill > 0.93:
            continue

        a = np.asarray(sub_img, dtype=np.float32)
        sel = sub_mask > 0.5
        if sel.sum() < 40:
            continue
        px = a[sel]
        chroma = float(np.mean(px.max(axis=1) - px.min(axis=1)))
        if chroma < 22:
            continue

        out.append((sub_img, sub_mask))
    return out


def paste(plate: Image.Image, item: Image.Image, mask: np.ndarray,
          rng: random.Random) -> Image.Image:
    """Drop a cut-out item onto a background plate, low and roughly centred."""
    # Wide on purpose. Every failure in the previous build sat at one extreme
    # or the other: three distant lemons that filled about a tenth of the frame
    # and were called tomato, and a close onion and tomato that filled most of
    # it. The composites only spanned 0.16-0.62, so neither end was represented.
    scale = rng.uniform(0.10, 0.88)
    iw, ih = item.size
    longest = max(iw, ih)
    factor = (WORK * scale) / longest
    tw, th = max(12, int(iw * factor)), max(12, int(ih * factor))

    it = item.resize((tw, th), Image.BILINEAR)
    mk = np.asarray(Image.fromarray((mask * 255).astype(np.uint8))
                    .resize((tw, th), Image.BILINEAR), dtype=np.float32) / 255.0
    if rng.random() < 0.5:
        it = it.transpose(Image.FLIP_LEFT_RIGHT)
        mk = mk[:, ::-1]

    cx = int(WORK * rng.uniform(0.32, 0.68))
    cy = int(WORK * rng.uniform(0.46, 0.80))
    x0 = int(np.clip(cx - tw // 2, 0, WORK - tw))
    y0 = int(np.clip(cy - th // 2, 0, WORK - th))

    bg = np.asarray(plate, dtype=np.float32).copy()
    fg = np.asarray(it, dtype=np.float32)
    a = mk[..., None]
    region = bg[y0:y0 + th, x0:x0 + tw]
    bg[y0:y0 + th, x0:x0 + tw] = region * (1 - a) + fg * a
    return Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def save(img: Image.Image, path: Path) -> None:
    img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR).save(path)


def main() -> None:
    rng = random.Random(config.SEED)
    np.random.seed(config.SEED)

    classes = config.class_names()
    empty_label = config.UNKNOWN_CLASS
    crops = [c for c in classes if c != empty_label]

    if config.DATA_PREPARED.exists():
        shutil.rmtree(config.DATA_PREPARED)
    for split in ("train", "val", "test"):
        for c in classes:
            (config.DATA_PREPARED / split / c).mkdir(parents=True, exist_ok=True)

    plates = box_backgrounds(rng)
    print(f"{len(plates)} background plates from the box\n")
    print(f"{'class':8} {'real':>5} {'train':>7} {'val':>5} {'test':>5}  source")

    for cls in crops:
        real = list_images(RIG_DIR / cls)
        synthetic_only = not real
        if synthetic_only:
            real = list_images(KAGGLE_FALLBACK.get(cls, Path("/nonexistent")))

        rng.shuffle(real)
        n = len(real)
        n_test = max(1, int(n * TEST_FRACTION))
        n_val = max(1, int(n * VAL_FRACTION))
        test_src, val_src, train_src = real[:n_test], real[n_test:n_test + n_val], real[n_test + n_val:]

        # val/test: the real photograph itself, no compositing
        for split, group in (("val", val_src), ("test", test_src)):
            for i, p in enumerate(group):
                img = load_square(p)
                if img is None:
                    continue
                if synthetic_only:
                    cut = cutouts([p])
                    if cut:
                        img = paste(rng.choice(plates), cut[0][0], cut[0][1], rng)
                        img = simulate_espcam(img, rng)
                save(img, config.DATA_PREPARED / split / cls / f"{cls}_{i:04d}.png")

        # train: real photos, then composites built from their cut-outs
        # Real photographs first, and heavily. Each one is kept as-is plus a
        # handful of ESP32-CAM simulations, which preserve the true geometry --
        # real object, real position, real shadow -- while varying only what
        # the sensor does to it. Composites vary the arrangement, so they are
        # useful, but they are a guess about the rig; the photographs are not.
        #
        # The first build emitted 2 real-derived images per class against ~286
        # composites, and four clearly yellow lemons came out as tomato.
        made = 0
        for i, p in enumerate(train_src):
            img = load_square(p)
            if img is None:
                continue
            if not synthetic_only:
                save(img, config.DATA_PREPARED / "train" / cls / f"{cls}_r{made:04d}.png")
                made += 1
                for _ in range(REAL_VARIANTS):
                    save(simulate_espcam(img, rng),
                         config.DATA_PREPARED / "train" / cls / f"{cls}_r{made:04d}.png")
                    made += 1

        cuts = cutouts(train_src)
        guard = 0
        while made < TRAIN_PER_CLASS and cuts and guard < TRAIN_PER_CLASS * 6:
            guard += 1
            item, mask = rng.choice(cuts)
            img = paste(rng.choice(plates), item, mask, rng)
            img = simulate_espcam(img, rng)
            save(img, config.DATA_PREPARED / "train" / cls / f"{cls}_c{made:04d}.png")
            made += 1

        counts = [len(list((config.DATA_PREPARED / s / cls).glob("*.png")))
                  for s in ("train", "val", "test")]
        tag = "KAGGLE (no real photos)" if synthetic_only else "real rig photos"
        print(f"{cls:8} {n:>5} {counts[0]:>7} {counts[1]:>5} {counts[2]:>5}  {tag}")

    # ---- empty box ---------------------------------------------------------
    for split, count in (("train", TRAIN_PER_CLASS), ("val", 30), ("test", 30)):
        for i in range(count):
            plate = rng.choice(plates)
            img = simulate_espcam(plate, rng) if split == "train" or rng.random() < 0.5 else plate
            save(img, config.DATA_PREPARED / split / empty_label / f"{empty_label}_{i:04d}.png")
    counts = [len(list((config.DATA_PREPARED / s / empty_label).glob("*.png")))
              for s in ("train", "val", "test")]
    print(f"{empty_label:8} {'1':>5} {counts[0]:>7} {counts[1]:>5} {counts[2]:>5}  empty-box crops")

    write_labels(config.LABELS_FILE, classes)
    print(f"\nclasses (crop_id order): {', '.join(f'{i}={c}' for i, c in enumerate(classes))}")
    print("Next: python scripts/train.py")


if __name__ == "__main__":
    main()
