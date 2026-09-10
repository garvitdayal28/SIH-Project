"""
Turn the downloaded Kaggle dataset into a training-ready folder tree.

    python scripts/prepare_data.py

Reads:  data/raw/       the dataset exactly as unzipped
Writes: data/prepared/  train|val|test / <class> / *.jpg

Three things happen here that matter more than they look:

1. Only the 12 crops in config.TARGET_CROPS are kept. The dataset has 36
   classes; carrying the other 24 as their own outputs would widen the final
   layer, need more data per class, and lower accuracy on the crops we actually
   care about.

2. The other 24 classes are folded into a single "unknown" class instead of
   being thrown away. They are the most useful negatives we have -- real produce
   photos that are not one of our crops -- and they are free.

3. The unknown class is capped and evenly sampled. 24 source classes against 12
   single crops is a 24:1 imbalance; left alone the model learns that answering
   "unknown" is right most of the time.
"""

from __future__ import annotations

import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

import config
from cropnet.labels import write_labels
from cropnet.preprocess import load_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = ("train", "val", "test")

# What the Kaggle set calls its splits, mapped to what we call them.
SOURCE_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


def normalize(name: str) -> str:
    """Fold class-folder names so 'Chilli Pepper' matches 'chilli pepper'."""
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_split_dirs(root: Path) -> dict[str, Path]:
    """Locate the dataset's own train/validation/test folders, if it has any.

    The Kaggle zip sometimes extracts with an extra nesting level, so this looks
    a couple of directories deep rather than assuming a fixed layout.
    """
    for base in [root, *(d for d in root.iterdir() if d.is_dir())]:
        found = {}
        for child in base.iterdir():
            if not child.is_dir():
                continue
            alias = SOURCE_SPLIT_ALIASES.get(normalize(child.name))
            if alias:
                found[alias] = child
        if "train" in found:
            return found
    return {}


def find_class_dirs(root: Path) -> dict[str, Path]:
    """Map normalized class name -> folder, for a flat one-folder-per-class tree."""
    classes = {}
    for child in sorted(root.iterdir()):
        if child.is_dir() and list_images(child):
            classes[normalize(child.name)] = child
    return classes


def collect_source_images(raw_root: Path) -> tuple[dict[str, dict[str, list[Path]]], bool]:
    """Gather images as {split: {normalized_class: [paths]}}.

    Returns the mapping plus a flag saying whether the dataset supplied its own
    splits. When it did we reuse them -- re-splitting would risk putting near
    duplicate photos of the same physical apple in both train and test, which
    would quietly inflate the accuracy numbers.
    """
    split_dirs = find_split_dirs(raw_root)

    if split_dirs:
        by_split: dict[str, dict[str, list[Path]]] = {}
        for split, directory in split_dirs.items():
            by_split[split] = {
                name: list_images(folder)
                for name, folder in find_class_dirs(directory).items()
            }
        # A dataset with train+test but no validation still needs one; carve it
        # out of train later rather than reusing test for model selection.
        return by_split, True

    flat = find_class_dirs(raw_root)
    if not flat:
        return {}, False
    return {"train": flat}, False


def split_list(items: list[Path], ratios: tuple[float, float, float], rng: random.Random):
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return (
        shuffled[:n_train],
        shuffled[n_train:n_train + n_val],
        shuffled[n_train + n_val:],
    )


def gather_background_images() -> list[Path]:
    if not config.BACKGROUND_DIR.exists():
        return []
    return [
        p for p in config.BACKGROUND_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def build_plan(raw_root: Path, rng: random.Random):
    """Decide, for every split and class, exactly which files get copied."""
    by_split, had_splits = collect_source_images(raw_root)
    if not by_split:
        raise SystemExit(
            f"No images found under {raw_root}.\n\n"
            "Download the Kaggle 'Fruit and Vegetable Image Recognition' dataset\n"
            "and unzip it into that folder. See README.md for the exact steps."
        )

    # Ensure every split key exists so later code can assume it.
    for split in SPLIT_NAMES:
        by_split.setdefault(split, {})

    if had_splits and not by_split["val"]:
        print("Dataset has no validation split; carving 15% out of train.")
        for name, paths in list(by_split["train"].items()):
            train_part, val_part, _ = split_list(paths, (0.85, 0.15, 0.0), rng)
            by_split["train"][name] = train_part
            by_split["val"][name] = val_part

    plan: dict[str, dict[str, list[Path]]] = {s: defaultdict(list) for s in SPLIT_NAMES}

    # --- the 12 target crops -------------------------------------------------
    missing = []
    for crop, source_name in sorted(config.TARGET_CROPS.items()):
        key = normalize(source_name)
        found_anywhere = False
        for split in SPLIT_NAMES:
            paths = by_split[split].get(key, [])
            if paths:
                found_anywhere = True
            plan[split][crop].extend(paths)
        if not found_anywhere:
            missing.append(f"{crop} (looked for folder '{source_name}')")

    if missing:
        available = sorted({k for s in SPLIT_NAMES for k in by_split[s]})
        raise SystemExit(
            "These crops from config.TARGET_CROPS were not found in the dataset:\n  "
            + "\n  ".join(missing)
            + "\n\nClass folders that do exist:\n  "
            + ", ".join(available)
            + "\n\nFix the folder names on the right-hand side of TARGET_CROPS in config.py."
        )

    report_unused_classes(by_split)

    if not had_splits:
        # Everything landed in "train"; split it now.
        for crop in list(plan["train"].keys()):
            train_part, val_part, test_part = split_list(
                plan["train"][crop], config.SPLIT_RATIOS, rng
            )
            plan["train"][crop] = train_part
            plan["val"][crop] = val_part
            plan["test"][crop] = test_part

    # --- the unknown class ---------------------------------------------------
    if config.INCLUDE_UNKNOWN:
        add_unknown_class(plan, by_split, had_splits, rng)

    return plan


def report_unused_classes(by_split) -> None:
    """Warn about dataset classes that are neither a target nor a negative.

    Every folder in the dataset should be a deliberate decision: a crop we
    detect, a negative we reject, or an explicit exclusion. A class that is in
    none of the three lists is usually a typo in config.py, and the symptom --
    slightly worse accuracy -- is invisible without this check.
    """
    available = {name for split in SPLIT_NAMES for name in by_split.get(split, {})}

    accounted = {normalize(n) for n in config.TARGET_CROPS.values()}
    accounted |= {normalize(n) for n in config.UNKNOWN_SOURCE_CLASSES}
    accounted |= {normalize(n) for n in getattr(config, "EXCLUDED_CLASSES", [])}

    unused = sorted(available - accounted)
    if unused:
        print(
            "\nNOTE: these dataset classes are in no list in config.py and are\n"
            "being ignored. Add them to UNKNOWN_SOURCE_CLASSES to use them as\n"
            "negatives, or to EXCLUDED_CLASSES to confirm the omission:\n  "
            + ", ".join(unused)
        )

    excluded = [normalize(n) for n in getattr(config, "EXCLUDED_CLASSES", [])]
    dropped = sorted(name for name in excluded if name in available)
    if dropped:
        print(f"\nDeliberately excluded (see config.py): {', '.join(dropped)}")

    missing = sorted(
        name for name in
        {normalize(n) for n in config.UNKNOWN_SOURCE_CLASSES} - available
    )
    if missing:
        print(
            "\nWARNING: these UNKNOWN_SOURCE_CLASSES are not in the dataset and\n"
            "contribute nothing. Check the spelling against the folder names:\n  "
            + ", ".join(missing)
        )


def add_unknown_class(plan, by_split, had_splits, rng: random.Random) -> None:
    """Build the unknown class, sampled evenly and capped against imbalance."""
    for split in SPLIT_NAMES:
        crop_counts = [len(plan[split][c]) for c in config.TARGET_CROPS]
        if not crop_counts or sum(crop_counts) == 0:
            continue

        average = sum(crop_counts) / len(crop_counts)
        budget = max(1, int(average * config.UNKNOWN_SIZE_MULTIPLIER))

        pools = []
        for source in config.UNKNOWN_SOURCE_CLASSES:
            paths = by_split[split].get(normalize(source), [])
            if paths:
                shuffled = paths[:]
                rng.shuffle(shuffled)
                pools.append(shuffled)

        # Round-robin across the source classes so no single vegetable
        # dominates what "unknown" looks like.
        chosen: list[Path] = []
        index = 0
        while pools and len(chosen) < budget:
            drained = []
            for pool in pools:
                if index < len(pool):
                    chosen.append(pool[index])
                    if len(chosen) >= budget:
                        break
                else:
                    drained.append(pool)
            if len(chosen) >= budget:
                break
            for pool in drained:
                pools.remove(pool)
            index += 1

        plan[split][config.UNKNOWN_CLASS].extend(chosen)

    # Your own background photos, if any, are worth more than the produce
    # negatives because they show the actual scene the camera will look at.
    backgrounds = gather_background_images()
    if backgrounds:
        train_bg, val_bg, test_bg = split_list(backgrounds, config.SPLIT_RATIOS, rng)
        plan["train"][config.UNKNOWN_CLASS].extend(train_bg)
        plan["val"][config.UNKNOWN_CLASS].extend(val_bg)
        plan["test"][config.UNKNOWN_CLASS].extend(test_bg)
        print(f"Folded in {len(backgrounds)} background images from {config.BACKGROUND_DIR}")


def write_plan(plan) -> None:
    """Resize every image to the model's input size and write it out.

    The resizing happens HERE, once, through cropnet.preprocess.load_image --
    the exact function predict.py uses on a new photo. That is the point of
    doing it at prepare time rather than letting Keras resize on the fly:
    Keras resizes with non-antialiased bilinear while Pillow antialiases, so
    the two produce visibly different pixels when downscaling a 1200x800 photo
    to 96x96. Training on one and predicting on the other cost about 3.5 points
    of accuracy, and it silently broke the promise that the desktop harness
    sees what the ESP32 will see.

    Output is PNG, not JPEG: these are already-small images and re-encoding
    them as JPEG would bake a second round of compression artefacts into the
    training data.
    """
    if config.DATA_PREPARED.exists():
        shutil.rmtree(config.DATA_PREPARED)

    total = sum(len(paths) for split in SPLIT_NAMES for paths in plan[split].values())
    done = 0
    skipped = []

    for split in SPLIT_NAMES:
        for class_name, paths in plan[split].items():
            destination = config.DATA_PREPARED / split / class_name
            destination.mkdir(parents=True, exist_ok=True)
            for i, source in enumerate(paths):
                try:
                    pixels = load_image(source, config.IMAGE_SIZE)
                except Exception as error:
                    # A handful of files in public datasets are truncated or
                    # mislabelled. Skip them loudly rather than dying halfway.
                    skipped.append(f"{source.name}: {type(error).__name__}")
                    continue
                # Rename on write: files from 28 different source folders can
                # collide on names like "Image_1.jpg" once merged into unknown.
                Image.fromarray(pixels).save(destination / f"{class_name}_{i:05d}.png")
                done += 1

            print(f"  {split}/{class_name}: {len(paths)} images", end="\r")

    print(f"  resized {done}/{total} images to "
          f"{config.IMAGE_SIZE}x{config.IMAGE_SIZE}" + " " * 20)

    if skipped:
        print(f"\n  skipped {len(skipped)} unreadable file(s):")
        for entry in skipped[:10]:
            print(f"    {entry}")


def report(plan) -> None:
    classes = config.class_names()
    width = max(len(c) for c in classes)

    print()
    print(f"{'class':<{width}}  {'train':>7} {'val':>7} {'test':>7} {'total':>7}")
    print("-" * (width + 34))

    totals = {s: 0 for s in SPLIT_NAMES}
    for class_name in classes:
        counts = [len(plan[s].get(class_name, [])) for s in SPLIT_NAMES]
        for split, count in zip(SPLIT_NAMES, counts):
            totals[split] += count
        print(f"{class_name:<{width}}  {counts[0]:>7} {counts[1]:>7} {counts[2]:>7} {sum(counts):>7}")

    print("-" * (width + 34))
    grand = sum(totals.values())
    print(f"{'TOTAL':<{width}}  {totals['train']:>7} {totals['val']:>7} {totals['test']:>7} {grand:>7}")

    train_counts = [len(plan["train"].get(c, [])) for c in classes]
    if train_counts and min(train_counts) > 0:
        imbalance = max(train_counts) / min(train_counts)
        print(f"\nTrain-set imbalance (largest/smallest class): {imbalance:.2f}x")
        if imbalance > 3.0:
            print(
                "  Warning: that is high enough to bias the model. Consider lowering\n"
                "  config.UNKNOWN_SIZE_MULTIPLIER or adding images to the thin classes."
            )

    thin = [c for c, n in zip(classes, train_counts) if n < 60]
    if thin:
        print(
            "\nThese classes have under 60 training images and will be the weakest\n"
            "in the confusion matrix: " + ", ".join(thin)
        )


def main() -> None:
    rng = random.Random(config.SEED)

    print(f"Reading raw dataset from {config.DATA_RAW}")
    plan = build_plan(config.DATA_RAW, rng)

    print(f"Writing prepared dataset to {config.DATA_PREPARED}")
    write_plan(plan)

    write_labels(config.LABELS_FILE, config.class_names())
    print(f"Wrote {config.LABELS_FILE} with {config.NUM_CLASSES} classes")

    report(plan)
    print("\nDone. Next: python scripts/train.py")


if __name__ == "__main__":
    main()
