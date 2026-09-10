"""
Strip the raw dataset down to what the rig can actually see.

    python scripts/curate_dataset.py --dry-run   # report, move nothing
    python scripts/curate_dataset.py             # apply
    python scripts/curate_dataset.py --restore   # put everything back

Reads:  data/raw/{train,validation,test}/<class>/   the Kaggle 36-class dataset
        data/raw/extra/<crop>/                     the Open Images top-up
Moves:  data/raw/_removed_offtarget/...            everything not wanted any more

Nothing is deleted. --restore reverses the whole thing, so this is safe to run
and safe to change your mind about.

What gets removed, and why
==========================

The rig is a fixed ESP32-CAM looking into a small white thermocol box holding
exactly one item. See my images/ for what that looks like: one whole raw crop,
close up, on a bright white speckled floor against a white wall. Under a warm
lamp, slightly blurred, washed out.

This script has two independent jobs, selected with --only. Read the first one
before using it, because it is the one that is usually wrong.

1. The non-target produce classes  (--only classes)   NOT USED BY DEFAULT
------------------------------------------------------------------------
beetroot, cabbage, carrot, grapes, kiwi, mango, orange, pomegranate and the rest
are folded into the `unknown` class as negatives -- real produce photographs that
are not one of the 8 crops, which is a genuinely good way to teach a model to say
"not one of mine".

Removing them looks defensible: the box is loaded by hand with known produce, so
why spend model capacity separating carrots from cabbages? It was tried, and it
was a mistake worth recording, because the argument sounds right and the
measurement disagrees.

With the negatives removed, `unknown` can only mean "empty tray". A real
pomegranate -- round, red, filling the frame, and one of the classes that had
just been deleted -- was then reported as an apple at 85% confidence, three
times out of three. The model had learned "crop versus bare thermocol" and
nothing about which crop, so anything crop-shaped was forced into one of the 8.

The lesson generalises past pomegranates: `unknown` has to mean "not one of my
8", not "nothing there", and a hand-loaded tray is exactly where a wrong item
turns up. So this removal is available but off by default. If you do run it,
empty config.UNKNOWN_SOURCE_CLASSES to match, and expect confident misreads on
any produce you did not train.

`sweetcorn` is a separate case and stays excluded either way -- it is the same
vegetable as `corn`, so using it as a negative would train the model to reject
real corn.

2. The whole data/raw/extra tree  (--only extras)
-------------------------------------------------
2,411 Open Images and Wikimedia photos, ~100 per crop folded into the training
split, collected to make the model survive arbitrary real-world photographs.

They are removed because they are photographs of a world the camera cannot see,
and because they are contaminated in a way that cannot be filtered cheaply.
Inspecting them by contact sheet:

    potato   roast potatoes, chips, mashed potato, potato salad, stew, and
             sweet potatoes -- a different vegetable in the potato class
    onion    market piles, sacks, garlic bulbs, spring onions, a tomato stall
    corn     popcorn, grilled cobs on plates, a dog eating corn, a cornfield
             with people in it, a supermarket aisle
    apple    candy apples on sticks, orchard trees, market bins, a cartoon

Notably the contamination survives the `_tight` framings, which are supposed to
be one labelled object filling the frame -- because Open Images draws boxes
around potatoes that are sitting in a dish.

Two attempts at filtering this by composition were abandoned. Scoring background
uniformity, brightness and how centred the edge energy was kept 35 of 670 apples
but only 1 of 516 tomatoes, because tomato photographs favour wooden tables and
blue bowls while apple photographs favour white studio sweeps -- the score was
measuring photographic convention, not suitability. Reweighting it and adding a
per-crop relative cut still kept the garlic, the scallions and the roast
potatoes, because what is wrong with those images is semantic, not structural,
and there is no cheap classifier to check against: ImageNet-1k has no onion,
potato, ginger or tomato class.

The measurement that settled it was scoring the six real rig photos. The
"is there a single centred subject" term came out at 0.00 on all six. The filter
was confidently rejecting its own target, which is a good sign the filter is the
wrong tool.

What the extras were actually worth is also on the record, in
config.EXTRA_TRAIN_MAX_PER_CROP: they moved Kaggle test accuracy 88.2% -> 89.8%,
about 1.6 points. The large gain they bought, 32.8% -> 65.5%, was on held-out
Open Images photographs -- a distribution this rig will never produce.

What stays
----------
The Kaggle images for the 8 crops: 68-94 photographs each, mostly one item, varied
backgrounds and lighting. That is the clean core.

The 27 non-target classes, as `unknown` negatives -- see job 1 above.

data/raw/extra_holdout/ is left alone. No script reads it; it was the held-out
Open Images benchmark, and it is no longer the benchmark that matters.

Recommended use
---------------
    python scripts/curate_dataset.py --only extras

That drops the unusable Open Images top-up and leaves everything else in place.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# Where displaced folders go. Inside data/raw/ so .gitignore already covers it.
#
# prepare_data.find_split_dirs() looks for train/validation/test in data/raw/
# and then one level deeper, so this directory does contain names it could in
# principle match on. It does not in practice: that function returns the first
# base that has a `train` child, and data/raw/ itself always does while the
# dataset is present. Worth knowing before renaming anything here.
HOLDING_DIRNAME = "_removed_offtarget"

# Dataset split folder names, as they appear on disk.
SPLIT_DIRNAMES = ("train", "validation", "test", "val", "training", "testing")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize(name: str) -> str:
    """Fold folder names so 'Chilli Pepper' matches 'chilli pepper'.

    Same rule as prepare_data.normalize, so the two agree on what a class is.
    """
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1 for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def target_dataset_names() -> set[str]:
    """The dataset folder names belonging to the 8 crops we detect.

    Read from config.TARGET_CROPS rather than hardcoded, so adding a crop back
    into the class list and re-running --restore then the script does the right
    thing without editing this file.
    """
    return {normalize(source) for source in config.TARGET_CROPS.values()}


def plan_removals() -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Work out what to move. Returns (class_moves, extra_moves).

    Each entry is (source, destination). Nothing touches the disk here -- this
    is what --dry-run prints.
    """
    keep = target_dataset_names()
    holding = config.DATA_RAW / HOLDING_DIRNAME

    class_moves: list[tuple[Path, Path]] = []
    for split_name in SPLIT_DIRNAMES:
        split_dir = config.DATA_RAW / split_name
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            if normalize(class_dir.name) in keep:
                continue
            class_moves.append(
                (class_dir, holding / split_name / class_dir.name)
            )

    extra_moves: list[tuple[Path, Path]] = []
    extra_root = config.DATA_RAW / "extra"
    if extra_root.is_dir():
        for crop_dir in sorted(extra_root.iterdir()):
            if crop_dir.is_dir():
                extra_moves.append((crop_dir, holding / "extra" / crop_dir.name))

    return class_moves, extra_moves


def move_all(moves: list[tuple[Path, Path]]) -> int:
    """Move directories, merging into the destination if it already exists."""
    moved = 0
    for source, destination in moves:
        if not source.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            # Merge rather than fail. Happens when the script is re-run after a
            # partial restore.
            for item in source.rglob("*"):
                if item.is_file():
                    target = destination / item.relative_to(source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        shutil.move(str(item), str(target))
            shutil.rmtree(source, ignore_errors=True)
        else:
            shutil.move(str(source), str(destination))
        moved += 1
    return moved


def apply_removals(class_moves, extra_moves) -> None:
    moved = move_all(class_moves)
    print(f"\nMoved {moved} off-target class folder(s).")
    moved = move_all(extra_moves)
    print(f"Moved {moved} extra-image folder(s).")
    print(f"\nEverything is under {config.DATA_RAW / HOLDING_DIRNAME}")
    print("Run with --restore to undo.")


def restore(only: str = "all") -> None:
    """Move things in the holding area back where they came from.

    `only` selects a group: "classes" for the off-target produce folders,
    "extras" for the Open Images top-up, "all" for both.
    """
    holding = config.DATA_RAW / HOLDING_DIRNAME
    if not holding.is_dir():
        raise SystemExit(f"Nothing to restore -- {holding} does not exist.")

    moves: list[tuple[Path, Path]] = []
    for group_dir in sorted(holding.iterdir()):
        if not group_dir.is_dir():
            continue
        is_extras = group_dir.name == "extra"
        if only == "classes" and is_extras:
            continue
        if only == "extras" and not is_extras:
            continue
        for class_dir in sorted(group_dir.iterdir()):
            if class_dir.is_dir():
                moves.append(
                    (class_dir, config.DATA_RAW / group_dir.name / class_dir.name)
                )

    if not moves:
        raise SystemExit(
            f"Nothing to restore for --only {only} under {holding}."
        )

    total = sum(count_images(source) for source, _ in moves)
    moved = move_all(moves)

    # Clean up the now-empty scaffolding so a later --dry-run reads honestly.
    for group_dir in sorted(holding.iterdir(), reverse=True):
        if group_dir.is_dir() and not any(group_dir.iterdir()):
            group_dir.rmdir()
    if holding.is_dir() and not any(holding.iterdir()):
        holding.rmdir()

    print(f"Restored {moved} folder(s), {total} images.")
    print("\nRemember to put the config back too, or prepare_data.py will")
    print("ignore what you just restored:")
    if only in ("all", "classes"):
        print("  - UNKNOWN_SOURCE_CLASSES  (the produce negatives)")
        print("  - EXCLUDED_CLASSES        (drop them back to just 'sweetcorn')")
    if only in ("all", "extras"):
        print("  - EXTRA_TRAIN_DIRS        (the Open Images top-up)")
    print("\nNext: python scripts/prepare_data.py")


def report(class_moves, extra_moves) -> None:
    keep = sorted(target_dataset_names())
    print(f"Raw dataset: {config.DATA_RAW}")
    print(f"Keeping the {len(keep)} target crops: {', '.join(keep)}\n")

    if class_moves:
        by_split: dict[str, list[tuple[str, int]]] = {}
        for source, _ in class_moves:
            by_split.setdefault(source.parent.name, []).append(
                (source.name, count_images(source))
            )
        print("Off-target produce classes to remove:")
        for split_name, entries in by_split.items():
            total = sum(n for _, n in entries)
            print(f"  {split_name:<11} {len(entries):>2} classes, {total:>5} images")
            names = ", ".join(name for name, _ in sorted(entries))
            print(f"              {names}")
    else:
        print("Off-target produce classes: none found (already removed?)")

    print()
    if extra_moves:
        total = sum(count_images(source) for source, _ in extra_moves)
        print(f"Open Images / Commons extras to remove: "
              f"{len(extra_moves)} crops, {total} images")
        for source, _ in extra_moves:
            print(f"  {source.name:<10} {count_images(source):>5}")
    else:
        print("Open Images / Commons extras: none found (already removed?)")

    remaining = 0
    for split_name in ("train", "validation", "test"):
        split_dir = config.DATA_RAW / split_name
        if not split_dir.is_dir():
            continue
        kept = [d for d in sorted(split_dir.iterdir())
                if d.is_dir() and normalize(d.name) in target_dataset_names()]
        counts = {d.name: count_images(d) for d in kept}
        remaining += sum(counts.values())
        print(f"\nRemaining in {split_name}: {sum(counts.values())} images")
        print("  " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    print(f"\nTotal crop images kept: {remaining}")

    if class_moves:
        print("\nWARNING: removing the produce classes leaves `unknown` meaning")
        print("only \"empty tray\". Measured consequence: a real pomegranate was")
        print("reported as apple at 85% confidence. See the module docstring.")
        print("Empty config.UNKNOWN_SOURCE_CLASSES to match, or restore with")
        print("  python scripts/curate_dataset.py --restore --only classes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove dataset images for scenarios the rig cannot produce.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would move, change nothing")
    parser.add_argument("--restore", action="store_true",
                        help="move things back and exit")
    parser.add_argument(
        "--only", choices=("all", "classes", "extras"), default="all",
        help="which group to act on. 'classes' is the off-target produce "
             "folders, 'extras' is the Open Images top-up. They are independent "
             "decisions and are usually wanted separately: the extras are "
             "unusable data, whereas the produce classes are good data whose "
             "usefulness depends on what `unknown` is supposed to mean.",
    )
    args = parser.parse_args()

    if args.restore:
        restore(args.only)
        return

    if not config.DATA_RAW.is_dir():
        raise SystemExit(f"No raw dataset at {config.DATA_RAW}")

    class_moves, extra_moves = plan_removals()
    if args.only == "classes":
        extra_moves = []
    elif args.only == "extras":
        class_moves = []

    report(class_moves, extra_moves)

    if args.dry_run:
        print("\nDry run -- nothing moved. Re-run without --dry-run to apply.")
        return

    if not class_moves and not extra_moves:
        print("\nNothing to do.")
        return

    apply_removals(class_moves, extra_moves)


if __name__ == "__main__":
    main()
