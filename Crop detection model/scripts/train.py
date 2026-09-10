"""
Train the crop classifier.

    python scripts/train.py

Two-phase transfer learning on an ImageNet-pretrained MobileNet backbone:

    phase 1  backbone frozen, train the new head        (fast, stable)
    phase 2  unfreeze the top of the backbone, low LR   (adapts features)

Phase 1 exists because a randomly initialised head produces large, meaningless
gradients. Fine-tuning straight away would push those gradients back into the
pretrained weights and destroy the very features we came for. Training the head
first means that by the time the backbone unfreezes, the gradients flowing into
it are already sensible.

Output: models/model.keras plus training curves in outputs/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from cropnet.augment import RandomWhiteBalance
from cropnet.labels import write_labels


def verify_prepared_size() -> None:
    """Fail fast if data/prepared/ was built for a different IMAGE_SIZE.

    prepare_data.py already resized everything to the model's input size, so
    Keras does no interpolation at all below -- which is the whole point, since
    Keras and Pillow resize differently. A stale prepared/ folder at the wrong
    size would silently reintroduce that mismatch, so check rather than trust.
    """
    from PIL import Image

    sample = next(
        (p for p in (config.DATA_PREPARED / "train").rglob("*.png")), None
    )
    if sample is None:
        return

    with Image.open(sample) as image:
        width, height = image.size

    if (width, height) != (config.IMAGE_SIZE, config.IMAGE_SIZE):
        raise SystemExit(
            f"data/prepared/ holds {width}x{height} images but config.IMAGE_SIZE "
            f"is {config.IMAGE_SIZE}.\nRe-run scripts/prepare_data.py."
        )


def build_datasets(tf):
    """Load the prepared folders as tf.data pipelines.

    The images on disk are already exactly IMAGE_SIZE square, resized by
    prepare_data.py through the same Pillow code path predict.py uses, so the
    image_size argument below is a no-op rather than a second, different
    resize. crop_to_aspect_ratio stays set as a belt-and-braces guard for the
    case where someone points this at unprepared data.
    """
    verify_prepared_size()

    common = dict(
        image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
        crop_to_aspect_ratio=True,
        seed=config.SEED,
    )

    splits = {}
    for split in ("train", "val", "test"):
        directory = config.DATA_PREPARED / split
        if not directory.exists():
            raise SystemExit(
                f"Missing {directory}. Run scripts/prepare_data.py first."
            )
        splits[split] = tf.keras.utils.image_dataset_from_directory(
            directory,
            shuffle=(split == "train"),
            **common,
        )

    class_names = splits["train"].class_names
    expected = config.class_names()
    if class_names != expected:
        raise SystemExit(
            "Class order on disk does not match config.class_names().\n"
            f"  on disk: {class_names}\n"
            f"  config : {expected}\n"
            "The order defines crop_id for ESP-NOW, so this must not drift. "
            "Re-run scripts/prepare_data.py."
        )

    autotune = tf.data.AUTOTUNE
    for split in splits:
        splits[split] = splits[split].prefetch(autotune)

    return splits, class_names


def build_augmentation(tf):
    """Assemble the training-only augmentation block.

    Ordered to mirror a real imaging chain: geometry first, then the lens
    (defocus), then the sensor and its white balance. The block is named
    "augmentation" because export_tflite.py strips it by that name, so nothing
    here costs flash or inference time on the device.

    Everything operates on raw 0-255 pixels -- the model's Rescaling layer runs
    after this block, not before.
    """
    layers = [
        tf.keras.layers.RandomRotation(config.AUG_ROTATION),
        tf.keras.layers.RandomZoom(config.AUG_ZOOM),
        tf.keras.layers.RandomTranslation(config.AUG_TRANSLATION, config.AUG_TRANSLATION),
    ]
    if config.AUG_HORIZONTAL_FLIP:
        layers.insert(0, tf.keras.layers.RandomFlip("horizontal"))

    # Defocus, applied after geometry so it blurs the final framing rather than
    # something that is about to be resampled again.
    #
    # This also closes a shortcut. The unknown class is generated from a real
    # photograph and carries genuine blur and JPEG artefacts, while the crop
    # classes are clean web photos; without blurring every class, sharpness
    # itself would separate "crop" from "empty" and the model would score well
    # for a reason that does not exist on the device.
    blur_sigma = getattr(config, "AUG_BLUR_SIGMA", 0)
    if blur_sigma:
        layers.append(
            tf.keras.layers.RandomGaussianBlur(
                factor=1.0, kernel_size=3, sigma=(0.0, blur_sigma)
            )
        )

    layers.extend([
        tf.keras.layers.RandomBrightness(config.AUG_BRIGHTNESS, value_range=(0, 255)),
        tf.keras.layers.RandomContrast(config.AUG_CONTRAST),
    ])

    # White balance last, so it acts on the final pixels the way a camera's AWB
    # would. See cropnet/augment.py for why this is not RandomHue.
    white_balance = getattr(config, "AUG_WHITE_BALANCE", 0)
    if white_balance:
        layers.append(
            RandomWhiteBalance(
                factor=white_balance, value_range=(0, 255), seed=config.SEED
            )
        )

    # Hue and saturation jitter stay available but measure badly; see AUG_HUE in
    # config.py before switching them on.
    if getattr(config, "AUG_HUE", 0):
        layers.append(tf.keras.layers.RandomHue(config.AUG_HUE, value_range=(0, 255)))
    if getattr(config, "AUG_SATURATION", 0):
        layers.append(
            tf.keras.layers.RandomSaturation(config.AUG_SATURATION, value_range=(0, 255))
        )

    return tf.keras.Sequential(layers, name="augmentation")


def compute_class_weights(tf, splits, class_names):
    """Weight each class inversely to its size, normalised to mean 1.

    Returns None when weighting is disabled, which is what model.fit expects.
    """
    if not getattr(config, "USE_CLASS_WEIGHTS", False):
        return None

    counts = []
    for name in class_names:
        folder = config.DATA_PREPARED / "train" / name
        counts.append(len(list(folder.glob("*.png"))) if folder.exists() else 0)

    total = sum(counts)
    if total == 0 or min(counts) == 0:
        return None

    n = len(counts)
    weights = {i: total / (n * c) for i, c in enumerate(counts)}
    spread = max(weights.values()) / min(weights.values())
    print(f"Class weights active (heaviest/lightest = {spread:.2f}x)")
    return weights


def build_backbone(tf):
    """Create the pretrained feature extractor.

    Always returns a model whose output is a pooled feature vector, whether or
    not the backbone is truncated, so the classifier head does not need to care.

    ImageNet weights are only published for a fixed set of input widths, and 96
    is one of them for both MobileNet families -- which is part of why 96x96 is
    the chosen input size rather than some other number that fits.
    """
    truncate = getattr(config, "TRUNCATE_AT", None)

    kwargs = dict(
        input_shape=config.INPUT_SHAPE,
        alpha=config.ALPHA,
        include_top=False,
        weights="imagenet",
    )

    if config.BACKBONE == "mobilenet_v2":
        base = tf.keras.applications.MobileNetV2(**kwargs, pooling=None if truncate else "avg")
    elif config.BACKBONE == "mobilenet_v1":
        if truncate:
            raise SystemExit("TRUNCATE_AT is only supported for mobilenet_v2.")
        base = tf.keras.applications.MobileNet(**kwargs, pooling="avg")
    else:
        raise SystemExit(f"Unknown BACKBONE in config.py: {config.BACKBONE!r}")

    if not truncate:
        return base

    try:
        cut = base.get_layer(truncate).output
    except ValueError:
        candidates = [layer.name for layer in base.layers if "project_BN" in layer.name
                      or "expand_relu" in layer.name]
        raise SystemExit(
            f"TRUNCATE_AT={truncate!r} is not a layer in this backbone.\n"
            f"Sensible cut points: {', '.join(candidates[-8:])}"
        )

    pooled = tf.keras.layers.GlobalAveragePooling2D(name="backbone_pool")(cut)
    trimmed = tf.keras.Model(base.input, pooled, name="backbone_truncated")
    print(f"Truncated backbone at {truncate!r}: "
          f"{len(trimmed.layers)} of {len(base.layers)} layers kept")
    return trimmed


def build_model(tf, backbone):
    """Assemble augmentation + rescaling + backbone + classifier head.

    The Rescaling layer is inside the model on purpose. It means the exported
    TFLite model accepts raw pixel bytes, so the ESP32 firmware hands the camera
    buffer almost straight to the interpreter instead of reimplementing
    MobileNet's /127.5 - 1 normalisation in C and getting it subtly wrong.
    """
    inputs = tf.keras.Input(shape=config.INPUT_SHAPE, name="image")

    x = build_augmentation(tf)(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.Dropout(config.DROPOUT, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        config.NUM_CLASSES, activation="softmax", name="predictions"
    )(x)

    return tf.keras.Model(inputs, outputs, name="farmfrost_crop_classifier")


def compile_model(tf, model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=getattr(config, "LABEL_SMOOTHING", 0.0)
        ),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
        ],
    )


def callbacks_for(tf, phase: str, checkpoint=None):
    callbacks = [checkpoint] if checkpoint else []
    return callbacks + [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, config.EARLY_STOPPING_PATIENCE // 2),
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(config.OUTPUTS_DIR / f"history_{phase}.csv")
        ),
    ]


def plot_history(histories: dict) -> None:
    """Save accuracy/loss curves. Skipped silently if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping training curves.")
        return

    merged = {}
    for phase in ("head", "finetune"):
        for key, values in histories.get(phase, {}).items():
            merged.setdefault(key, []).extend(values)

    if not merged:
        return

    boundary = len(histories.get("head", {}).get("loss", []))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, (metric, title) in zip(axes, [("accuracy", "Accuracy"), ("loss", "Loss")]):
        if metric in merged:
            axis.plot(merged[metric], label=f"train {metric}")
        if f"val_{metric}" in merged:
            axis.plot(merged[f"val_{metric}"], label=f"val {metric}")
        if boundary:
            axis.axvline(boundary - 0.5, color="grey", linestyle="--", linewidth=1)
            axis.text(boundary - 0.4, axis.get_ylim()[0], " fine-tune", fontsize=8, color="grey")
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.legend()
        axis.grid(alpha=0.3)

    figure.tight_layout()
    path = config.OUTPUTS_DIR / "training_curves.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(f"Saved training curves to {path}")


def main() -> None:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(config.SEED)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"TensorFlow {tf.__version__}")
    print(f"Backbone: {config.BACKBONE} alpha={config.ALPHA} input={config.INPUT_SHAPE}")

    splits, class_names = build_datasets(tf)
    write_labels(config.LABELS_FILE, class_names)
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")

    class_weight = compute_class_weights(tf, splits, class_names)

    backbone = build_backbone(tf)
    backbone.trainable = False
    model = build_model(tf, backbone)
    model.summary()

    histories = {}

    # One checkpoint instance shared by both phases. EarlyStopping only restores
    # the best epoch *within* a phase, so without this a fine-tuning run that
    # makes things worse would silently overwrite a better head-only model --
    # which is exactly what happens on small datasets. Reusing the same callback
    # object carries its best-so-far across the phase boundary.
    best_path = config.MODELS_DIR / "_best.keras"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(best_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=0,
    )

    print("\n=== Phase 1: training the classifier head (backbone frozen) ===")
    compile_model(tf, model, config.HEAD_LR)
    history = model.fit(
        splits["train"],
        validation_data=splits["val"],
        epochs=config.HEAD_EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks_for(tf, "head", checkpoint),
    )
    histories["head"] = history.history
    head_best = checkpoint.best

    if config.FINETUNE_EPOCHS > 0:
        print("\n=== Phase 2: fine-tuning the top of the backbone ===")
        backbone.trainable = True

        frozen_until = max(0, len(backbone.layers) - config.FINETUNE_UNFREEZE_LAYERS)
        for layer in backbone.layers[:frozen_until]:
            layer.trainable = False

        # BatchNorm stays frozen even in the unfrozen region. With batches this
        # small the batch statistics are noisy, and letting them update is a
        # classic way to make fine-tuning accuracy collapse for no clear reason.
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        trainable = sum(1 for layer in backbone.layers if layer.trainable)
        print(f"Unfroze {trainable} of {len(backbone.layers)} backbone layers")

        compile_model(tf, model, config.FINETUNE_LR)
        history = model.fit(
            splits["train"],
            validation_data=splits["val"],
            epochs=config.FINETUNE_EPOCHS,
            class_weight=class_weight,
            callbacks=callbacks_for(tf, "finetune", checkpoint),
        )
        histories["finetune"] = history.history

        if checkpoint.best > head_best:
            print(f"\nFine-tuning improved val_accuracy "
                  f"{head_best:.4f} -> {checkpoint.best:.4f}")
        else:
            print(f"\nFine-tuning did NOT improve on the head-only model "
                  f"({head_best:.4f}); keeping the phase 1 weights.")
            print("  On a dataset this small that is common. Lower FINETUNE_LR,\n"
                  "  reduce FINETUNE_UNFREEZE_LAYERS, or set FINETUNE_EPOCHS = 0\n"
                  "  in config.py to skip the phase entirely and save the time.")

    # Load the best weights seen across BOTH phases, not just the last one.
    if best_path.exists():
        print(f"\nLoading best checkpoint (val_accuracy {checkpoint.best:.4f})")
        model = tf.keras.models.load_model(best_path)

    print("\n=== Test set (Keras float model) ===")
    results = model.evaluate(splits["test"], return_dict=True)
    for name, value in results.items():
        print(f"  {name}: {value:.4f}")

    model.save(config.KERAS_MODEL)
    print(f"\nSaved {config.KERAS_MODEL}")

    best_path.unlink(missing_ok=True)

    (config.OUTPUTS_DIR / "float_test_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    plot_history(histories)

    print("\nDone. Next: python scripts/export_tflite.py")


if __name__ == "__main__":
    main()
