"""
Central configuration for the FarmFrost crop-detection model.

Every script in this folder reads its settings from here. If you want to change
the input size, the backbone, the class list, or where data lives, change it in
this file only -- nothing else should need editing.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_RAW = ROOT / "data" / "raw"            # dataset exactly as downloaded
DATA_PREPARED = ROOT / "data" / "prepared"  # filtered + split, ready for training
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

KERAS_MODEL = MODELS_DIR / "model.keras"
TFLITE_MODEL = MODELS_DIR / "model_int8.tflite"
C_ARRAY_SOURCE = MODELS_DIR / "model_data.cc"
C_ARRAY_HEADER = MODELS_DIR / "model_data.h"
LABELS_FILE = MODELS_DIR / "labels.txt"

# The ESP32-CAM sketch compiles model_data.h/.cpp from its own folder, so the
# export copies them there as well as into models/. Keeping this automatic is
# not tidiness: the firmware was found carrying a stale 9-class model while the
# trained one had 5, and nothing about that fails loudly -- the sketch compiles,
# boots, and reports confident nonsense against the wrong label table.
#
# Set to None to skip the copy.
FIRMWARE_DIR = ROOT.parent / "ESP Firmware" / "ESP32_CAM"

# --------------------------------------------------------------------------
# Input geometry
#
# 96x96 RGB is the largest input that keeps inference on a plain ESP32 in the
# 1-3 second range. The ESP32-CAM will capture a larger frame, centre-crop it to
# a square, and downscale to this size -- the desktop preprocessing in
# cropnet/preprocess.py does exactly the same thing so the two agree.
# --------------------------------------------------------------------------

IMAGE_SIZE = 96
IMAGE_CHANNELS = 3
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)

# --------------------------------------------------------------------------
# Classes
#
# TARGET_CROPS maps our internal crop name -> the folder name used by the Kaggle
# "Fruit and Vegetable Image Recognition" dataset (kritikseth). Our name is what
# ends up in labels.txt and in the ESP-NOW crop table; the dataset name is only
# used while preparing data.
#
# The whole class set is now scoped to one physical setup: a fixed ESP32-CAM
# looking into a small white thermocol box holding exactly one item. See
# my images/ for what the camera actually returns. Everything below follows from
# that -- 8 crops, plus `unknown` meaning "the box is empty", and no training
# data for anything that cannot happen inside the box.
# --------------------------------------------------------------------------

TARGET_CROPS = {
    "banana": "banana",
    "lemon": "lemon",
    "onion": "onion",
    "tomato": "tomato",
}

# The "unknown" class stops the model from confidently reporting a crop when
# there is no crop to report. The Main ESP32 must never change fan speed on a
# misread, so this class is what gives the confidence threshold in the firmware
# spec something real to act on.
#
# It means: NOT ONE OF THE 8 CROPS. That covers two different situations, and the
# class is built from a source for each:
#
#   some other produce   -> UNKNOWN_SOURCE_CLASSES, the Kaggle set's other 27
#                           classes. A pomegranate is not a crop we handle.
#   nothing at all       -> BACKGROUND_DIR, photographs of the empty box.
#
# Both are needed, and the test images in my images/ show why. Three of them are
# pomegranates, which no amount of empty-box training would reject -- a
# pomegranate is round, red and fills the frame, so a model that only knows
# "crop" versus "empty tray" reports it as an apple with 85% confidence. That was
# measured, not assumed. Equally, the produce negatives alone never show a bare
# tray, so an emptied tray used to come out as whatever the last crop looked
# most like.
UNKNOWN_CLASS = "empty"
INCLUDE_UNKNOWN = True

# Produce classes folded into `unknown` as negatives: everything in the dataset
# that is not one of the 8 targets. These are the most useful negatives available
# -- real produce photographs that are not one of our crops -- and they are free,
# since the images are already downloaded.
#
# `pomegranate` earns its place here explicitly. It is the negative the rig is
# actually tested with, and it is visually the closest of all of them to apple
# and tomato.
UNKNOWN_SOURCE_CLASSES = []  # nothing but the 4 crops is ever placed in the box

# Dataset classes deliberately used as NEITHER a target nor a negative.
#
# "sweetcorn" is the same vegetable as our "corn" target. Training the model to
# answer "unknown" for it would directly teach it to reject real corn, which is
# the opposite of what we want. It is dropped instead. If it turns out the
# dataset's corn/sweetcorn folders show visually different things (cob vs
# loose kernels), the better move is to fold sweetcorn into corn as extra
# training data -- add it as a second entry rather than leaving it here.
#
# prepare_data.py reports any dataset class missing from all three lists, so
# nothing gets discarded silently.
EXCLUDED_CLASSES = ["sweetcorn"]

# data/raw/background/ holds photographs of the empty box, folded into `unknown`
# alongside the produce negatives above. This is the "nothing at all" half of the
# class, and without it the model has never seen a bare tray.
#
# scripts/make_background.py generates these from my images/environment/. Anything
# you drop in here by hand is used as well, and frames captured through the
# ESP32-CAM itself, at the real mounting distance, are worth far more than
# generated crops of one phone photo.
BACKGROUND_DIR = DATA_RAW / "background"

# Extra images per crop, folded into the TRAIN split only. Deliberately empty.
#
# This used to point at data/raw/extra/, 2,411 Open Images and Wikimedia photos
# collected to make the model survive arbitrary real-world photographs of
# produce. They are no longer used, for two independent reasons.
#
# The first is that they are photographs of a world this camera cannot see:
# apples on a tree, a market bin of two hundred onions, a cornfield with people
# in it. Training on them widens each class until "onion" means "anything
# onion-ish anywhere", which costs accuracy on the one setting that does occur.
#
# The second is that they are contaminated, and not slightly. From contact
# sheets of the supposedly clean `_tight` framings -- one labelled object filling
# the frame -- the potato class contained roast potatoes, chips, mashed potato,
# potato salad, stew and sweet potatoes, which are a different vegetable; the
# onion class contained garlic bulbs and spring onions; corn contained popcorn,
# grilled cobs and a dog. Open Images draws boxes around potatoes that are
# sitting in a dish, so tight framing does not imply raw whole produce. The
# three-stage filter described in this comment's previous version -- label
# filter, ImageNet visual filter, hand review -- demonstrably did not catch it.
#
# Two attempts to filter it here by composition were abandoned. Scoring
# background uniformity, brightness and how centred the edge energy was kept 35
# of 670 apples but 1 of 516 tomatoes, because tomato photographs favour wooden
# tables and blue bowls while apple photographs favour white studio sweeps -- the
# score measured photographic convention, not suitability. The decisive
# measurement was running that score over the six real rig photos in my images/:
# the "single centred subject" term came out at 0.00 on all six. Filtering
# semantics with a structural heuristic does not work, and there is no cheap
# classifier to check against, because ImageNet-1k has no onion, potato, ginger
# or tomato class.
#
# What they were worth is recorded under EXTRA_TRAIN_MAX_PER_CROP below: 1.6
# points on the Kaggle test split. The large gain, 32.8% -> 65.5%, was on
# held-out Open Images photographs, which is not a distribution this rig will
# ever produce.
#
# To put them back: restore the line below and run
# `python scripts/curate_dataset.py --restore`.
# Rig-domain composites: produce photographed against the real box interior,
# generated by scripts/make_rig_composites.py. Every class gets them, including
# `unknown`, and that universality is the entire point.
#
# Before these existed, images taken inside the box appeared in exactly ONE
# class -- `unknown`, via the empty-box crops in BACKGROUND_DIR -- while the 8
# crop classes held only Kaggle web photos. The cheapest rule separating those
# classes is "thermocol texture -> unknown", and that is what the model learned.
# Measured on the real rig photos:
#
#     lemon in the box     -> unknown 90.2%    (same lemon as a web photo: lemon 99.6%)
#     tomato in the box    -> unknown 79.7%    (same tomato as a web photo: TOMATO)
#     pomegranate in box   -> potato  79.3%    accepted, and wrong
#
# The answer is not to drop the empty-box images -- recognising an empty box is
# a real requirement. It is to make background useless as a cue by giving every
# class the same background, which forces the model onto the object itself.
EXTRA_TRAIN_DIRS = {}  # scripts/build_rig_dataset.py writes data/prepared directly

# EXTRA_TRAIN_DIRS = {c: DATA_RAW / "extra" / c for c in TARGET_CROPS}

# Cap on the produce-negative half of the unknown class, as a multiple of the
# average crop class size. The 27 source classes together hold far more images
# than any single crop; left alone that imbalance would teach the model to answer
# "unknown" for everything. prepare_data.py samples evenly across the source
# classes up to this budget.
#
# The background images are added ON TOP of this budget, not inside it, so the
# unknown class ends up at roughly (1.0 x average crop) + however many
# backgrounds exist. Keep scripts/make_background.py --count modest for that
# reason; 60 keeps the train-set imbalance near the 1.76x at which the
# class-weighting and label-smoothing experiments below were measured.
UNKNOWN_SIZE_MULTIPLIER = 1.0


def class_names():
    """The model's output classes, in the fixed order used everywhere.

    Plain alphabetical, including `empty`. The order defines the crop_id byte
    sent over ESP-NOW, so it must stay stable -- and it must also match the
    order Keras assigns when it reads the folders, which is alphabetical. The
    two used to disagree, with `unknown` appended last; sorting everything
    together removes that trap.
    """
    names = list(TARGET_CROPS)
    if INCLUDE_UNKNOWN:
        names.append(UNKNOWN_CLASS)
    return sorted(names)


NUM_CLASSES = len(class_names())

# --------------------------------------------------------------------------
# Architecture
#
# Switching backbone is a one-line change here. If flash or latency turns out
# tighter on real hardware, ALPHA = 0.25 gives a 305 KB model at roughly 84%
# int8 accuracy, against 968 KB and 88% for the current setting.
# --------------------------------------------------------------------------

# MobileNetV1 rather than V2, chosen on measured int8 accuracy -- which is the
# only accuracy that matters, since int8 is what runs on the ESP32.
#
# MobileNetV2 is the better model in float and the worse model on the device.
# Its depthwise convolutions and linear bottlenecks have activation ranges that
# per-tensor int8 quantization handles badly, so post-training quantization
# throws away most of its advantage. Measured on this 9-class dataset:
#
#     backbone              float     int8     drop    predictions changed
#     MobileNetV1 a=0.5     92.1%    91.5%    -0.6%     4.5%
#     MobileNetV2 a=0.35    94.8%    82.4%   -12.4%    15.3%
#
# More calibration data does not help -- 300, 633 and 672 representative images
# all gave V2 the same ~84%. The loss is structural, not a calibration problem.
#
# The cost of V1 a=0.5 is size (968 KB vs 624 KB) and inference time. Both are
# affordable on a 4 MB board; 9 points of on-device accuracy is not.
#
# Quantization-aware training would likely let V2 keep its float accuracy, and
# is the one route to beating this. It needs tensorflow-model-optimization plus
# legacy-Keras mode, and a QuantizeConfig for the Rescaling layer, which
# quantize_model rejects. Worth revisiting only if 91.5% proves insufficient.
BACKBONE = "mobilenet_v1"   # "mobilenet_v2" | "mobilenet_v1"
ALPHA = 0.5
DROPOUT = 0.2

# Optional: cut the backbone short and pool from an earlier feature map.
#
# MobileNetV2 ends with a 1x1 convolution to 1280 channels that Keras does NOT
# scale down with alpha -- it is 1280 wide whether alpha is 0.35 or 1.0. At
# alpha=0.35 that single layer is most of the model. Cutting it costs some
# accuracy and saves a lot of flash.
#
# Measured int8 sizes for this 13-class model at 96x96:
#
#     None                        618 KB   full backbone, best accuracy
#     "block_16_project_BN"       430 KB   drops only the 1280-wide head
#     "block_13_expand_relu"      206 KB   also drops real depth
#
# For reference, BACKBONE="mobilenet_v1" with ALPHA=0.25 measures 300 KB.
#
# Only applies to mobilenet_v2. Leave as None unless flash gets tight -- see
# MAX_MODEL_BYTES below for why 618 KB is fine on a 4 MB board.
TRUNCATE_AT = None

# --------------------------------------------------------------------------
# Training
#
# Two phases: first train only the new classifier head with the backbone frozen,
# then unfreeze the top of the backbone and fine-tune everything at a much lower
# learning rate. Fine-tuning at the normal rate would destroy the pretrained
# features before the head is any good, which is why the head is trained first.
# --------------------------------------------------------------------------

BATCH_SIZE = 32
SEED = 1337

HEAD_EPOCHS = 25
HEAD_LR = 1e-3

FINETUNE_EPOCHS = 40
FINETUNE_LR = 1e-4
FINETUNE_UNFREEZE_LAYERS = 40   # how many layers from the end to unfreeze

EARLY_STOPPING_PATIENCE = 8

# Split ratios used by prepare_data.py when the raw dataset has no split of its
# own. The Kaggle set ships with train/validation/test already, and those are
# reused when present.
SPLIT_RATIOS = (0.70, 0.15, 0.15)

# --------------------------------------------------------------------------
# Augmentation
#
# Geometry stays mild: the camera is fixed above the tray, so the model does not
# need to survive large rotations.
#
# Photometrics are not mild, and that is the point. With the off-target produce
# removed, the remaining gap between the training data and the device is not
# "which vegetable" -- it is that the training images are sharp, well-exposed web
# photographs and the device returns soft, warm, washed-out OV2640 frames. Look
# at my images/from ESPCAM/ next to any Kaggle photo; that gap is now the main
# source of error, and augmentation is the only lever available for it without
# capturing real frames per crop.
# --------------------------------------------------------------------------

AUG_ROTATION = 0.08         # fraction of 2*pi
AUG_ZOOM = 0.15
AUG_TRANSLATION = 0.10
AUG_BRIGHTNESS = 0.20
AUG_CONTRAST = 0.20
AUG_HORIZONTAL_FLIP = True

# Defocus. The OV2640 behind a cheap fixed lens is never quite sharp, and at
# 96x96 a soft edge and a sharp one are genuinely different inputs. sigma is
# sampled up to this value.
#
# This also removes a shortcut the model would otherwise take. The unknown class
# is generated from a real photograph and carries real blur and JPEG artefacts,
# while the crop classes are clean web photos. Blurring every class during
# training stops sharpness itself from being the feature that separates
# "crop" from "empty".
AUG_BLUR_SIGMA = 0.5

# White balance, as a per-channel gain: red up and blue down for a warm cast,
# the reverse for a cool one. The value is the maximum shift in either
# direction, so 0.12 spans roughly the range between the warm ESPCAM frames in
# my images/ and a cooler LED lamp.
#
# This is deliberately NOT RandomHue. A hue rotation moves red towards green and
# destroys the colour identity the thin classes depend on, which is what the
# measurement below found. A white-balance gain shifts the white point while
# leaving the ordering of object colours intact -- a warm-cast tomato is still
# the reddest thing in the frame.
AUG_WHITE_BALANCE = 0.06

# Hue and saturation jitter: OFF, and it must stay off. Measured on the test set:
#
#     hue=0    sat=0      88.2%   <- current
#     hue=0.03 sat=0.10   73.1%   (onion 0.30, unknown 0.43)
#     hue=0.08 sat=0.25   79.6%
#
# The intuition that it should help -- apples are red, green and yellow, and
# the training images are overwhelmingly red -- is right for apple and wrong
# for the class set as a whole. Onion, potato and ginger are separated mainly
# BY their brown/tan colour, so perturbing hue destroys the main signal. It
# does lift apple (0.70 -> 0.90) while costing far more elsewhere.
#
# Those numbers were measured when `unknown` was 27 other vegetables, so the
# unknown figure no longer transfers. The reason the crop classes suffered does:
# onion, potato and ginger are still in the set and still separated by colour.
# AUG_WHITE_BALANCE above is the intended way to handle a colour cast.
AUG_HUE = 0.0
AUG_SATURATION = 0.0

# Class weighting: OFF. Measured 84.9% with it against 88.2% without, on a
# 1.76x imbalance that is evidently mild enough not to need correcting.
#
# The imbalance is smaller now, not larger: the crop classes are the Kaggle
# folders alone at 68-94 training images each, and the unknown class is sized to
# match by scripts/make_background.py --count. So the case for leaving this off
# is stronger than when it was measured.
USE_CLASS_WEIGHTS = False

# Label smoothing: OFF, tested together with class weighting above.
LABEL_SMOOTHING = 0.0

# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

TOP_K = 5

# Below this, the Main ESP32 should keep its previous fan state rather than act
# on the detection. Mirrors the reliability requirement in the firmware spec.
CONFIDENCE_THRESHOLD = 0.60

# --------------------------------------------------------------------------
# Deployment budget -- export_tflite.py warns if the model exceeds this.
#
# The model is linked into the app binary as a C array, so it competes with the
# firmware for the app partition, not for the whole 4 MB of flash. Budget:
#
#     camera driver + esp-tflite-micro + ESP-NOW/WiFi    ~700 KB - 1 MB
#     model (MobileNetV1 a=0.5)                           ~968 KB
#                                                        ------------
#                                                        ~1.7 - 2.0 MB
#
# That does not fit ESP-IDF's default single-app partition table (1 MB app), so
# the firmware needs a custom partition table with roughly a 2.5 MB app
# partition. On a 4 MB board there is room for that with the remainder left for
# NVS and PHY init data. The limit below is set where a model would actually
# start causing trouble rather than at a round number.
# --------------------------------------------------------------------------

MAX_MODEL_BYTES = 1200 * 1024

# Cap on how many extra images a single class may absorb from EXTRA_TRAIN_DIRS.
#
# 150 lets each crop take all ~110 of its composites while holding `unknown` to
# 150 of its 324. Unknown already carries the produce negatives and the
# empty-box crops, so without the cap it would outweigh every crop class.
#
# Resulting train split: ~178-204 per crop, 272 unknown, 1.53x imbalance.
EXTRA_TRAIN_MAX_PER_CROP = 150
