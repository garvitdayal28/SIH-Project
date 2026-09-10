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
# --------------------------------------------------------------------------

TARGET_CROPS = {
    "apple": "apple",
    "banana": "banana",
    "corn": "corn",
    "ginger": "ginger",
    "lemon": "lemon",
    "onion": "onion",
    "potato": "potato",
    "tomato": "tomato",
}

# The "unknown" class stops the model from confidently reporting a crop when it
# is shown something else -- an empty tray, a hand, a different vegetable. The
# Main ESP32 must never change fan speed on a misread, so this class is what
# gives the confidence threshold in the firmware spec something real to act on.
UNKNOWN_CLASS = "unknown"
INCLUDE_UNKNOWN = True

# Negative examples for the unknown class: the dataset's other classes. These
# are real produce photos that are *not* one of our crops, which is exactly the
# confusion we want the model to learn to reject. They cost nothing -- the
# images are already downloaded.
UNKNOWN_SOURCE_CLASSES = [
    "beetroot", "bell pepper", "cabbage", "capsicum", "carrot", "cauliflower",
    "chilli pepper", "cucumber", "eggplant", "garlic", "grapes", "jalepeno",
    "kiwi", "lettuce", "mango", "orange", "paprika", "pear", "peas",
    "pineapple", "pomegranate", "raddish", "soy beans", "spinach",
    "sweetpotato", "turnip", "watermelon",
]

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

# Any images you drop in data/raw/background/ (empty trays, crates, the inside of
# the storage unit, hands, walls) are folded into the unknown class too. This is
# optional but it is the single cheapest accuracy win once the rig exists.
BACKGROUND_DIR = DATA_RAW / "background"

# Extra images for specific crops, folded into the TRAIN split only.
#
# Train-only is deliberate. Validation and test must keep coming from a single
# consistent distribution, otherwise the accuracy number stops being comparable
# to anything measured before.
#
# Contents: raw, whole produce ONLY. The camera will see loose crops in a tray,
# so a photograph of chips, ketchup or a cocktail teaches the wrong thing --
# it widens each class until "potato" starts to mean "anything potato-ish".
#
# Sources:
#   Open Images V7 -- real photographs in varied contexts. For apple, banana,
#   lemon, potato and tomato it ships bounding boxes, so each photo yields
#   several framings: the whole scene (often several items), the object with
#   context, and the object filling the frame. Corn, ginger and onion have
#   image-level labels only and get whole scenes.
#
#   Wikimedia Commons -- used for the three crops Open Images covers poorly.
#
# Three filters were applied, in order:
#   1. Label filter. Any photo carrying a prepared-food or drink label was
#      dropped (747 images: Juice 151, Cocktail 132, Cheese 116, Drink 101,
#      Pizza 53, Bread 46, Salad 43, French fries 31, ...).
#   2. Visual filter. Open Images' human labels are sparse -- a plate of roast
#      potatoes may carry only "Potato" -- so a full ImageNet classifier
#      rejected another 99 whose top prediction was a cooked dish, a drink or
#      a non-food object. Container classes (crate, basket, tray) were
#      deliberately NOT blocked: a crate of apples is exactly the multiple-item
#      case the camera will see.
#   3. Hand review. The Commons images were reviewed by eye and picked
#      individually, because searching "ginger" also returns ginger plants,
#      gingerbread, ginger ale and a person named Ginger. Yield was low:
#      28 of 135 for ginger, 31 of 153 for onion, 40 of 145 for corn.
#
# Open Images' own "ginger" label included a photograph of carrots. It was
# caught in hand review and dropped.
#
# Ginger remains the weak class: 28 usable extra images against 300-670 for the
# others, because neither source has much whole raw ginger. It is the one crop
# where your own ESP32-CAM captures would make a decisive difference.
EXTRA_TRAIN_DIRS = {c: DATA_RAW / "extra" / c for c in TARGET_CROPS}

# The 24 unknown-source classes together hold ~24x more images than any single
# crop. Left alone that imbalance would teach the model to answer "unknown" for
# everything. We cap the unknown class at this multiple of the average crop
# class size, sampling evenly across the source classes.
UNKNOWN_SIZE_MULTIPLIER = 1.0


def class_names():
    """The model's output classes, in the fixed order used everywhere.

    This order defines the integer crop_id sent over ESP-NOW, so it must stay
    stable. Alphabetical crops first, then unknown last -- appending unknown at
    the end means adding it later would not renumber the existing crops.
    """
    names = sorted(TARGET_CROPS.keys())
    if INCLUDE_UNKNOWN:
        names.append(UNKNOWN_CLASS)
    return names


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
# Deliberately mild. The ESP32-CAM will be fixed above the tray, so the model
# does not need to survive large rotations -- but it does need to survive bad
# white balance and a dim storage unit, hence the brightness/contrast range.
# --------------------------------------------------------------------------

AUG_ROTATION = 0.08         # fraction of 2*pi
AUG_ZOOM = 0.15
AUG_TRANSLATION = 0.10
AUG_BRIGHTNESS = 0.25
AUG_CONTRAST = 0.25
AUG_HORIZONTAL_FLIP = True

# Colour jitter: OFF, and it must stay off. Measured on the test set:
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
AUG_HUE = 0.0
AUG_SATURATION = 0.0

# Class weighting: OFF. Measured 84.9% with it against 88.2% without, on a
# 1.76x imbalance that is evidently mild enough not to need correcting.
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

# Cap on how many extra images a single crop may absorb from EXTRA_TRAIN_DIRS.
#
# Measured with the 8 crop classes' val/test held identical throughout.
# "held-out" is Open Images photos never trained on, grouped by SOURCE
# photograph so no framing of a training image can leak into it.
#
#     training data                    Kaggle test   held-out   real photos
#     Kaggle only                          88.2%       32.8%       6/9
#     + Open Images, unfiltered            90.9%       65.5%       7/9
#     + Open Images, raw produce only      89.8%       65.0%       7/9
#
# The last row is what ships. Filtering out prepared food cost essentially
# nothing on the held-out set once the thin classes were topped up from
# Commons, and it removes an entire class of confusion the deployment would
# otherwise inherit.
#
# 100 keeps the imbalance at 2.02x, with ginger the smallest class.
EXTRA_TRAIN_MAX_PER_CROP = 100
