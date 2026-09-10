# FarmFrost — Crop Detection Model

TinyML crop classifier for the ESP32-CAM. Trains on a development machine,
exports an int8 TFLite model small enough to run on-device, and gives you a
command-line test harness so you can validate the whole thing before the camera
hardware arrives.

**Scope:** the model only. No camera driver, no ESP-NOW, no fan control, no BLE.
See [FarmFrost_Firmware_V1_Overview.md](../FarmFrost_Firmware_V1_Overview.md)
for the wider system and [OPTIONS.md](OPTIONS.md) for why each choice was made.

```text
image file  ->  centre-crop + resize 96x96  ->  int8 MobileNetV1  ->  top-5 crops
```

## Configuration in force

| | |
|---|---|
| Framework | TensorFlow / Keras → TFLite int8 → TFLite Micro |
| Architecture | MobileNetV1, alpha=0.5, 96×96 RGB, ImageNet transfer learning |
| Dataset | Kaggle "Fruit and Vegetable Image Recognition" (36 classes) |
| Classes | 8 crops + `unknown` |
| Target | AI-Thinker ESP32-CAM (works unchanged, and much faster, on ESP32-S3) |

Crops: **apple, banana, corn, ginger, lemon, onion, potato, tomato**.

The dataset's other 27 classes are not thrown away — they become the `unknown`
class, which is what stops the model reporting a crop when it is shown an empty
tray or a vegetable it does not handle. `sweetcorn` is the one exception: it is
the same vegetable as `corn`, so using it as a negative would teach the model to
reject real corn. It is excluded outright (`EXCLUDED_CLASSES` in config.py).

Measured for the int8 model — the same bytes that run on the board:

| | Kaggle test (88) | held-out Open Images (420) |
|---|---:|---:|
| Kaggle data only | 88.2% | 32.8% |
| **+ extra data, raw produce only** | **89.8%** | **65.0%** |

Top-5 is 96.6% / 96.0%.

All 8 crops are topped up (see `EXTRA_TRAIN_DIRS` in config.py) with real
photographs in varied framings — whole scenes with several items, objects with
context, and objects filling the frame. The extra data is **raw whole produce
only**: images of chips, sauces, cocktails and cooked dishes were filtered out,
because the camera sees loose crops in a tray and training on prepared food
widens each class until "potato" starts to mean "anything potato-ish".

Ginger is the weak class — 96 training images against ~180 for the others.
Neither Open Images nor Wikimedia Commons has much whole raw ginger, and it is
the crop where your own captures would matter most.

The exported model measures **968 KB int8**. It is linked into the app binary,
so the firmware needs a custom partition table with roughly a 2.5 MB app
partition — ESP-IDF's default 1 MB single-app table is not enough. Smaller
configurations are a config change away; see Tuning below.

---

## Setup

TensorFlow publishes no wheels for Python 3.14, so this project uses a venv
pinned to **Python 3.11** (already installed alongside your 3.14).

```powershell
cd "Crop detection model"
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Everything below assumes `.venv\Scripts\python.exe`. Activating the venv
(`.venv\Scripts\Activate.ps1`) lets you just type `python` instead.

---

## Get the dataset

Kaggle: **kritikseth/fruit-and-vegetable-image-recognition**

<https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition>

Either download the zip in a browser and unzip it into `data/raw/`, or use the
Kaggle CLI if you have an API token in `~/.kaggle/kaggle.json`:

```powershell
.venv\Scripts\python.exe -m pip install kaggle
.venv\Scripts\kaggle.exe datasets download -d kritikseth/fruit-and-vegetable-image-recognition -p data\raw --unzip
```

You should end up with `data/raw/train/`, `data/raw/validation/` and
`data/raw/test/`, each holding 36 class folders. An extra level of nesting is
fine — `prepare_data.py` looks a couple of directories deep.

### Extra images from Open Images

`data/raw/extra/<crop>/` is folded into the **training split only**, capped by
`EXTRA_TRAIN_MAX_PER_CROP`. Val and test stay on the Kaggle distribution so
accuracy numbers remain comparable across experiments.

`data/raw/extra_holdout/<crop>/` is 519 Open Images photos never trained on,
split by **source photograph** — the tight and wide crops of one photo are
near-duplicates, so splitting them across train and eval would leak and flatter
the number. It is the more realistic of the two evaluations: larger than the
Kaggle test set and full of cluttered, multi-object scenes.

This is also where your own ESP32-CAM captures should go once you have the
hardware, and for ginger it is the only way to close the gap.

### Optional but worth it: background images

Anything you drop in `data/raw/background/` is folded into the `unknown` class:
empty trays, crates, the inside of the storage unit, a hand in frame. These are
the most valuable negatives you can supply, because they are the actual scene
the camera will look at when there is no crop present.

---

## Run the pipeline

```powershell
.venv\Scripts\python.exe scripts\prepare_data.py     # filter, fold unknown, split
.venv\Scripts\python.exe scripts\train.py            # two-phase transfer learning
.venv\Scripts\python.exe scripts\export_tflite.py    # int8 quantization
.venv\Scripts\python.exe scripts\evaluate.py         # accuracy of the int8 model
.venv\Scripts\python.exe scripts\export_c_array.py   # C source for the firmware
```

`prepare_data.py` prints a class-balance table and warns about thin classes.
`train.py` writes curves to `outputs/`. `export_tflite.py` prints the model size
against the flash budget and the exact input/output quantization parameters the
firmware will need. `evaluate.py` is the one that produces the numbers worth
quoting — it measures the quantized model, not the float one.

---

## Test it on an image

This is the part that stands in for the ESP32-CAM:

```powershell
.venv\Scripts\python.exe predict.py C:\path\to\tomato.jpg
```

```text
tomato.jpg
------------------------------------------------------------
 > 1. tomato     92.14%  ##########################..
   2. apple       4.02%  #...........................
   3. onion       1.88%  ............................
   4. lemon       0.91%  ............................
   5. unknown     0.42%  ............................
------------------------------------------------------------
   Detected : TOMATO
   Confidence: 92.1%
   ACCEPTED  : crop_id 7 would be sent.
```

Other forms:

```powershell
predict.py C:\path\to\folder          # every image in a folder
predict.py image.jpg --json           # machine-readable
predict.py image.jpg --esp-now        # show the 2-byte packet that would be sent
predict.py image.jpg --top 3          # fewer matches
```

`predict.py` runs `models/model_int8.tflite` through the same preprocessing the
firmware will use, so what you see here is what the board will produce.

---

## Why top-5 here but top-1 on the wire

The model always produces a full probability vector over all 9 classes. Top-5
is just a sort of it, and it exists as a **debugging surface** — it tells you
what the model confuses with what, which is the information you need to decide
where to add training images.

The ESP-NOW packet in the V1 spec carries only `crop_id` and `confidence`, i.e.
top-1, in two bytes. Nothing about the model changes between the two; `--esp-now`
shows you exactly what would be transmitted.

## Why an `unknown` class

Without it, a model shown an empty tray still outputs a probability distribution
over the 8 crops, and the largest one wins. The Main ESP32 would then change fan
speed based on nothing. With `unknown` in the label set, and the confidence
threshold in `config.py`, both failure modes have somewhere to land.

`evaluate.py` reports the number that actually matters here: how often the model
is **wrong and above the threshold** — the cases where the fan would be set for
the wrong crop.

---

## Tuning

Everything lives in [config.py](config.py). The ones you are most likely to touch:

| Setting | Effect |
|---|---|
| `TARGET_CROPS` | which crops the model knows; the key is our name, the value is the dataset's folder name |
| `BACKBONE`, `ALPHA` | `0.25` gives a 305 KB model at ~84% int8; MobileNetV2 quantizes badly, see config.py |
| `IMAGE_SIZE` | 96 is the practical ceiling for a plain ESP32; ImageNet weights exist for it |
| `CONFIDENCE_THRESHOLD` | trades responsiveness against acting on a misread |
| `UNKNOWN_SIZE_MULTIPLIER` | how much of the training set is negatives |
| `FINETUNE_UNFREEZE_LAYERS` | more layers = more capacity to adapt, more overfitting risk on ~100 images/class |

Changing `TARGET_CROPS` changes the class order, and **class order is `crop_id`**
on the ESP-NOW wire. Re-run `export_c_array.py` and update the Main ESP32's
fan-speed table together, or tomato quietly becomes onion.

---

## Layout

```text
config.py                 all settings
predict.py                the CLI test harness
cropnet/
  preprocess.py           centre-crop + resize; the definition the firmware mirrors
  labels.py               label order (= crop_id) handling
  tflite_utils.py         interpreter loading and top-k
scripts/
  prepare_data.py         filter to the 8 crops, build unknown, split
  train.py                two-phase transfer learning
  export_tflite.py        int8 quantization + deployment report
  evaluate.py             accuracy of the quantized model, confusion matrix
  export_c_array.py       .tflite -> model_data.cc/.h for ESP-IDF
data/raw/                 dataset as downloaded
data/prepared/            train|val|test / class / images
models/                   model.keras, model_int8.tflite, labels.txt, model_data.*
outputs/                  curves, metrics, confusion matrix
```

---

## What happens after this

Once accuracy looks acceptable, `export_c_array.py` produces `model_data.cc`
and `model_data.h` for an ESP-IDF component using `esp-tflite-micro`. The header
carries the input geometry, the `crop_id_t` enum, and the confidence threshold,
so the firmware side does not restate any of it.

Two things to expect when the camera arrives:

1. **Accuracy will drop.** Web photos and OV2640 frames are different
   distributions. The fix is 30–50 captures per crop through the real camera at
   the real mounting distance, fine-tuned on top of this model — the pipeline is
   already structured so that is a config change, not a rewrite.

2. **Set the tensor arena in PSRAM**, not internal SRAM. `export_tflite.py`
   prints a starting figure; call `arena_used_bytes()` after
   `AllocateTensors()` and shrink to the real number plus headroom.
