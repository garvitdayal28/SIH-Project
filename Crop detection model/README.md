# FarmFrost — Crop Detection Model

TinyML classifier for the ESP32-CAM. Trains on a development machine, exports an
int8 TFLite model, and gives you a command-line harness to test it on real
photographs.

**Scope:** the model only. No camera driver, no ESP-NOW, no fan control, no BLE.
See [FarmFrost_Firmware_V1_Overview.md](../FarmFrost_Firmware_V1_Overview.md)
for the wider system.

```text
image  ->  centre-crop + resize 96x96  ->  int8 MobileNetV1  ->  crop + confidence
```

---

## What this model is for

One fixed camera looking into a white thermocol box holding exactly one item.
The item is always one of four, or the box is empty. Nothing else is ever placed
in it, so the model is not asked to recognise anything else.

| | |
|---|---|
| Classes | `banana`, `empty`, `lemon`, `onion`, `tomato` |
| crop_id | 0=banana, 1=empty, 2=lemon, 3=onion, 4=tomato |
| Architecture | MobileNetV1, alpha=0.5, 96x96 RGB, ImageNet transfer learning |
| Model size | 966 KB int8 |
| Training data | the rig photographs in `my images/`, expanded by recompositing |

Measured on the int8 model — the same bytes that run on the board:

| | |
|---:|---|
| **72/72** | every real rig photograph supplied |
| **53/53** | held-out split, never trained on |
| 99.6% | median confidence; worst case 64.5% |

An empty box is recognised at 99.6%.

Apple was deliberately dropped from an earlier 8-crop version: it is red and
round like a tomato, and removing it makes the remaining job easier rather than
harder.

### Two things to know

**Pomegranate is not a class.** The three ESP32-CAM frames in `my images/` are of
a pomegranate, and the model calls it `lemon` at 97%. That is correct behaviour
for a model that was never trained to reject anything, because nothing else goes
in the box. If that assumption changes, an `unknown` class needs real photographs
of whatever else might appear — produce negatives scraped from the web will not
do it.

**Banana has no real photographs.** Its training, validation and test images are
all synthesised from Kaggle bananas composited into the box, so its 12/12 is
optimistic in a way the other three classes' scores are not. Photograph a banana
in the box and re-run the pipeline to fix that.

---

## Setup

TensorFlow publishes no wheels for Python 3.14, so this project uses a venv
pinned to **Python 3.11**.

```powershell
cd "Crop detection model"
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Activating the venv (`.venv\Scripts\Activate.ps1`) lets you type `python`
instead of the full path.

---

## Rebuilding the model

```powershell
python scripts\build_rig_dataset.py    # dataset from my images/
python scripts\train.py                # two-phase transfer learning
python scripts\export_tflite.py        # int8 quantization
python scripts\export_c_array.py       # model_data.cc / .h for the firmware
```

`build_rig_dataset.py` replaces the old Kaggle pipeline. Kaggle is still on disk
and is used for exactly one thing: banana.

### Adding photographs

Drop them in `my images/from phone camera/<class>/` and re-run the four commands
above. More real photographs is the single most useful change you can make —
every accuracy gain in this project came from that, not from architecture or
hyperparameters.

To add a class, add it to `TARGET_CROPS` in [config.py](config.py) as well.
**Class order is `crop_id` on the ESP-NOW wire**, and it is plain alphabetical
including `empty`, so inserting a class renumbers the ones after it. Re-run
`export_c_array.py` and update the Main ESP32's fan table together.

---

## Testing an image

```powershell
python predict.py "my images\from phone camera\tomato"   # a whole folder
python predict.py C:\path\to\photo.jpg                   # one image
python predict.py photo.jpg --esp-now                    # the 2-byte packet
python predict.py photo.jpg --json                       # machine-readable
```

```text
tomato.jpeg
------------------------------------------------------------
 > 1. tomato   99.61%  ############################
   2. lemon     0.39%  ............................
------------------------------------------------------------
   Detected : TOMATO
   Confidence: 99.6%
   ACCEPTED  : crop_id 4 would be sent.
```

`predict.py` runs `models/model_int8.tflite` through the same preprocessing the
firmware will use, so what you see here is what the board will produce.

---

## How the dataset is built

There are only ~24 real photographs per class, which is not enough on its own.
`build_rig_dataset.py` expands them without leaving the real domain:

1. Each real photo is segmented — the coloured item against near-white thermocol
   separates cleanly on colour distance.
2. The cut-out item is pasted back onto **verified-empty** box backgrounds at
   other positions and scales (10%–88% of frame width, covering both a distant
   item and one filling the view).
3. Every training image passes through a rough ESP32-CAM simulation: lower
   resolution, softer lens, sensor noise, mild exposure and white-balance drift.

Validation and test are **real photographs only** — composites go into train and
nowhere else, so the reported accuracy is measured on genuine frames.

### Two traps this pipeline exists to avoid

**Background must not identify the class.** An earlier version put rig-domain
images in only one class (`empty`, built from crops of the empty box) while the
crop classes held web photos. The model learned "thermocol texture → empty" and
became useless on the hardware: a real lemon in the box came out as `unknown` at
90.2% while the same lemon as a web photo scored `lemon` 99.6%. Every class now
shares the same backgrounds.

**Background plates must be checked, not assumed.** Cropping "the top part of a
rig photo" and assuming the item is lower down put tomatoes and onions into the
`empty` class and pasted lemons on top of plates that already held an onion.
`is_empty_region()` now rejects any plate containing a coloured blob.

---

## Tuning

Everything lives in [config.py](config.py).

| Setting | Effect |
|---|---|
| `TARGET_CROPS` | which items the model knows |
| `CONFIDENCE_THRESHOLD` | below this the firmware keeps its previous fan state |
| `BACKBONE`, `ALPHA` | `0.25` gives a ~305 KB model if flash gets tight |
| `AUG_*` | augmentation strength — see the warning below |

**Do not stack augmentation.** `build_rig_dataset.py` already applies camera
blur, exposure and white-balance drift when it writes the images. Adding strong
versions of the same things again at training time was worth 8 real photographs:
softening `AUG_BLUR_SIGMA` 1.2→0.5, `AUG_WHITE_BALANCE` 0.12→0.06 and
brightness/contrast 0.30→0.20 took the score from 64/72 to 72/72.

---

## Deployment note

The model is 966 KB and is linked into the app binary, so the firmware needs a
custom partition table with roughly a 2.5 MB app partition — ESP-IDF's default
1 MB single-app table is not enough on a 4 MB board.

Input quantization comes out as scale 1.0, zero-point −128, so the entire
device-side preprocessing is:

```c
input[i] = (int8_t)(rgb_byte[i] - 128);
```

Expect 2–4 s per inference on a plain ESP32. Fine for one-shot capture.

---

## Layout

```text
config.py                     all settings
predict.py                    the CLI test harness
my images/                    the real rig photographs (the core dataset)
cropnet/                      preprocessing, labels, TFLite helpers, augmentation
scripts/
  build_rig_dataset.py        dataset from my images/  <- start here
  train.py                    two-phase transfer learning
  export_tflite.py            int8 quantization
  export_c_array.py           .tflite -> model_data.cc/.h
  evaluate.py                 accuracy of the quantized model
  prepare_data.py             the old Kaggle pipeline, superseded
data/prepared/                train|val|test / class / 96x96 PNGs
models/                       model.keras, model_int8.tflite, labels.txt, model_data.*
```
