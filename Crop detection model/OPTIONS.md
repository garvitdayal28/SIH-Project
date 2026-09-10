# Crop Detection Model — Design Options

Scope of this document: **only** the crop-classification model for the ESP32-CAM.
No camera driver, no ESP-NOW, no fan control, no BLE. Those come later.

Target for this stage:

```text
image file on disk (JPG/PNG)
        ↓
  preprocessing
        ↓
  quantized model
        ↓
  top-5 crops + confidences printed to stdout
```

No GUI. A CLI that takes a file path.

---

## 0. Hard constraints from the hardware

The AI-Thinker ESP32-CAM is the baseline assumption.

| Resource | Available | Practical budget for the model |
|---|---|---|
| CPU | ESP32, dual Xtensa LX6 @ 240 MHz, **no SIMD / vector unit** | 1–3 s per inference is realistic and acceptable |
| SRAM | 520 KB (fragmented, WiFi stack takes a chunk) | tensor arena must live in PSRAM |
| PSRAM | 4 MB | ~200 KB tensor arena, ~600 KB camera frame buffer |
| Flash | 4 MB | model is linked into the app binary; ~2 MB app partition needed |

Consequences that drive every choice below:

- The model **must be int8 quantized**. Float32 is too slow and too large.
- Input resolution realistically caps at **96×96**. 160×160 is possible but pushes
  inference past ~5 s on the plain ESP32.
- Classes should be kept to roughly **8–15 crops**, not 100+. More classes means a
  wider final layer, more training data needed, and worse per-class accuracy.

### Hardware: settled

The board is a **plain AI-Thinker ESP32-CAM**. Everything in this document
assumes it, which is the worse case — the same `.tflite` would also run on an
ESP32-S3 roughly 5–10× faster if one is ever added, with no changes here.

The one consequence to carry into firmware: the model is linked into the app
binary, so the app partition needs to be about 2 MB. ESP-IDF's default 1 MB
single-app partition table is not enough.

---

## 1. Framework / toolchain options

### Option A — TensorFlow / Keras → TFLite int8 → TFLite Micro  *(recommended)*

Train in Keras on your machine, export a `.tflite` with full int8 quantization,
convert to a C array, run on device with the `esp-tflite-micro` ESP-IDF component.

- Pro: the standard, best-documented path for MCU inference; Espressif maintains
  the ESP-IDF component; full control over training; works completely offline.
- Pro: the same `.tflite` file runs in Python via `tflite-runtime`, so the desktop
  test harness executes **the exact bytes the ESP32 will execute**. What you see
  in the terminal is what the board will produce.
- Con: TensorFlow tooling on Windows is heavy (~600 MB install).
- Con: requires Python 3.11/3.12 (see §5).

### Option B — Edge Impulse

Upload the dataset to their web studio, click through training, download a
generated Arduino/ESP-IDF library.

- Pro: fastest route to something working; handles quantization and the C++
  wrapper for you; their EON compiler beats plain TFLM on RAM.
- Con: training happens **on their cloud**. Your brief is "no internet, no cloud" —
  that applies to runtime, not training, so this is not disqualifying, but it is a
  dependency and a likely question at evaluation.
- Con: much less control; generated code is a black box; harder to build the
  "give it a file path" desktop harness you asked for.
- Con: free tier has job-time limits.

### Option C — PyTorch → ONNX → TFLite

- Pro: nicer training ergonomics if you already know PyTorch.
- Con: the ONNX→TFLite chain is fragile, especially for quantized models. You will
  spend more time fighting converters than training. Not worth it here.

### Option D — Espressif ESP-DL

Espressif's own inference library plus their quantization toolkit.

- Pro: fastest possible inference on ESP32-S3.
- Con: rough tooling, thin documentation, heavily S3-oriented. Poor fit for a
  plain ESP32-CAM and for a time-boxed SIH build.

**Recommendation: A.** It is the only option that gives you a bit-identical
desktop test harness, which is exactly what you need while waiting on hardware.

---

## 2. Model architecture options

All assume int8 quantization and a softmax output over N crop classes.

### Option 1 — MobileNetV2, alpha=0.35, 96×96 RGB, transfer learning  *(recommended initially; rejected on measurement)*

Load ImageNet-pretrained weights, freeze the backbone, train the classifier head,
then fine-tune the last few blocks.

- Size: **618 KB int8, measured** (at 13 classes; ~612 KB at the 8 now in use).
- Arena: ~150–200 KB, needs PSRAM.
- Speed: ~1.5–3 s on plain ESP32, ~0.3 s on S3.
- Accuracy: **best of these options**, because ImageNet pretraining already knows
  what fruit and vegetable texture looks like. Works with only a few hundred
  images per class.

### Option 2 — MobileNetV1, alpha=0.25, 96×96 RGB, transfer learning

The classic TFLM reference scale.

- Size: **300 KB int8, measured**.
- Speed: ~1 s on plain ESP32.
- Accuracy: a few points below Option 1.
- Pick this if flash or latency turns out tighter than expected. It is a drop-in
  swap — same training script, two config lines.

### The decision that actually mattered: int8, not float

MobileNetV2 was the wrong choice, and only measurement showed it. Its
depthwise convolutions and linear bottlenecks produce activation ranges that
per-tensor int8 quantization handles badly, so post-training quantization threw
away most of its advantage. Measured on the 9-class dataset, over all 949
images:

| Backbone | float | **int8** | drop | predictions changed | size |
|---|---:|---:|---:|---:|---:|
| MobileNetV1 α=0.5 | 92.1% | **91.5%** | −0.6 | 4.5% | 968 KB |
| MobileNetV1 α=0.25 | 85.5% | 83.6% | −1.9 | 4.2% | 305 KB |
| MobileNetV2 α=0.35 | 94.8% | 82.4% | −12.4 | 15.3% | 624 KB |

MobileNetV2 is the better model in float and the worse model on the device.
More calibration data does not rescue it — 300, 633 and 672 representative
images all gave the same ~84%, so the loss is structural rather than a
calibration problem.

**MobileNetV1 α=0.5 is what shipped.** It costs size (968 KB) and some
inference time, both affordable on a 4 MB board; 9 points of on-device accuracy
was not.

Quantization-aware training would likely let MobileNetV2 keep its float
accuracy and is the one route to beating this. It needs
`tensorflow-model-optimization` plus legacy-Keras mode, and a custom
`QuantizeConfig` for the `Rescaling` layer, which `quantize_model` rejects
outright. Worth revisiting only if 91.5% proves insufficient.

### Sizes, measured rather than estimated

The estimate of "~400–500 KB" for Option 1 in the first draft of this document
was wrong, and worth recording why. MobileNetV2 ends with a 1×1 convolution to
1280 channels, and Keras does **not** scale that layer with `alpha` — it is 1280
wide at alpha=0.35 just as it is at alpha=1.0. At this scale that one layer is
most of the model.

Converting each candidate to int8 gives:

| Configuration | Params | int8 size |
|---|---:|---:|
| MobileNetV2 α=0.35, full | 426,861 | **618 KB** |
| MobileNetV2 α=0.35, cut at `block_16_project_BN` | 263,197 | 430 KB |
| MobileNetV2 α=0.35, cut at `block_13_expand_relu` | 101,021 | 206 KB |
| MobileNetV1 α=0.25 | 221,885 | 300 KB |
| MobileNetV1 α=0.5 | 836,205 | 964 KB |

618 KB is deployable — the model is linked into the app binary, so with a
firmware of roughly 700 KB – 1 MB the app partition needs to be about 2 MB,
which a custom partition table on a 4 MB board provides with room to spare. It
does **not** fit ESP-IDF's default 1 MB single-app table, so that is a firmware
task to remember rather than a model problem.

The cut points are available as `TRUNCATE_AT` in `config.py` if flash does get
tight. Default is no truncation.

### Option 3 — Small custom CNN from scratch, 64×64

Four conv blocks, trained from random init.

- Size: ~80–150 KB. Fastest.
- Con: no pretrained features, so it needs **far more data** and still generalizes
  worse, especially to backgrounds and lighting it has not seen.
- Only worth it if Options 1 and 2 somehow do not fit.

### Option 4 — MCUNet / TinyNAS

- Pro: best published accuracy-per-byte.
- Con: unfamiliar toolchain, sparse docs, high risk for a deadline project.

**Recommendation: 1, with 2 kept as a config flag.** The training script should
make backbone and alpha a single parameter, so switching is a one-line change
rather than a rewrite.

---

## 3. Dataset options

The crops in use: **apple, banana, corn, ginger, lemon, onion, potato, tomato**.

The dataset's other 27 classes become `unknown` negatives rather than being
discarded. `sweetcorn` is excluded from both roles — it is the same vegetable as
`corn`, so using it as a negative would train the model to reject real corn.

### Option I — Kaggle "Fruit and Vegetable Image Recognition"  *(recommended)*

36 classes, ~3,800 images. Real photographs — varied backgrounds, lighting,
angles, sometimes hands or multiple items in frame.

- Pro: realistic. A model trained on this has a chance of working on an actual
  ESP32-CAM frame.
- Pro: already contains most of the crops you need; you select a subset.
- Con: only ~100 images per class. Manageable with transfer learning plus
  augmentation, but it is the main accuracy ceiling.

### Option II — Fruits-360

90,000+ images, 130+ classes, a single rotating fruit on a **pure white background**.

- Pro: huge, clean, perfectly labelled, trivially easy to hit 99% validation
  accuracy.
- Con: that 99% is misleading. The white background means the model never learns
  to ignore a background, and it collapses on real camera images. This is the most
  common reason student TinyML crop projects work in the notebook and fail on the
  bench.
- Use only if you commit to shooting crops against a plain white sheet in the
  actual demo — which, for a controlled cold-storage rig, is not unreasonable.

### Option III — Kaggle 36-class + your own ESP32-CAM captures

Train on Option I, then fine-tune on 30–50 images per crop captured through the
actual ESP32-CAM at the actual mounting distance and lighting.

- Pro: **by far the highest real-world accuracy.** Closes the domain gap between
  clean web photos and a grainy OV2640 frame.
- Con: needs the hardware, so it is a later step, not a now step.

### Option IV — Combination: I as the base, II folded in for volume

Merge the two, mapping Fruits-360 classes onto your label set.

- Pro: more data.
- Con: the white-background images dominate by count and drag the model toward the
  same failure mode as Option II unless carefully reweighted. Fiddly.

**Recommendation: I now, III later.** Build and validate the whole pipeline on the
Kaggle 36-class set today. Once the camera arrives, fine-tune on real captures —
the training script should be written so that is a config change, not a rewrite.

---

## 4. What "top-5" means here

The model produces a full softmax vector over all N classes. Top-5 is just a sort
of that vector. Two separate things:

- **Desktop harness (this stage):** prints the top 5 with confidences, so you can
  see what the model confuses with what. This is your debugging surface.
- **On-device (later):** the ESP-NOW packet in the V1 spec carries only `crop_id`
  and `confidence`, i.e. top-1. The firmware will still compute the full vector and
  can log top-5 over serial, but the packet stays small as specified.

So top-5 is a diagnostic output and top-1 is the transmitted result. Nothing about
the model itself changes.

An **"unknown / background" class** is strongly recommended. Without one, a model
shown an empty tray will confidently report a crop, and the Main ESP32 will change
fan speed based on nothing. With one, the confidence threshold in §15 of the
firmware spec has something real to act on.

---

## 5. Environment prerequisite

Your machine currently has **only Python 3.14**, and TensorFlow publishes no wheels
for 3.14 — verified here: `pip install tensorflow` fails with "No matching
distribution found for tensorflow (from versions: none)".

So before any training work: install **Python 3.11** (most stable for the TF and
TFLite tooling) alongside 3.14. It does not replace 3.14; the `py` launcher keeps
both, and this project will use a `.venv` pinned to 3.11.

This applies to Option A. Option B (Edge Impulse) trains in the browser and would
sidestep it, but you would still want a local venv for the test harness.

---

## 6. Proposed folder layout, once choices are made

```text
Crop detection model/
├── OPTIONS.md              this file
├── README.md               how to run it
├── requirements.txt
├── config.py               classes, input size, backbone, alpha, paths
├── data/
│   ├── raw/                downloaded dataset
│   └── prepared/           filtered to your classes, train/val/test split
├── scripts/
│   ├── prepare_data.py     filter + split + class-balance report
│   ├── train.py            transfer learning + fine-tune
│   ├── export_tflite.py    int8 quantization + size/accuracy report
│   └── export_c_array.py   .tflite → model_data.cc for ESP-IDF
├── predict.py              ← the CLI you asked for: predict.py <image path>
├── models/
│   ├── model.keras
│   └── model_int8.tflite
└── outputs/                training curves, confusion matrix
```

`predict.py` runs the **`.tflite`** file, not the Keras model, so its output
matches what the ESP32 will produce.

---

## 7. Decisions needed from you

1. Framework — A / B / C / D
2. Architecture — 1 / 2 / 3 / 4
3. Dataset — I / II / III / IV
4. Class list — confirm or edit the crops in §3
5. Include an "unknown" class? — recommended yes
6. Camera board — plain ESP32-CAM, or upgrade to an ESP32-S3 board?

Chosen: **A + I**, MobileNetV1 α=0.5 (not the initially recommended V2 — see §2), 8 crops plus unknown, on a plain
ESP32-CAM with the S3 kept open as an option.
