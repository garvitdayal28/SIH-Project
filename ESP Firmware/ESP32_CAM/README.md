# ESP32-CAM — FarmFrost V1

Capture → crop model → desktop viewer + ESP-NOW.

The MobileNetV1 α=0.5 96×96 int8 model trained in `Crop detection model` runs
**on the board**. No laptop, no server, no internet in the loop.

It classifies exactly five things, because that is all the rig ever contains:

| crop_id | class |
|---:|---|
| 0 | `banana` |
| 1 | `empty` — the box has nothing in it |
| 2 | `lemon` |
| 3 | `onion` |
| 4 | `tomato` |

**The order is the wire format.** The main board's fan table is indexed by it,
so if the class list ever changes, change that table in the same commit.

`empty` is a normal class, not an error code. When it wins, the sketch logs it
and sends nothing — an empty box is a correct answer with no crop to act on, so
the fan holds whatever the last real crop set.

Anything that is *not* one of the four will still be reported as one of them,
confidently. The model was trained on a closed set and has no reject option;
a pomegranate reads as `lemon` at 97%. That is fine while the rig only ever
holds these four, and wrong the moment it does not.

| File | What it is |
|---|---|
| `ESP32_CAM.ino` | Camera, Wi-Fi viewer, ESP-NOW, the detection cycle |
| `crop_model.h/.cpp` | TFLite Micro + the preprocessing that matches `preprocess.py` |
| `model_data.cpp/.h` | The model itself, as a C array. Generated — never hand-edit |

---

## 1. One-time Arduino IDE setup

1. **File → Preferences → Additional boards manager URLs**, add:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
2. **Tools → Board → Boards Manager**, search `esp32`, install
   **esp32 by Espressif Systems**. (Large download, ~1 GB on disk.)
3. Select these under **Tools**:

   | Setting | Value | Why |
   |---|---|---|
   | Board | AI Thinker ESP32-CAM | |
   | PSRAM | **Enabled** | The camera frame buffer needs it, and the arena falls back to it |
   | Partition Scheme | **Huge APP (3MB No OTA/1MB SPIFFS)** | The model alone is ~968 KB |
   | Upload Speed | 921600 (drop to 115200 if it fails) | |

Both bold settings are **required**, not preferences. With the default
partition scheme the link step fails with "text section exceeds available
space"; without PSRAM the arena allocation fails at boot and you get
`[CV] No PSRAM. This model cannot run on this module.`

## 2. Install the TFLite Micro library

**Tools → Manage Libraries**, search `tflm_esp32`, install
**tflm_esp32 by Simone Salerno** (version 2.0.0 is what this was built
against).

This is the one that works. Some notes, because the landscape is confusing:

- **`esp-tflite-micro`** is Espressif's own and is the best of them — but it is
  an **ESP-IDF component only**. `idf.py add-dependency`. It cannot be added to
  the Arduino IDE, and it is *not* bundled with the ESP32 Arduino core, despite
  a widely-copied claim that it is.
- **`Arduino_TensorFlowLite_ESP32`** (tanakamasayuki) is deprecated and predates
  ESP32 core 3.x. `crop_model.cpp` still has a compatibility branch for its
  older `ErrorReporter` API, but it is untested.

`crop_model.cpp` detects which library is present with `__has_include` and
adapts. The unconditional `#include <tflm_esp32.h>` near the top of the `.ino`
is *not* redundant — see the comment there. It is what makes Arduino's library
resolver find the library at all.

### You must patch the library once after installing

`tflm_esp32` 2.0.0 ships its entire **signal** subsystem twice: flattened as
`.cc` files at `src/` and `src/kiss_fft_wrappers/`, and again in its proper tree
as `.cpp` under `src/signal/src/`. Arduino compiles both sets, and the build
dies at link time with pages of:

```
multiple definition of `tflm_signal::RfftInt32Init(long, void*, unsigned int)'
multiple definition of `kiss_fft_fixed16::kiss_fftr_alloc(...)'
```

This is a packaging bug in the library, not in this sketch. The nested
`signal/src/` tree is the complete one and is what every include in the library
actually references, so rename the flattened copies out of the build — 25 files
in two folders:

```bash
cd ~/Documents/Arduino/libraries/tflm_esp32/src
for f in *.cc kiss_fft_wrappers/*.cc; do mv "$f" "$f.disabled"; done
```

PowerShell equivalent:

```powershell
cd ~\Documents\Arduino\libraries\tflm_esp32\src
Get-ChildItem *.cc, kiss_fft_wrappers\*.cc | Rename-Item -NewName { $_.Name + '.disabled' }
```

**This has already been done on this machine.** You only need it again after
reinstalling or updating the library.

Then **clean the build cache** — this part is easy to miss. Arduino keeps
compiled library objects in an archive keyed to the sketch, and renaming a
source file does not evict the `.o` that was already built from it. The link
keeps failing with exactly the same errors and it looks like the rename did not
work. In the IDE: **Sketch → Clean**. Or delete the cached build folder under
`%LOCALAPPDATA%\arduino\sketches\`.

Keep every `.h` — they are still included. The disabled files are FFT and
filter-bank helpers for audio models; nothing in the crop pipeline touches them.
If a future version fixes the packaging, the rename becomes unnecessary rather
than harmful — but check for `.disabled` files before reporting a build error.

## 3. Deploy the model

The model is already deployed — `model_data.cpp` and `model_data.h` are in this
folder and ready to compile. **You only need this section after retraining.**

```bash
cd "Crop detection model"
python scripts/export_tflite.py     # model.keras  -> model_int8.tflite
python scripts/export_c_array.py    # .tflite      -> model_data.cc/.h,
                                    # and copies them into this folder
```

`export_c_array.py` writes into this folder itself, renaming `.cc` to `.cpp` on
the way. Both details matter and neither fails loudly:

- **The `.cc` → `.cpp` rename is required.** The Arduino build compiles `.c`,
  `.cpp`, `.S` and `.ino` files in a sketch folder — `.cc` is silently ignored.
  Leave it as `.cc` and the build fails at link time with `undefined reference
  to g_crop_model_data`, which does not obviously point at a file extension.
- **The copy used to be manual, and it went stale.** This folder was found
  carrying a 9-class model while the trained one had 5. Nothing about that
  fails: the sketch compiles, boots, and reports confident nonsense against the
  wrong label table. It is automatic now; the destination is `FIRMWARE_DIR` in
  `config.py`, set it to `None` to opt out.

`model_data.h` also carries the label list and the class enum, and the sketch
uses them directly rather than keeping its own copy. That is deliberate: the
index **is** the `crop_id` on the wire, so a sketch-side table that drifted one
position from the model would silently make the main board run the wrong fan
profile for every crop.

## 4. Wiring for upload

The ESP32-CAM has no USB port, so it needs a USB-TTL (FTDI/CP2102) adapter set
to **5 V** or with a separate 5 V supply:

| ESP32-CAM | Adapter |
|---|---|
| 5V | 5V |
| GND | GND |
| U0T | RX |
| U0R | TX |
| GPIO0 | GND — **only while uploading** |

Sequence: jumper GPIO0→GND, press RESET, hit Upload. When `Hard resetting via
RTS pin...` appears, remove the GPIO0 jumper and press RESET again. The sketch
does not run while GPIO0 is grounded — that is flash mode.

Do not power the camera from a 3.3 V pin. It browns out the moment the radio
transmits. The model makes this worse, not better: inference holds the CPU at
full clock for seconds at a time.

> First compile takes several minutes — TFLM is built from source. Later builds
> are cached and much faster.

This build is verified. On ESP32 core 3.3.11 with `tflm_esp32` 2.0.0 it links at:

```
Sketch uses 2094785 bytes (66%) of program storage space. Maximum is 3145728 bytes.
Global variables use 90344 bytes (27%) of dynamic memory, leaving 237336 bytes
for local variables. Maximum is 327680 bytes.
```

66% of flash with the model in it, and 27% of internal RAM before the arena.
Whether the remaining internal SRAM can hold the 176 KB arena decides how fast
inference runs — see the boot log below, which says which memory it got.

You will also see this, twice, and it is harmless:

```
Library tflm_esp32 has been declared precompiled:
Precompiled library in ".../tflm_esp32/src/esp32" not found
```

The library ships a prebuilt archive for the ESP32-**S3** only. For plain ESP32
there is nothing to find, so it falls back to compiling from source — which is
what you want, and why the first build is slow.

## 5. What a good boot looks like

Serial Monitor at **115200 baud**:

```
[CAM] FarmFrost ESP32-CAM starting
[CAM] PSRAM: found
[CAM] Camera ready
[CV] Arena: 180224 bytes in internal SRAM (fast)
[CV] Model loaded: 989304 bytes of weights
[CV] Arena used: 159548 of 180224 bytes
[CV] Input  int8 96x96x3, scale 1.000000 zp -128
[CV] Output int8 5 classes, scale 0.003906 zp -128
[CAM] Desktop viewer ready
[CAM]   1. Connect this computer to Wi-Fi "FarmFrost-CAM" (password: farmfrost)
[CAM]   2. Open http://192.168.4.1
```

Then every 5 s:

```
[CAM] Image captured (320x240, 11482 bytes)
[CAM] Crop: tomato
[CAM] Confidence: 99%
[CAM] prep 61ms, infer 1840ms | mean RGB 142,98,71
```

Once it is running, **`Arena used:` tells you what to set `CROP_ARENA_BYTES` to**
in `crop_model.h` — that figure plus ~10%. It is 176 KB, measured rather than
guessed, and the size is not only about memory: at 600 KB the arena could only
ever live in PSRAM, while at this size it may fit internal SRAM, which is
several times faster for the scattered access inference does. `cropModelInit()`
tries internal first, falls back to PSRAM, and prints which it got.

If `AllocateTensors` fails, raising this is the *second* thing to try. See
Troubleshooting first — a per-channel-quantized classifier head produces the
same error message and is not a memory problem at all.

## 6. Seeing it on the desktop

The board makes its own Wi-Fi network — no router, no internet, so the offline
requirement in the spec stays intact.

1. Connect the laptop's Wi-Fi to **FarmFrost-CAM** / `farmfrost`.
   Windows warns "No internet, open anyway" — expected.
2. Open <http://192.168.4.1>.

The page shows the captured frame plus crop, confidence, timings and the mean
RGB check. The image is always the exact frame that was classified, never a
newer one, so the picture and the reading never disagree.

### Check the colour channels before trusting any accuracy number

The page prints **mean RGB** of the 96×96 tensor the model actually received.
Point the camera at something strongly red — a tomato, a red cloth. **R should
come back clearly above B.**

If they are reversed, set `CROP_SWAP_RB` to `0` in `crop_model.h` and reflash.

This is worth doing once, deliberately, because the failure is silent. The
esp32-camera JPEG decoder emits **BGR**, not RGB — its converter is shared with
the BMP writer, where BGR is the on-disk format. The model was trained on RGB.
Feed it BGR and nothing errors; you just get a model that is quietly much worse,
and this model leans hard on colour — a yellow lemon against a red tomato is
most of what separates those two classes. `CROP_SWAP_RB` defaults to `1` on the assumption that
the decoder is doing this. Verify rather than assume.

## 7. Testing the model against the desktop

The point of this section: the board and `predict.py` should agree. If they do,
the numbers measured on the desktop (72/72 on real rig photographs) carry over.
If they disagree, something in the imaging chain differs and it is worth finding
before blaming the model.

ESP-NOW is on by default now and does no harm here — if the NodeMCU is not
powered the sends simply fail and say so. Set `ENABLE_ESPNOW 0` if you want the
log quiet.

1. Put one item in the box, close it as you would for a real reading, and watch
   the serial monitor at 115200:

   ```
   [CAM] Crop: tomato
   [CAM] Confidence: 99%
   [CAM] prep 61ms, infer 1840ms | mean RGB 142,98,71
   ```

2. Open `http://192.168.4.1` after joining the `FarmFrost-CAM` Wi-Fi and save
   the frame it shows.

3. Run the same frame through the desktop harness:

   ```powershell
   cd "Crop detection model"
   python predict.py C:\path	o\saved_frame.jpg --top 3
   ```

The two should name the same class. Small confidence differences are expected —
the board decodes a JPEG the browser re-encoded — but the winner should match.

### What to expect

Each of the four items, centred in the box, should come back at high confidence;
on the desktop the median was 99.6% and the worst single real photograph was
64.5%. An empty box should read `empty` at ~99%.

Two failure modes worth telling apart:

- **Everything reads as one class, or confidence sits near 20%.** Suspect the
  model or the label table, not the camera. Check `[CV] Output int8 5 classes`
  in the boot log — if it says 9, this folder has a stale `model_data.cpp`.
- **Colours look wrong and confidence is mediocre but not random.** Suspect the
  channel order. Point the camera at the tomato and read `mean RGB` from the
  log: red should be clearly the largest. If blue is, flip `CROP_SWAP_RB` in
  `crop_model.h`. Section 6 covers this.

### If the board disagrees with the desktop

In order of likelihood:

1. **Framing.** The desktop harness centre-crops to a square. If the item sits
   near the bottom edge of the camera's view it can fall outside that crop on
   the board while surviving it in a phone photo. Move the item towards the
   centre of frame, or move the camera back.
2. **Exposure.** The box is nearly all white, so auto-exposure has an easy time
   but tends to sit bright. A washed-out frame loses the colour that separates
   lemon from tomato.
3. **Channel order.** See above.

### If a class is consistently wrong

That is a data problem, not a firmware one. Photograph the failing item in the
box — twenty or so frames, varied distance and position — drop them into
`Crop detection model/my images/from phone camera/<class>/`, and re-run:

```powershell
python scriptsuild_rig_dataset.py
python scripts	rain.py
python scripts\export_tflite.py
python scripts\export_c_array.py
```

then re-flash. **Banana especially**: it currently has no real photographs at
all and is trained entirely on Kaggle bananas composited into the box, so it is
the class most likely to disappoint on the real camera.

## 8. ESP-NOW

**On by default**, with a real NodeMCU SoftAP MAC already filled in
(`8E:AA:B5:4F:E4:66`). If you are using a different NodeMCU you must change
`MAIN_ESP_MAC` — the sketch prints a loud boot warning if it is ever left at the
old `AA:BB:CC:...` placeholder, but it cannot tell that a real-looking address
belongs to somebody else's board.

1. Flash `../Main_ESP8266/Main_ESP8266.ino` and read its serial output. It
   prints two MACs and labels them — use the **SoftAP MAC**.
2. Paste it into `MAIN_ESP_MAC` in this sketch.
3. `WIFI_CHANNEL` must be the same number in both sketches (1 by default).

Two failure modes worth knowing, both previously silent and both now fixed:

- **`peer.ifidx`.** Zero-initialising `esp_now_peer_info_t` leaves it at
  `WIFI_IF_STA`, but with the viewer on the board runs `WiFi.mode(WIFI_AP)` and
  the station interface never starts. Every send failed with
  `ESP_ERR_ESPNOW_IF` while classification looked perfect. The sketch now sets
  `WIFI_IF_AP` when the viewer is enabled.
- **`esp_now_send()`'s return value** was discarded. That is the local call
  failing outright — wrong interface, peer not added — and is distinct from the
  send *callback*, which only reports whether the frame was acknowledged. Both
  are logged now.

An `empty` reading is never transmitted: it is a correct answer with no crop to
act on, so the fan holds whatever the last real crop set.

The SoftAP-vs-station MAC distinction matters: the NodeMCU is in access-point
mode so the phone can join it, and ESP-NOW must be addressed to the AP-side
address. Sending to the station MAC fails *silently* — the send callback still
reports `delivered`, because ESP-NOW only confirms the frame left the radio.

Only `crop_id` + `confidence` + `seq` are sent — 6 bytes. **The image is never
sent.** ESP-NOW caps a payload at 250 bytes, and section 2 of the spec rules it
out anyway; the main board has no use for pixels.

## 9. Performance, and what is honest about it

Expect **roughly 1.5–4 s per inference** on this board. The plain ESP32 has no
vector unit (that is the S3), and the arena sits in PSRAM, which is several
times slower than internal SRAM. `DETECT_INTERVAL_MS` is 5000 to leave room.

`loop()` is single-threaded, so **the web page stops responding during
inference** and will skip a poll every cycle. That is expected here, not a bug.

If it needs to be faster:

- `ESP_TF` (Library Manager) bundles **esp-nn**, Espressif's optimised kernels.
  Meaningful speedup, and `crop_model.cpp` should build against it unchanged.
- Retrain at α=0.35, or at 64×64 instead of 96×96.
- An ESP32-**S3** board would be several times quicker on the same model.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `#error "No TensorFlow Lite Micro library found"` | `tflm_esp32` not installed — section 2. |
| `multiple definition of tflm_signal::...` or `kiss_fft_...` | The library's duplicated signal sources. Patch it — section 2. |
| `undefined reference to g_crop_model_data` | `model_data` is still `.cc`. Rename to `.cpp` — section 3. |
| `text section exceeds available space` | Partition Scheme is not Huge APP. |
| `[CV] No PSRAM` | PSRAM disabled under Tools, or a module without it. |
| `FullyConnected per-channel quantization not yet supported`, then `AllocateTensors failed` | **Not a memory problem**, despite the message. The classifier head was exported per-channel and TFLM only implements per-tensor for it. Re-export: `export_tflite.py` sets `_experimental_disable_per_channel_quantization_for_dense_layers`. |
| `[CV] AllocateTensors failed`, no other message | Raise `CROP_ARENA_BYTES` in `crop_model.h`. |
| `[CV] Arena: ... in PSRAM -- expect much slower inference` | Not an error. Internal SRAM was full; inference still works, just slower. |
| `[CV] Model schema N, library expects M` | Library and export are different TFLM generations. Re-export, or change library version. |
| Every crop reads as one class, high confidence | Almost always the R/B swap — section 6. |
| Viewer picture never updates while serial shows captures | Fixed: the cache-buster was keyed to the *detection* counter, which never advances when inference fails. It uses the capture counter now. |
| Classification works, fan never moves | ESP-NOW addressed to the wrong interface or the wrong MAC — section 8. |
| Confidence always low | Lighting, or the object is not filling the centre square. The viewer shows the full frame; the model only sees the centre crop. |
| `Camera init failed: 0x105` | Ribbon cable not seated, or 3.3 V power. Reseat, use 5 V. |
| Board reboots in a loop | Brownout. Needs a supply that can do ~500 mA at 5 V. |
| `Failed to connect... Timed out waiting for packet header` | GPIO0 not grounded, or RESET not pressed before upload. |
| Sketch never starts, only garbage on serial | GPIO0 still grounded after upload. |
| Red LED blinking fast forever | Camera init failed — the sketch halted deliberately. |
| Page loads but image is broken | Wait one detection cycle; nothing is captured for the first few seconds. |
