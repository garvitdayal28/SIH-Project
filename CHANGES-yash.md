# Changes on this branch

Everything below was done bringing up one real ESP32-CAM + NodeMCU + L298N
pair after cloning the repo — fixes for bugs hit during bring-up, plus wiring
the fan controller for an H-bridge instead of a bare MOSFET. Grouped by file.

---

## `ESP Firmware/ESP32_CAM/ESP32_CAM.ino`

**ESP-NOW would not deliver a single packet, silently.**
Zero-initializing `esp_now_peer_info_t` leaves `peer.ifidx` at `WIFI_IF_STA`.
With `ENABLE_WEB_VIEWER` on, the sketch runs `WiFi.mode(WIFI_AP)`, so the
station interface never comes up — every send failed with
`ESP_ERR_ESPNOW_IF`, and `WiFi.macAddress()` printed `00:00:00:00:00:00`
because the STA interface has no address in AP mode. Fixed by setting
`peer.ifidx = WIFI_IF_AP` when the viewer is on, and printing the *SoftAP*
MAC instead of the (meaningless, in this mode) station MAC. Also added a
`Serial.printf` on `esp_now_send()`'s own return value, which reports the
local call failing outright — distinct from the sent-callback, which reports
whether the *peer* acknowledged the frame.

**`MAIN_ESP_MAC` was still the all-`0xFF` placeholder.**
Filled in with the real NodeMCU SoftAP MAC (`8E:AA:B5:4F:E4:66`, cross-checked
two ways: scanning the `FarmFrost` AP's BSSID over Wi-Fi, and reading it
directly from the NodeMCU's own boot log). Also added a boot-time check that
prints a hard-to-miss warning if `MAIN_ESP_MAC` is ever left at the
placeholder again — sending to a nonexistent peer used to fail with no
visible symptom at all.

**The live viewer froze on the first frame and never updated.**
The image URL was cache-busted with `g_seq`, which only increments on a
*successful* inference. Any run of failed inferences (model not loaded,
decode failure, whatever) left the URL unchanged, so the browser kept serving
its cached copy of frame #1 forever while the serial log showed captures
ticking by underneath. Added a separate `g_capSeq` counter that increments on
every capture regardless of inference outcome, and keyed the image URL and
polling to that instead.

**Added a flash LED sync for capture.**
`FLASH_LED_PIN` now lights for the capture itself (`FLASH_SETTLE_MS` for the
sensor's auto-exposure to adapt, `FLASH_DISCARD_FRAMES` pre-flash frames
dropped) and switches off again before inference — not for the several
seconds inference takes, which would have made the LED and the 5 V rail run
far hotter than needed. Deliberately `digitalWrite`, not PWM/LEDC: the
camera driver already owns an LEDC timer for the 20 MHz XCLK at the IDF
level, invisible to Arduino's LEDC allocator, so `ledcAttach()` could
silently steal and reconfigure that same timer and kill the pixel clock.

**Inference measured at ~19.5 s/frame, consistently, on this board.**
Far past the README's documented 1.5–4 s estimate. Root-caused to the tensor
arena landing in PSRAM. Added an internal-SRAM-first allocation
(`heap_caps_malloc(..., MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)`, falling back
to `ps_malloc` only if that fails) in `crop_model.cpp`/`.h` — see below.
**On this board internal SRAM is still full by the time the arena is
requested** (camera driver + ~90 KB of static globals), so it still falls
back to PSRAM and inference is still ~19.5 s. Left the fallback logic in
since it's correct and free, and left the real fix (see Follow-ups) for
whoever picks this up next.

**`DETECT_INTERVAL_MS` changed 5000 → 1000, not to 0.**
Tried 0 (back-to-back capture) first — it does raise throughput, but
`loop()` is single-threaded, so the entire gap that used to let the web
server answer a request disappears with it: the viewer page stops updating
even though detection keeps running. 1000 ms is the compromise that leaves
room for roughly one status poll + one image fetch per cycle.

**JSON status buffer widened 384 → 512 bytes.**
Added the new `cap` field to the payload; `snprintf` truncates rather than
overflows, but a truncated response is just invalid JSON and looks like a
hung board.

---

## `ESP Firmware/ESP32_CAM/crop_model.h` / `crop_model.cpp`

**`CROP_ARENA_BYTES` lowered 600 KB → 176 KB.**
600 KB is guaranteed too large to ever fit in internal SRAM on this chip, so
it forced PSRAM unconditionally. The real measured usage on this board is
159548 bytes (`[CV] Arena used: ...`, printed after `AllocateTensors()`);
176 KB gives ~10% headroom. This number is *specific to the current model* —
re-measure and adjust after retraining.

**Arena allocation now tries internal SRAM first.**
`heap_caps_malloc(CROP_ARENA_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)`,
falling back to `ps_malloc()` (PSRAM) only if that fails. Logs which one it
got. Every intermediate activation during inference is read/written through
this buffer, and PSRAM is reached over a much slower bus — this is the
single biggest inference-speed lever on this chip, when it lands.

---

## `ESP Firmware/ESP32_CAM/model_data.cpp`

**The committed model failed to load at all.**
Boot log:
```
FullyConnected per-channel quantization not yet supported.
Node FULLY_CONNECTED (number 30f) failed to prepare with status 1
[CV] AllocateTensors failed -- raise CROP_ARENA_BYTES
```
The last line is a red herring the kernel itself doesn't produce — TFLM's
`fully_connected` kernel asserts `filter_quantization->scale->size == 1` and
the exported classifier head was quantized **per-channel** (9 scales, one per
class). `tflm_esp32` has no per-channel path for this op at all (Conv2D and
DepthwiseConv2D do support it; only the FC head doesn't).

No dataset, `model.keras`, TensorFlow, or Python 3.11 were available on this
machine to retrain, so the fix was done directly on the exported flatbuffer:
extracted the `.tflite` from the C array, located the FC filter/bias
quantization tensors (confirmed unshared with any other tensor — safe to
resize in place), collapsed the 9 per-channel scales to one per-tensor scale,
requantized the int8 weights and int32 bias against it, and regenerated the
C array with the same formatting and identical byte length (991560 bytes).

Costs almost nothing: the original per-channel scales spanned only 1.30×, so
the max weight quantization error introduced is 7.99e-4 — half a
quantization step, i.e. pure rounding, with zero weights clipped. Verified by
round-tripping the patched `.tflite` back out of the regenerated `.cpp` and
diffing byte-for-byte against the patched source (identical), and by
re-parsing the flatbuffer schema to confirm the FC filter/bias report
`nscale=1` while the 54 Conv2D/DepthwiseConv2D tensors elsewhere in the model
are untouched.

---

## `Crop detection model/scripts/export_tflite.py`

Added `converter._experimental_disable_per_channel_quantization_for_dense_layers
= True` with a comment explaining the failure above, so a future retrain
produces a model `tflm_esp32` can actually load instead of reproducing this.

---

## `ESP Firmware/Main_ESP8266/Main_ESP8266.ino`

**Fan output rewired from a single MOSFET pin to an L298N H-bridge.**
The original code assumed one PWM pin (`FAN_PWM_PIN`, D5) into a MOSFET
gate. The hardware on hand is an L298N, which needs three signals:

| Signal | Pin | GPIO | Role |
|---|---|---|---|
| ENA | D5 | GPIO14 | PWM — sets speed |
| IN1 | D6 | GPIO12 | direction bit |
| IN2 | D7 | GPIO13 | direction bit |

D5/D6/D7 chosen because none has a role at the ESP8266 boot sequence, unlike
D3/D4/D8 which the bootloader samples or pulls.

`setFanSpeed()` now also:
- drives `IN1=HIGH, IN2=LOW` for the (only) forward direction, and both LOW
  ("coast") when the target speed is 0,
- treats anything below `FAN_MIN_DUTY_PCT` (20%, new) as 0 rather than
  driving a duty the motor can only hum at without turning — relevant
  because ginger/onion in the placeholder `FAN_SPEED` table sit at 40%/35%,
  and the L298N itself eats another 1.4–2 V off the top,
- kicks the motor at 100% duty for `KICKSTART_MS` (250 ms, new) when
  starting from a full stop before settling to the target duty, since a
  stopped motor needs more duty to break away than to keep spinning. Only
  fires from rest — changing between two non-zero speeds skips it.

`setup()` now configures and parks `IN1`/`IN2` (both LOW) before PWM is
armed, so the H-bridge can't glitch while pins settle at boot.

---

## `ESP Firmware/Main_ESP8266/README.md`

Rewrote the "Fan wiring" section to match the above: L298N pin table,
"remove the ENA jumper" (ships tied to +5 V on most boards — the single most
common reason PWM appears to do nothing), the 5V-EN / power-source warning,
common-ground note, and a new subsection documenting the stall-floor and
kickstart settings and how to tune them for a different motor.

---

## Verified so far, on real hardware

- ESP32-CAM boots, camera initializes, PSRAM found.
- Model loads and runs inference (post fix); confidences in the 33–74% range
  observed so far (unfocused/poorly lit test subjects — not yet a fair
  accuracy test).
- ESP-NOW delivers: `[CAM] ESP-NOW packet delivered` confirmed on-device
  after the `peer.ifidx` fix and the real MAC.
- `Main_ESP8266.ino` and `ESP32_CAM.ino` both compile clean against the
  project's documented toolchain (esp8266 core 3.1.2 / esp32 core 3.3.11 +
  `tflm_esp32` 2.0.0, patched per the ESP32_CAM README).

## Not yet verified

- The L298N + motor wiring above is **not yet confirmed spinning a real
  motor** end-to-end from a live detection — verify the ENA jumper is
  removed and the kickstart/stall-floor values suit the actual motor before
  trusting it unattended.
- Model accuracy, generally — needs the camera focused and well-lit test
  subjects, not the placeholder captures used to prove the pipeline.

## Follow-ups worth doing next

- Inference is ~19.5 s/frame on this specific board because the arena still
  lands in PSRAM (internal SRAM is exhausted by the camera driver + web
  viewer's static allocations before the arena is even requested). Two real
  options, not yet attempted: swap `tflm_esp32` for `ESP_TF` (Espressif's
  esp-nn-optimized kernels — the real fix, at the cost of redoing the
  duplicate-symbol library patch), or set `ENABLE_WEB_VIEWER 0` to free
  enough internal SRAM for the 176 KB arena to fit directly (loses the live
  page, keeps the fan pipeline).
- All non-spec `FAN_SPEED[]` entries (apple, corn, ginger, lemon) are still
  placeholders, not real FarmFrost storage requirements.
- `AP_PASSWORD` on the NodeMCU is still the shipped default; change before
  any real demo.
