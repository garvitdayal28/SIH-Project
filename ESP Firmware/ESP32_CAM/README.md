# ESP32-CAM — FarmFrost V1

Capture → (crop model) → desktop viewer + ESP-NOW.

Right now the capture and the viewer are real; the crop model is a placeholder
that rotates through the crop list so the rest of the pipeline can be tested.

## 1. One-time Arduino IDE setup

1. **File → Preferences → Additional boards manager URLs**, add:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
2. **Tools → Board → Boards Manager**, search `esp32`, install
   **esp32 by Espressif Systems**. (Large download, ~1 GB on disk.)
3. Select these under **Tools**:

   | Setting | Value |
   |---|---|
   | Board | AI Thinker ESP32-CAM |
   | PSRAM | Enabled |
   | Partition Scheme | Huge APP (3MB No OTA/1MB SPIFFS) |
   | Upload Speed | 921600 (drop to 115200 if it fails) |

   The partition scheme matters later, not now — the crop model is ~968 KB and
   will not fit the default 1 MB app partition. Setting it now means nothing
   changes when the model goes in.

## 2. Wiring for upload

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
transmits.

## 3. Seeing the image on the desktop

The board makes its own Wi-Fi network — no router, no internet, which keeps the
offline requirement in the spec intact.

1. Open Serial Monitor at **115200 baud**. You should see:
   ```
   [CAM] FarmFrost ESP32-CAM starting
   [CAM] PSRAM: found
   [CAM] Camera ready
   [CAM] Desktop viewer ready
   [CAM]   1. Connect this computer to Wi-Fi "FarmFrost-CAM" (password: farmfrost)
   [CAM]   2. Open http://192.168.4.1
   ```
2. Connect the laptop's Wi-Fi to **FarmFrost-CAM** / `farmfrost`.
   Windows will warn "No internet, open anyway" — that is expected and fine.
3. Open <http://192.168.4.1>.

The page shows the captured frame plus the crop, confidence and frame number,
refreshing every 3 s. The image you see is always the exact frame that was
classified, not a newer one, so picture and reading never disagree.

Serial keeps logging regardless:

```
[CAM] Image captured (28714 bytes)
[CAM] Crop: TOMATO (placeholder)
[CAM] Confidence: 92%
```

## 4. Turning on ESP-NOW

Off by default because it needs the main board's MAC address.

1. Flash `../Main_ESP8266/Main_ESP8266.ino` and read its serial output. It
   prints two MACs and labels them — use the **SoftAP MAC**.
2. Paste it into `MAIN_ESP_MAC` in this sketch.
3. Set `#define ENABLE_ESPNOW 1`.
4. `WIFI_CHANNEL` must be the same number in both sketches (1 by default).

The SoftAP-vs-station MAC distinction matters: the NodeMCU is in access-point
mode so the phone can join it, and ESP-NOW must be addressed to the AP-side
address. Sending to the station MAC fails *silently* — the send callback still
reports `delivered`, because ESP-NOW only confirms the frame left the radio.

Only `crop_id` + `confidence` + `seq` are sent — 6 bytes. **The image is never
sent.** ESP-NOW caps a payload at 250 bytes, and section 2 of the spec rules it
out anyway; the main board has no use for pixels.

## 5. Replacing the placeholder model

`runInference()` is the only function that changes. It must reproduce what
`Crop detection model/cropnet/preprocess.py` does on the desktop, or the model
sees a different picture than it was trained on:

1. Decode JPEG → RGB (`fmt2rgb888()` from `img_converters.h`, or capture a
   second frame as `PIXFORMAT_RGB565`).
2. Centre-crop to a square.
3. Downscale to 96×96 (`config.IMAGE_SIZE`).
4. int8 quantize, run with **esp-tflite-micro**.
5. Return argmax + softmax as 0–100.

The crop table in the sketch is in the same order as `class_names()` in
`config.py` — crops alphabetically, then `unknown` last. The index **is** the
`crop_id`, so it cannot be rearranged without retraining.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Camera init failed: 0x105` | Ribbon cable not seated, or 3.3 V power. Reseat, use 5 V. |
| Board reboots in a loop | Brownout. Needs a supply that can do ~500 mA at 5 V. |
| `Failed to connect... Timed out waiting for packet header` | GPIO0 not grounded, or RESET not pressed before upload. |
| Sketch never starts, only garbage on serial | GPIO0 still grounded after upload. |
| Red LED blinking fast forever | Camera init failed — the sketch halted deliberately. |
| Page loads but image is broken | Wait one detection cycle; nothing is captured for the first ~3 s. |
