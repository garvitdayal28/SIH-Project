# Main controller (NodeMCU ESP8266) — FarmFrost V1

ESP-NOW in → crop→fan lookup → PWM → Wi-Fi out to the app.

## Why Wi-Fi and not BLE

The V1 spec specifies BLE GATT for the phone link. The ESP8266MOD has **no
Bluetooth radio of any kind**, so that step is not implementable on this board.
This firmware runs a Wi-Fi access point instead and the phone joins it directly.

Nothing about the offline requirement changes — no router, no internet, no
backend, no broker. The phone associates straight to the board:

```
ESP32-CAM ──ESP-NOW──► NodeMCU ──Wi-Fi AP──► Phone
                          │
                          └──PWM──► MOSFET ──► Fan
```

The spec's "prefer notifications over polling" rule was aimed at BLE, where a
poll costs a GATT round trip. Over Wi-Fi the app fetches a ~150-byte JSON once
a second, which is free by comparison and keeps a WebSocket server off a board
with ~40 KB of usable heap. `arduinoWebSockets` on port 81 is the upgrade if
push is ever wanted.

## 1. Arduino IDE setup

1. **File → Preferences → Additional boards manager URLs**, add:
   ```
   https://arduino.esp8266.com/stable/package_esp8266com_index.json
   ```
2. **Boards Manager** → install **esp8266 by ESP8266 Community**.
3. **Tools → Board → NodeMCU 1.0 (ESP-12E Module)**. Defaults are fine for
   everything else. Upload over the onboard USB — no jumper needed.

No extra libraries. Everything used ships with the core.

## 2. Fan wiring

**The GPIO must never drive the fan directly** (spec §7). GPIO gives 3.3 V at a
few mA; a fan wants its own supply at amps.

This build drives the motor through an **L298N** H-bridge.

```
   NodeMCU                 L298N                    Motor / supply

   D5 (GPIO14) ──────────► ENA   (speed, PWM)
   D6 (GPIO12) ──────────► IN1   (direction)
   D7 (GPIO13) ──────────► IN2   (direction)
   GND ─────────────────── GND ──────────────────── supply GND
                           OUT1 ─────────────────── motor +
                           OUT2 ─────────────────── motor -
                           +12V ◄────────────────── motor supply +
                           +5V  ── (see note below)
```

- **Remove the ENA jumper.** L298N boards ship with a jumper tying ENA to +5 V,
  which pins the motor at full speed and ignores your PWM entirely. The single
  most common reason "the fan only runs flat out".
- **Do not feed the NodeMCU from the L298N's 5 V pin** unless you are sure the
  board has its onboard regulator enabled (the 5V-EN jumper) *and* the motor
  supply is 12 V or less. Powering the NodeMCU from USB while the motor runs
  off its own supply is the safe arrangement.
- **Grounds must be common** — NodeMCU GND to L298N GND to supply GND. Without
  it the direction inputs have no reference and the motor behaves erratically.
- The L298N drops roughly **1.4–2 V** across its output stage, so a 12 V supply
  gives the motor about 10 V. Size the supply accordingly.
- IN1/IN2 must **never both be HIGH**. The firmware only ever sets IN1 HIGH and
  IN2 LOW, so this is handled — but check it if you rewire.
- Test with an **LED + resistor on ENA** first, before a motor goes anywhere
  near it. It should visibly dim across the speed steps.

### Stall floor and kickstart

A stopped motor needs much more duty to break away than to keep turning. Two
settings in the sketch handle that:

| Setting | Default | Does |
|---|---|---|
| `FAN_MIN_DUTY_PCT` | 20 | Below this the motor is switched off rather than left buzzing |
| `KICKSTART_MS` | 250 | Drives 100% briefly when starting from rest, then drops to target |

If your motor still hums without turning at the lower crop speeds (onion is
35%, ginger 40%), raise `FAN_MIN_DUTY_PCT` or lengthen `KICKSTART_MS`.

## 3. Bring-up order

Follow the spec's incremental order — do not wire everything at once.

**Step 1 — fan only.** Flash as-is with nothing connected but an LED on D5.
No packets arrive, so the fan stays at `FAN_SPEED_DEFAULT` (0). To exercise the
PWM path, temporarily call `setFanSpeed(0/25/50/75/100)` on a timer in `loop()`.

**Step 2 — the link.** Read the serial output at 115200:

```
[MAIN] ================================================
[MAIN] Paste this into the ESP32-CAM sketch as MAIN_ESP_MAC:
[MAIN]   5E:CF:7F:1A:2B:3C   <- SoftAP MAC, this is the one to use
[MAIN]   5C:CF:7F:1A:2B:3C   (station MAC, NOT the one to use)
[MAIN] ================================================
```

Put the SoftAP MAC into `MAIN_ESP_MAC` in the camera sketch and set
`ENABLE_ESPNOW 1` there. The camera's placeholder model rotates through crops,
so you should see the fan step through the table every 3 s:

```
[MAIN] Crop received: TOMATO
[MAIN] Confidence: 92%
[MAIN] Fan target: 70%
[MAIN] PWM updated: 70%
```

**Step 3 — the phone.** Join Wi-Fi `FarmFrost` / `farmfrost`, open
<http://192.168.4.1>. Android will complain there is no internet; stay
connected. That page is the same dashboard the app has to draw.

## 4. What the app reads

`GET http://192.168.4.1/status`, once a second:

```json
{
  "crop": "TOMATO",
  "crop_id": 7,
  "confidence": 92,
  "fan_speed": 70,
  "cam_online": true,
  "threshold": 60,
  "seq": 412,
  "packets": 410,
  "ignored": 51,
  "uptime_s": 1237
}
```

`crop`, `confidence`, `fan_speed` and `cam_online` are the four fields the V1
dashboard in spec §11 needs. The rest are for debugging. CORS is open, so a
browser-based app can fetch it without a proxy.

The app is a **display only**. There is no endpoint that changes the fan, by
design — spec §11.

## 5. Reliability behaviour

All of spec §15, and what it does:

| Situation | Behaviour |
|---|---|
| `crop_id` outside the table | Fan unchanged, counted in `ignored` |
| Confidence < 60% | Fan unchanged, counted in `ignored` |
| `UNKNOWN` at high confidence | Fan unchanged (`UNKNOWN_KEEPS_PREVIOUS 1`) |
| Camera silent > 15 s | `cam_online: false`, **fan keeps running** |
| Packet lost | Gap logged, last valid state retained |
| Phone disconnects | No effect whatsoever on fan control |
| Wrong packet size | Dropped in the receive callback |

## 6. Before this is real

- `FAN_SPEED[]` is **placeholder data**. Four values come from spec §6, the
  rest are filler. Replace all of them with the actual FarmFrost storage
  requirements.
- `AP_PASSWORD` is a default. Change it before any demo.
- The crop table order must stay in step with `class_names()` in
  `Crop detection model/config.py` and with the camera sketch. The index is the
  `crop_id` on the wire.

## Troubleshooting

| Symptom | Cause |
|---|---|
| No packets, but camera says `delivered` | Sent to the station MAC instead of the SoftAP MAC. ESP-NOW confirms transmission, not reception. |
| No packets, `NOT delivered` on the camera | Channel mismatch. `WIFI_CHANNEL` must be identical in both sketches. |
| Fan never moves | `FAN_SPEED_DEFAULT` is 0 and nothing valid has arrived. Check `ignored` in `/status`. |
| Fan buzzes | Raise `PWM_FREQ_HZ` to 20000. |
| Phone drops the Wi-Fi | Android leaving a network with no internet. Disable "auto-switch to mobile data" for it. |
| Board reboots when the fan starts | Fan supply and logic supply are shared. Give the fan its own supply, common ground only. |
