# FarmFrost Firmware V1 — Embedded System Overview

## 1. Objective

Build a small offline prototype of the FarmFrost embedded system.

The system has two ESP devices:

- **ESP32-CAM** — captures an image and uses a computer-vision model to identify the crop.
- **Main ESP32** — receives the detected crop, determines the appropriate fan speed, controls the fan using PWM, and exposes the result to the FarmFrost mobile app over BLE.

There must be **no internet, cloud server, Wi-Fi router, or backend dependency**.

### Target data flow

```text
ESP32-CAM
    │
    │ Crop + confidence
    │ ESP-NOW
    ▼
Main ESP32
    │
    ├──► Fan controller
    │      │
    │      └──► PWM → Fan
    │
    └──► BLE
             │
             ▼
       FarmFrost Mobile App
```

---

# 2. Responsibilities

## ESP32-CAM

The ESP32-CAM is responsible only for vision-related work:

1. Initialize the camera.
2. Capture an image.
3. Preprocess the image as required by the CV model.
4. Run crop-detection inference.
5. Produce:
   - Crop ID/name
   - Confidence score
6. Send the result to the Main ESP32 using ESP-NOW.
7. Repeat the detection cycle at a reasonable interval.

### Important

Do **not** send the captured image to the Main ESP32.

The camera should send only the inference result, for example:

```text
crop_id = TOMATO
confidence = 92%
```

This keeps communication fast, lightweight, and reliable.

---

# 3. Main ESP32 Responsibilities

The Main ESP32 is the central controller.

It should:

1. Initialize ESP-NOW.
2. Receive crop-detection results from the ESP32-CAM.
3. Validate the received data.
4. Determine the appropriate fan speed for the detected crop.
5. Control the fan using PWM.
6. Run a BLE GATT server.
7. Send the current crop, confidence, and fan speed to the mobile app.
8. Continue controlling the fan even when the phone is disconnected.

The Main ESP32 should **not depend on the mobile app for control**.

The fan-control logic must work autonomously.

---

# 4. ESP32-CAM → Main ESP32 Communication

Use **ESP-NOW** for communication between the ESP32-CAM and Main ESP32.

Reasons:

- Completely offline.
- No router required.
- No internet required.
- Direct ESP-to-ESP communication.
- Low latency.
- Suitable for small data packets.

### Basic packet

The first version should keep the packet very small.

Conceptually:

```cpp
struct CropResult {
    uint8_t crop_id;
    uint8_t confidence;
};
```

Example:

```text
crop_id    = 1
confidence = 92
```

The Main ESP32 interprets the crop ID using a predefined crop table.

Later versions may add:

- Sequence number
- Timestamp
- Protocol version
- Model version
- Packet checksum/validation
- ACK/retry mechanism

Do not add unnecessary complexity to V1.

---

# 5. Crop Detection Pipeline

The ESP32-CAM pipeline should be:

```text
Camera
   ↓
Capture image
   ↓
Preprocessing
   ↓
CV model
   ↓
Crop classification
   ↓
Confidence score
   ↓
ESP-NOW packet
   ↓
Main ESP32
```

Example:

```text
Image
  ↓
CV Model
  ↓
Tomato — 92%
  ↓
ESP-NOW
```

The model should perform inference on the ESP32-CAM.

Model training should happen externally on a development machine. The ESP32-CAM should perform inference only.

---

# 6. Fan-Control Logic

For V1, fan speed should be controlled using a simple deterministic lookup table.

Do **not** use another ML model to determine fan speed at this stage.

Example:

| Crop | Example Fan Speed |
|---|---:|
| Tomato | 70% |
| Potato | 45% |
| Onion | 35% |
| Banana | 60% |
| Mango | 65% |

These values are placeholders for development. Replace them with the actual values determined from the FarmFrost storage requirements.

Conceptually:

```text
Detected Crop
     ↓
Crop → Fan-Speed Lookup
     ↓
Target Fan Speed
     ↓
PWM
     ↓
Fan
```

Example:

```text
Tomato
  ↓
70%
  ↓
PWM
  ↓
Fan
```

---

# 7. Fan Hardware

The ESP32 GPIO must **not directly power the fan**.

Use an appropriate:

- MOSFET
- Transistor/fan driver
- External power supply

The exact driver must be selected according to the fan's voltage and current requirements.

The ESP32 generates the PWM control signal.

Example logical levels:

```text
0%    → Fan OFF
25%   → Low
50%   → Medium
75%   → High
100%  → Maximum
```

The actual PWM frequency and electrical configuration should be chosen according to the fan/driver hardware.

---

# 8. Main ESP32 Decision Flow

When a valid crop result is received:

```text
ESP-NOW packet received
        ↓
Validate packet
        ↓
Identify crop
        ↓
Look up fan speed
        ↓
Set PWM
        ↓
Update current system state
        ↓
Send BLE update
```

Conceptually:

```cpp
onCropDetected(result)
{
    fanSpeed = getFanSpeed(result.crop_id);

    setFanSpeed(fanSpeed);

    sendBLEUpdate(result, fanSpeed);
}
```

The Main ESP32 should always maintain the latest valid state.

---

# 9. BLE Communication

Use **BLE GATT** between the Main ESP32 and the FarmFrost mobile app.

Architecture:

```text
Main ESP32
    │
    │ BLE
    ▼
FarmFrost Mobile App
```

The Main ESP32 acts as the BLE peripheral/GATT server.

The mobile app acts as the BLE client.

The app should be able to receive:

- Detected crop
- Crop confidence
- Current fan speed
- Device connection/status information

For example:

```json
{
  "crop": "tomato",
  "confidence": 92,
  "fan_speed": 70
}
```

For V1, a simple structured payload is acceptable. Optimize into a compact binary protocol later if necessary.

---

# 10. Prefer BLE Notifications

The mobile app should subscribe to updates rather than repeatedly polling the ESP32.

Preferred flow:

```text
ESP32 state changes
       ↓
BLE Notification
       ↓
Mobile App
       ↓
Update UI
```

Avoid constantly doing:

```text
Phone → ESP32: "What is the crop?"
ESP32 → Phone: "Tomato"

Phone → ESP32: "What is the fan speed?"
ESP32 → Phone: "70%"
```

Instead, the ESP32 should push the latest state when it changes.

---

# 11. Mobile App Requirements

The mobile app only needs a minimal dashboard for V1.

Example:

```text
FARMFROST

Detected Crop
TOMATO

Confidence
92%

Fan Speed
70%

Device
● Connected
```

The app is a **display/monitoring interface**, not the controller.

The app should not be required for the fan to operate.

---

# 12. Offline Requirement

The entire V1 system must work without:

- Internet
- Cloud services
- REST APIs
- Firebase
- MQTT broker
- Backend server
- Wi-Fi router

Communication should be:

```text
ESP32-CAM
    │
    │ ESP-NOW
    ▼
Main ESP32
    │
    │ BLE
    ▼
Mobile App
```

This is the complete communication path.

---

# 13. Firmware Technology Recommendations

Use:

- **C/C++** for ESP32 firmware.
- **ESP-IDF** as the primary ESP32 development framework.
- **FreeRTOS** through ESP-IDF when concurrency becomes necessary.
- **ESP-NOW** for ESP32-CAM → Main ESP32.
- **BLE GATT** for Main ESP32 → Mobile App.
- A lightweight embedded-compatible CV/ML inference runtime for the crop model.

The initial implementation should remain simple even if the underlying framework supports more advanced architecture.

---

# 14. Development Strategy

Implement and test the system incrementally.

## Step 1 — Fan Control

First verify:

```text
Main ESP32
   ↓
PWM
   ↓
Fan
```

Test:

```text
0%
25%
50%
75%
100%
```

Do this before integrating the camera.

---

## Step 2 — Crop Detection

Independently verify:

```text
ESP32-CAM
   ↓
Camera
   ↓
CV Model
   ↓
Crop + Confidence
```

Print the result to the serial monitor.

Example:

```text
Detected: TOMATO
Confidence: 92%
```

---

## Step 3 — ESP-NOW

Initially use fake crop data.

Example:

```text
ESP32-CAM
    ↓
TOMATO, 92%
    ↓
ESP-NOW
    ↓
Main ESP32
```

Verify reliable reception before connecting the actual CV model.

---

## Step 4 — Connect CV + ESP-NOW

Replace the fake data with the actual model output:

```text
Camera
   ↓
CV model
   ↓
Tomato, 92%
   ↓
ESP-NOW
   ↓
Main ESP32
```

---

## Step 5 — Automatic Fan Control

Implement:

```text
Tomato
   ↓
Lookup table
   ↓
70%
   ↓
PWM
   ↓
Fan
```

Verify that changing the detected crop changes the fan speed correctly.

---

## Step 6 — BLE

Independently verify:

```text
Main ESP32
   ↓
BLE
   ↓
Phone
```

Then transmit:

```text
Crop
Confidence
Fan speed
Status
```

---

## Step 7 — Full Integration

The final V1 flow must work end-to-end:

```text
┌────────────────┐
│   ESP32-CAM    │
│                │
│ Camera + CV    │
└───────┬────────┘
        │
        │ ESP-NOW
        ▼
┌────────────────┐
│   Main ESP32   │
│                │
│ Crop received  │
│      ↓         │
│ Fan lookup     │
│      ↓         │
│ PWM controller │
└───────┬────────┘
        │
        │ BLE
        ▼
┌────────────────┐
│ FarmFrost App  │
│                │
│ Crop: Tomato   │
│ Confidence:92% │
│ Fan: 70%       │
└────────────────┘
```

---

# 15. Reliability Requirements

Even in V1, implement basic validation.

### Invalid crop ID

Do not change the fan speed.

### Low-confidence detection

Define a minimum confidence threshold.

For example:

```text
confidence < threshold
        ↓
Do not blindly change fan speed
```

The exact threshold should be configurable.

### ESP32-CAM unavailable

The Main ESP32 should continue running safely using its previous/default fan state rather than crashing.

### Mobile disconnected

Fan control must continue normally.

### ESP-NOW packet lost

The Main ESP32 should continue using the last valid state until a new valid result arrives.

---

# 16. Logging

During development, use clear serial logs.

Example:

```text
[CAM] Image captured
[CAM] Crop: TOMATO
[CAM] Confidence: 92%
[CAM] ESP-NOW packet sent

[MAIN] Crop received: TOMATO
[MAIN] Confidence: 92%
[MAIN] Fan target: 70%
[MAIN] PWM updated
[BLE] State notification sent
```

This will make debugging the complete pipeline much easier.

---

# 17. V1 Scope — Keep It Small

The first working prototype should contain ONLY:

### ESP32-CAM

- Camera
- Crop CV model
- Crop classification
- Confidence
- ESP-NOW transmission

### Main ESP32

- ESP-NOW receiver
- Crop → fan-speed lookup
- PWM fan control
- BLE GATT server
- Current-state transmission

### Mobile App

- BLE connection
- Crop display
- Confidence display
- Fan-speed display
- Connection status

Do **not** implement yet:

- Sensor fusion
- Second ML model
- Cloud/backend
- User accounts
- Remote control
- Historical analytics
- Complex storage
- OTA updates
- Advanced fault management
- Complex decision engine

Those can be added after the V1 pipeline is stable.

---

# 18. Definition of Done

V1 is complete when this happens reliably:

```text
1. A crop is placed in front of the ESP32-CAM.
              ↓
2. ESP32-CAM identifies the crop.
              ↓
3. ESP32-CAM produces crop + confidence.
              ↓
4. Result is sent to Main ESP32 through ESP-NOW.
              ↓
5. Main ESP32 identifies the crop.
              ↓
6. Main ESP32 selects the correct fan-speed profile.
              ↓
7. Main ESP32 changes the fan PWM.
              ↓
8. Main ESP32 sends crop + confidence + fan speed over BLE.
              ↓
9. FarmFrost mobile app displays the result.
```

The complete system must operate **without internet or any external network infrastructure**.

---

# 19. Implementation Principle

Prioritize:

1. **Reliability**
2. **Simplicity**
3. **Low latency**
4. **Offline operation**
5. **Easy debugging**
6. **Modularity**

Do not over-engineer V1.

The immediate goal is a reliable:

**Vision → ESP-NOW → Controller → Fan → BLE → App**

pipeline.

Once this pipeline is working, the system can be expanded with environmental sensors, the second ML model, smarter fan-control logic, data logging, and other FarmFrost features.
