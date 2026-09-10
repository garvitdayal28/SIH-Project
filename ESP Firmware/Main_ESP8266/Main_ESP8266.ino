/*
 * FarmFrost V1 -- Main controller firmware
 * ----------------------------------------
 * Board : NodeMCU 1.0 / ESP-12E (ESP8266MOD)
 *
 * Role
 *
 *   Receives crop_id + confidence from the ESP32-CAM over ESP-NOW, looks up
 *   the fan speed for that crop, drives the fan with PWM, and serves the
 *   current state to the FarmFrost mobile app over Wi-Fi.
 *
 * Wi-Fi instead of BLE
 *
 *   The V1 spec calls for BLE GATT here. The ESP8266 has no Bluetooth radio at
 *   all, so this board runs its own Wi-Fi access point instead and the phone
 *   joins it directly. Still fully offline -- no router, no internet, no
 *   backend. The app talks to http://192.168.4.1/status.
 *
 *   The spec also says to prefer push notifications over polling. That rule
 *   exists because BLE polling is genuinely expensive -- connection intervals
 *   and a GATT round trip per read. Over Wi-Fi on a two-device LAN, fetching a
 *   ~150 byte JSON once a second costs nothing, and it avoids putting a
 *   WebSocket server on a board with ~40 KB of free heap. If push is wanted
 *   later, arduinoWebSockets on port 81 is the drop-in.
 *
 * Autonomy
 *
 *   The fan is controlled entirely by this board. Nothing here waits for the
 *   phone, and nothing the phone does can change the fan. Disconnecting the
 *   app, or never connecting it, leaves fan control running untouched.
 *
 * Arduino IDE setup
 *
 *   Boards Manager URL : https://arduino.esp8266.com/stable/package_esp8266com_index.json
 *   Board              : NodeMCU 1.0 (ESP-12E Module)
 *   Flash Size         : 4MB (FS:2MB OTA:~1019KB)  -- the default is fine
 *   Upload Speed       : 115200
 *
 *   The NodeMCU has onboard USB, so it just needs the USB cable. No GPIO0
 *   jumper, unlike the camera board.
 */

#include <ESP8266WiFi.h>
#include <espnow.h>
#include <ESP8266WebServer.h>

// ==========================================================================
// Configuration
// ==========================================================================

// --- Access point the phone joins ----------------------------------------
static const char *AP_SSID     = "FarmFrost";
static const char *AP_PASSWORD = "farmfrost";   // min 8 chars, or "" for open

// Must match WIFI_CHANNEL in the ESP32-CAM sketch. ESP-NOW only talks between
// devices on the same channel, and for an access point the channel is whatever
// the AP was started on -- so this one setting drives both.
#define WIFI_CHANNEL         1

// --- Fan: L298N H-bridge --------------------------------------------------
// The motor is driven through an L298N, so three pins instead of one: ENA
// carries the PWM that sets the speed, IN1/IN2 set the direction. The spec's
// rule from section 7 still holds and is satisfied -- the GPIO drives the
// driver's logic input, never the motor itself.
//
//   NodeMCU          L298N
//   D5 (GPIO14) ---> ENA      speed (PWM). Remove the ENA jumper!
//   D6 (GPIO12) ---> IN1      direction
//   D7 (GPIO13) ---> IN2      direction
//   GND         ---> GND      must be common with the motor supply
//
// D5/D6/D7 are chosen because none of them has a role at boot. D3, D4 and D8
// are pulled or sampled by the bootloader and a driver board tied to them can
// stop the NodeMCU from starting at all.
#define FAN_ENA_PIN          D5
#define FAN_IN1_PIN          D6
#define FAN_IN2_PIN          D7

// The ESP8266 generates PWM in software, so high frequencies cost CPU time.
// 1 kHz is inaudible enough through a fan and leaves the radio alone. If the
// fan whines, 20000 is usually the fix, at some cost in timing jitter.
#define PWM_FREQ_HZ          1000
#define PWM_RANGE            255

// A motor that is stopped needs far more duty to break away than it needs to
// keep turning, and the L298N eats roughly 1.4-2 V of the supply on top of
// that. Without help, the lower entries in FAN_SPEED (ginger at 40%, onion at
// 35%) tend to sit there buzzing instead of turning.
//
// So: when starting from rest, drive 100% for KICKSTART_MS, then drop to the
// target. Only from rest -- changing between two non-zero speeds does not need
// it, and kicking every time would be audible.
#define KICKSTART_ENABLED    1
#define KICKSTART_MS         250
// Below this the motor is treated as stopped rather than driven at a duty it
// would only hum at. Raise it if your motor still buzzes without turning.
#define FAN_MIN_DUTY_PCT     20

// --- Reliability ----------------------------------------------------------
// Below this confidence the detection is ignored and the fan holds its
// previous speed. Mirrors CONFIDENCE_THRESHOLD in the model's config.py.
#define CONFIDENCE_THRESHOLD 60

// If nothing arrives from the camera for this long, report it as offline --
// but keep the fan exactly where it is. A dead camera must not stop the fan.
#define CAM_TIMEOUT_MS       15000

// What to do when the camera reports UNKNOWN with high confidence, i.e. it is
// confident there is no crop it recognises in front of it (empty tray, a hand,
// some other produce). Holding the previous speed is the conservative choice:
// a hand passing over the tray should not spin the fan down.
// Set to 0 to fall back to FAN_SPEED_DEFAULT instead.
#define UNKNOWN_KEEPS_PREVIOUS 1

// Speed used before the first valid detection ever arrives.
#define FAN_SPEED_DEFAULT    0

// ==========================================================================
// Crop table
//
// Index order comes from class_names() in "Crop detection model/config.py":
// crops alphabetically, then unknown last. The index IS the crop_id on the
// wire, so this must stay in step with the model and with the camera sketch.
// ==========================================================================

static const char *CROP_NAMES[] = {
  "APPLE", "BANANA", "CORN", "GINGER",
  "LEMON", "ONION", "POTATO", "TOMATO", "UNKNOWN"
};

// Fan speed per crop, in percent.
//
// PLACEHOLDER VALUES. Tomato/Potato/Onion/Banana are the examples given in
// section 6 of the spec; the rest are filler in the same spirit. Replace all
// of them with the real FarmFrost storage requirements before this means
// anything. UNKNOWN's entry is never used while UNKNOWN_KEEPS_PREVIOUS is 1.
static const uint8_t FAN_SPEED[] = {
   55,   // APPLE    placeholder
   60,   // BANANA   from spec
   50,   // CORN     placeholder
   40,   // GINGER   placeholder
   60,   // LEMON    placeholder
   35,   // ONION    from spec
   45,   // POTATO   from spec
   70,   // TOMATO   from spec
    0    // UNKNOWN  unused, see UNKNOWN_KEEPS_PREVIOUS
};

static const uint8_t CROP_COUNT   = sizeof(CROP_NAMES) / sizeof(CROP_NAMES[0]);
static const uint8_t CROP_UNKNOWN = CROP_COUNT - 1;

static_assert(sizeof(FAN_SPEED) / sizeof(FAN_SPEED[0]) == CROP_COUNT,
              "FAN_SPEED must have one entry per crop");

// ==========================================================================
// Packet -- must match the CropResult struct in the ESP32-CAM sketch byte for
// byte. If you change one, change both.
// ==========================================================================

typedef struct __attribute__((packed)) {
  uint8_t  crop_id;
  uint8_t  confidence;
  uint32_t seq;
} CropResult;

// ==========================================================================
// State
//
// g_rx* are written inside the ESP-NOW receive callback, which runs in the
// SDK's context, and read in loop(). The callback does nothing but copy and
// set a flag -- no serial, no PWM, no HTTP -- so the window is a few
// instructions wide. g_rxPending is volatile because loop() polls it.
// ==========================================================================

static volatile bool       g_rxPending = false;
static volatile CropResult g_rx;

static uint8_t  g_cropId      = CROP_UNKNOWN;   // last ACCEPTED detection
static uint8_t  g_confidence  = 0;
static uint8_t  g_fanSpeed    = FAN_SPEED_DEFAULT;
static uint32_t g_lastSeq     = 0;
static uint32_t g_lastRxMs    = 0;
static uint32_t g_packetCount = 0;
static uint32_t g_rejectCount = 0;
static bool     g_everReceived = false;

ESP8266WebServer server(80);

// ==========================================================================
// Fan
// ==========================================================================

static void setFanSpeed(uint8_t percent) {
  if (percent > 100) percent = 100;

  const bool wasStopped = (g_fanSpeed == 0);
  g_fanSpeed = percent;

  // Anything at or below the stall duty is off outright. Driving a motor at a
  // duty it can only hum at wastes current and cooks the driver for nothing.
  if (percent == 0 || percent < FAN_MIN_DUTY_PCT) {
    if (percent > 0) {
      Serial.printf("[MAIN] %u%% is below the %u%% stall floor, motor off\n",
                    percent, FAN_MIN_DUTY_PCT);
    }
    analogWrite(FAN_ENA_PIN, 0);
    digitalWrite(FAN_IN1_PIN, LOW);    // both low = coast
    digitalWrite(FAN_IN2_PIN, LOW);
    g_fanSpeed = 0;
    Serial.println("[MAIN] PWM updated: 0%");
    return;
  }

  // Direction. One way only -- a fan has no reason to reverse, and IN1/IN2
  // must never both be HIGH.
  digitalWrite(FAN_IN1_PIN, HIGH);
  digitalWrite(FAN_IN2_PIN, LOW);

#if KICKSTART_ENABLED
  if (wasStopped && percent < 100) {
    analogWrite(FAN_ENA_PIN, PWM_RANGE);
    Serial.printf("[MAIN] Kickstart 100%% for %u ms\n", KICKSTART_MS);
    delay(KICKSTART_MS);               // safe here: called from loop(), never
                                       // from the ESP-NOW receive callback
  }
#endif

  analogWrite(FAN_ENA_PIN, (percent * PWM_RANGE) / 100);

  Serial.printf("[MAIN] PWM updated: %u%%\n", percent);
}

// ==========================================================================
// ESP-NOW
// ==========================================================================

static void onDataRecv(uint8_t *mac, uint8_t *data, uint8_t len) {
  if (len != sizeof(CropResult)) return;   // not ours, or a version mismatch
  memcpy((void *)&g_rx, data, sizeof(CropResult));
  g_rxPending = true;
}

// Apply a received packet. Everything the spec's section 15 asks for happens
// here: an invalid crop id or a low-confidence reading leaves the fan exactly
// where it was.
static void handleDetection(const CropResult &r) {
  const bool firstEver = !g_everReceived;

  g_packetCount++;
  g_lastRxMs = millis();
  g_everReceived = true;

  Serial.printf("[MAIN] Crop received: %s\n",
                r.crop_id < CROP_COUNT ? CROP_NAMES[r.crop_id] : "INVALID");
  Serial.printf("[MAIN] Confidence: %u%%\n", r.confidence);

  if (r.crop_id >= CROP_COUNT) {
    Serial.printf("[MAIN] Invalid crop id %u, fan unchanged\n", r.crop_id);
    g_rejectCount++;
    return;
  }

  if (r.confidence < CONFIDENCE_THRESHOLD) {
    Serial.printf("[MAIN] Below %u%% threshold, fan unchanged\n",
                  CONFIDENCE_THRESHOLD);
    g_rejectCount++;
    return;
  }

  // Gaps in seq mean packets were lost on the way. Nothing to do about it --
  // the next detection is only 3 s away and carries the full state, so there
  // is no history to reconstruct. Worth logging because a steady stream of
  // gaps means the link needs attention.
  //
  // Skipped on the first packet: the camera may have been running for a while
  // before this board booted, so its seq starts wherever it happens to be and
  // the "gap" is meaningless.
  if (!firstEver && r.seq > g_lastSeq + 1) {
    Serial.printf("[MAIN] Missed %lu packet(s)\n",
                  (unsigned long)(r.seq - g_lastSeq - 1));
  }
  g_lastSeq = r.seq;

  if (r.crop_id == CROP_UNKNOWN) {
#if UNKNOWN_KEEPS_PREVIOUS
    Serial.println("[MAIN] No known crop in frame, fan unchanged");
    g_rejectCount++;
    return;
#else
    g_cropId = r.crop_id;
    g_confidence = r.confidence;
    setFanSpeed(FAN_SPEED_DEFAULT);
    return;
#endif
  }

  g_cropId     = r.crop_id;
  g_confidence = r.confidence;

  uint8_t target = FAN_SPEED[r.crop_id];
  Serial.printf("[MAIN] Fan target: %u%%\n", target);

  if (target != g_fanSpeed) {
    setFanSpeed(target);
  } else {
    Serial.println("[MAIN] Fan already at target");
  }
}

static bool camOnline() {
  return g_everReceived && (millis() - g_lastRxMs) < CAM_TIMEOUT_MS;
}

// ==========================================================================
// Wi-Fi interface for the app
// ==========================================================================

static const char PAGE_HTML[] PROGMEM = R"HTML(
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FarmFrost</title>
<style>
 body{margin:0;background:#101413;color:#e8efe9;
      font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
 .wrap{max-width:420px;margin:0 auto;padding:28px 20px}
 h1{font-size:16px;letter-spacing:.22em;margin:0 0 24px;color:#8fd6a8}
 .label{color:#7d8a82;font-size:11px;letter-spacing:.1em;text-transform:uppercase}
 .value{font-size:34px;font-weight:600;margin:2px 0 22px}
 .low{color:#e6a15c}
 .bar{height:6px;background:#26312c;border-radius:3px;overflow:hidden;
      margin:-14px 0 22px}
 .bar i{display:block;height:100%;background:#8fd6a8;transition:width .4s}
 .dot{display:inline-block;width:9px;height:9px;border-radius:50%;
      background:#4b5a52;margin-right:7px;vertical-align:middle}
 .on{background:#8fd6a8}
 .foot{color:#5d6a63;font-size:12px;border-top:1px solid #26312c;
       padding-top:14px;margin-top:8px}
</style></head><body><div class="wrap">
<h1>FARMFROST</h1>
<div class="label">Detected crop</div><div class="value" id="crop">&mdash;</div>
<div class="label">Confidence</div><div class="value" id="conf">&mdash;</div>
<div class="label">Fan speed</div><div class="value" id="fan">&mdash;</div>
<div class="bar"><i id="fanbar" style="width:0"></i></div>
<div class="label">Camera</div>
<div class="value" style="font-size:18px">
  <span class="dot" id="dot"></span><span id="link">&mdash;</span></div>
<p class="foot" id="foot"></p>
</div><script>
async function tick(){
  try{
    const s = await (await fetch('/status')).json();
    document.getElementById('crop').textContent = s.crop;
    const c = document.getElementById('conf');
    c.textContent = s.confidence + '%';
    c.className = 'value' + (s.confidence < s.threshold ? ' low' : '');
    document.getElementById('fan').textContent = s.fan_speed + '%';
    document.getElementById('fanbar').style.width = s.fan_speed + '%';
    document.getElementById('dot').className = 'dot' + (s.cam_online ? ' on' : '');
    document.getElementById('link').textContent =
      s.cam_online ? 'Connected' : 'No signal';
    document.getElementById('foot').textContent =
      s.packets + ' packets · ' + s.ignored + ' ignored · up ' +
      Math.floor(s.uptime_s/60) + 'm';
  }catch(e){
    document.getElementById('link').textContent = 'No signal';
    document.getElementById('dot').className = 'dot';
  }
}
tick();
setInterval(tick, 1000);
</script></body></html>
)HTML";

static void handleRoot() {
  server.send_P(200, "text/html", PAGE_HTML);
}

// This is the endpoint the mobile app reads. Everything the app needs to draw
// its dashboard is in one response, so one request per second is enough.
static void handleStatus() {
  char json[288];
  snprintf(json, sizeof(json),
           "{\"crop\":\"%s\",\"crop_id\":%u,\"confidence\":%u,\"fan_speed\":%u,"
           "\"cam_online\":%s,\"threshold\":%u,\"seq\":%lu,\"packets\":%lu,"
           "\"ignored\":%lu,\"uptime_s\":%lu}",
           CROP_NAMES[g_cropId], g_cropId, g_confidence, g_fanSpeed,
           camOnline() ? "true" : "false", CONFIDENCE_THRESHOLD,
           (unsigned long)g_lastSeq, (unsigned long)g_packetCount,
           (unsigned long)g_rejectCount, (unsigned long)(millis() / 1000));

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "application/json", json);
}

// ==========================================================================
// Setup / loop
// ==========================================================================

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println();
  Serial.println("[MAIN] FarmFrost main controller starting");

  pinMode(FAN_ENA_PIN, OUTPUT);
  pinMode(FAN_IN1_PIN, OUTPUT);
  pinMode(FAN_IN2_PIN, OUTPUT);
  // Park the H-bridge before PWM is configured, so the motor cannot twitch
  // while the pins settle.
  digitalWrite(FAN_IN1_PIN, LOW);
  digitalWrite(FAN_IN2_PIN, LOW);
  analogWriteRange(PWM_RANGE);
  analogWriteFreq(PWM_FREQ_HZ);
  setFanSpeed(FAN_SPEED_DEFAULT);

  // AP mode, started before ESP-NOW so the radio is already parked on
  // WIFI_CHANNEL when ESP-NOW comes up.
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASSWORD, WIFI_CHANNEL);

  Serial.println();
  Serial.println("[MAIN] ================================================");
  Serial.printf ("[MAIN] Paste this into the ESP32-CAM sketch as MAIN_ESP_MAC:\n");
  Serial.printf ("[MAIN]   %s   <- SoftAP MAC, this is the one to use\n",
                 WiFi.softAPmacAddress().c_str());
  Serial.printf ("[MAIN]   %s   (station MAC, NOT the one to use)\n",
                 WiFi.macAddress().c_str());
  Serial.println("[MAIN] ================================================");
  Serial.println();
  Serial.printf ("[MAIN] Phone: join Wi-Fi \"%s\"", AP_SSID);
  if (strlen(AP_PASSWORD)) Serial.printf(" (password: %s)", AP_PASSWORD);
  Serial.println();
  Serial.print  ("[MAIN] Then open http://");
  Serial.println(WiFi.softAPIP());
  Serial.printf ("[MAIN] App endpoint: http://%s/status\n",
                 WiFi.softAPIP().toString().c_str());
  Serial.println();

  if (esp_now_init() != 0) {
    Serial.println("[MAIN] ESP-NOW init failed, restarting");
    delay(1000);
    ESP.restart();
  }
  // This board only ever receives crop results; it never sends any.
  esp_now_set_self_role(ESP_NOW_ROLE_SLAVE);
  esp_now_register_recv_cb(onDataRecv);
  Serial.printf("[MAIN] ESP-NOW listening on channel %u\n", WIFI_CHANNEL);

  server.on("/", handleRoot);
  server.on("/status", handleStatus);
  server.begin();
  Serial.println("[MAIN] HTTP server ready");
}

void loop() {
  server.handleClient();

  if (g_rxPending) {
    // Copy out first, then clear the flag. The ESP8266 SDK runs the ESP-NOW
    // callback from its own task rather than from an interrupt, and task
    // switches only happen at yield points -- a 6-byte memcpy contains none,
    // so it cannot be interleaved with the callback's write. Clearing the flag
    // after the copy means a packet arriving in between is simply dropped,
    // which is the same outcome as it being lost on the air.
    CropResult r;
    memcpy(&r, (const void *)&g_rx, sizeof(r));
    g_rxPending = false;

    handleDetection(r);
  }

  // Camera going quiet is worth reporting once, but must not touch the fan.
  static bool wasOnline = false;
  bool online = camOnline();
  if (wasOnline && !online) {
    Serial.printf("[MAIN] Camera silent for %u s, holding fan at %u%%\n",
                  CAM_TIMEOUT_MS / 1000, g_fanSpeed);
  }
  wasOnline = online;
}
