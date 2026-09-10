/*
 * FarmFrost V1 -- ESP32-CAM firmware
 * ---------------------------------
 * Board : AI-Thinker ESP32-CAM (chip marked ESP-32S), OV2640
 *
 * What this sketch does today
 *
 *   Every DETECT_INTERVAL_MS it captures a JPEG, runs crop detection on it,
 *   and keeps that frame plus its result as "the current state". The frame is
 *   the one that was classified -- not a newer one -- so the picture you see
 *   on the desktop always matches the crop and confidence printed next to it.
 *
 *   The state is exposed three ways:
 *     1. Serial monitor  -- log lines, always on.
 *     2. Desktop browser -- the board runs its own Wi-Fi access point and a
 *                           small web page. No router, no internet. This is
 *                           the "show it on the desktop" testing path.
 *     3. ESP-NOW         -- crop_id + confidence to the main board. Off until
 *                           you fill in the main board's MAC (see below).
 *
 * What is still a placeholder
 *
 *   runInference() returns rotating fake results. The TFLite Micro model from
 *   "Crop detection model" is not linked in yet -- that is Step 4 of the spec.
 *   Everything the fake result touches (logging, web page, ESP-NOW packet) is
 *   real, so swapping the model in later is a one-function change.
 *   The web page labels itself PLACEHOLDER so a fake reading is never mistaken
 *   for a real one.
 *
 * Flashing from the Arduino IDE
 *
 *   Boards Manager  : esp32 by Espressif Systems
 *   Board           : "AI Thinker ESP32-CAM"
 *   Partition Scheme: "Huge APP (3MB No OTA/1MB SPIFFS)"   <- needed later for
 *                     the model; set it now so nothing changes at Step 4.
 *   PSRAM           : Enabled
 *   Upload Speed    : 115200 if 921600 fails
 *
 *   Wiring to the USB-TTL adapter (the CAM has no USB port):
 *     5V  -> 5V     (do NOT power the camera from a 3.3V pin, it browns out)
 *     GND -> GND
 *     U0T -> RX
 *     U0R -> TX
 *     GPIO0 -> GND  ONLY while uploading. Remove the jumper and press RESET
 *                   once "Hard resetting" appears, or the sketch will not run.
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <esp_now.h>
#include <esp_wifi.h>

// The AI-Thinker board's 3.3V regulator dips when the camera and the radio
// draw at the same time, and the brownout detector reboots the board mid-boot.
// Disabling it is the standard fix; a decent 5V supply is the real fix.
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// ==========================================================================
// Configuration
// ==========================================================================

// --- Desktop viewer (Wi-Fi access point) ---------------------------------
// The board creates its own network. Connect the laptop to it and open
// http://192.168.4.1 -- nothing else on the network, no router involved.
#define ENABLE_WEB_VIEWER   1
static const char *AP_SSID     = "FarmFrost-CAM";
static const char *AP_PASSWORD = "farmfrost";   // min 8 chars, or "" for open

// --- ESP-NOW to the main board -------------------------------------------
// Leave at 0 until you know the main board's MAC.
//
// The main board (NodeMCU) runs in access-point mode so the phone can join it,
// which means ESP-NOW has to be addressed to its **SoftAP MAC**, not its
// station MAC. Those are two different addresses on the same chip -- sending
// to the wrong one fails silently, with the send callback reporting success.
// Main_ESP8266.ino prints both at boot and labels which is which.
#define ENABLE_ESPNOW       0
static uint8_t MAIN_ESP_MAC[6] = { 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF };

// ESP-NOW and the access point must sit on the same Wi-Fi channel, so this
// value is used for both. The main board has to listen on it too.
#define WIFI_CHANNEL        1

// --- Detection cycle ------------------------------------------------------
#define DETECT_INTERVAL_MS  3000

// Below this the main board keeps its previous fan state. Mirrors
// CONFIDENCE_THRESHOLD = 0.60 in "Crop detection model/config.py".
#define CONFIDENCE_THRESHOLD 60

// --- Board pins -----------------------------------------------------------
#define FLASH_LED_PIN        4    // the bright white LED
#define STATUS_LED_PIN      33    // the small red LED, active LOW

// ==========================================================================
// Crop table
//
// The order here is the model's output order, defined by class_names() in
// "Crop detection model/config.py": crops alphabetically, then unknown last.
// The integer index IS the crop_id sent over ESP-NOW, so this order must not
// be rearranged without retraining.
// ==========================================================================

static const char *CROP_NAMES[] = {
  "APPLE",    // 0
  "BANANA",   // 1
  "CORN",     // 2
  "GINGER",   // 3
  "LEMON",    // 4
  "ONION",    // 5
  "POTATO",   // 6
  "TOMATO",   // 7
  "UNKNOWN"   // 8
};
static const uint8_t CROP_COUNT   = sizeof(CROP_NAMES) / sizeof(CROP_NAMES[0]);
static const uint8_t CROP_UNKNOWN = CROP_COUNT - 1;

// ==========================================================================
// Camera pinout -- AI-Thinker ESP32-CAM
// ==========================================================================

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ==========================================================================
// The packet sent to the main board
//
// Only the result travels, never the image -- 8 bytes against ~30 KB. ESP-NOW
// caps a payload at 250 bytes, so the image could not be sent this way even if
// we wanted to, and the main board has nothing useful to do with pixels.
// ==========================================================================

typedef struct __attribute__((packed)) {
  uint8_t  crop_id;     // index into CROP_NAMES
  uint8_t  confidence;  // 0-100
  uint32_t seq;         // increments per detection; lets the receiver spot drops
} CropResult;

// ==========================================================================
// Current state -- the classified frame and its result
//
// loop() is the only task that touches these: the capture cycle and the web
// server both run in it, one after the other, so there is no concurrent access
// to guard against. Keep it that way, or add a mutex.
// ==========================================================================

static uint8_t *g_jpeg       = nullptr;   // copy of the classified frame
static size_t   g_jpegLen    = 0;
static uint8_t  g_cropId     = CROP_UNKNOWN;
static uint8_t  g_confidence = 0;
static uint32_t g_seq        = 0;
static uint32_t g_lastDetectMs = 0;

#if ENABLE_WEB_VIEWER
WebServer server(80);
#endif

// ==========================================================================
// Camera
// ==========================================================================

static bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode    = CAMERA_GRAB_LATEST;

  // With PSRAM we can afford a bigger frame and two buffers. Without it the
  // board still works, just at a smaller size -- worth knowing, because a
  // module with no PSRAM will not be able to run the model later either.
  if (psramFound()) {
    config.frame_size   = FRAMESIZE_SVGA;   // 800x600
    config.jpeg_quality = 12;               // lower number = better quality
    config.fb_count     = 2;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
  } else {
    config.frame_size   = FRAMESIZE_VGA;    // 640x480
    config.jpeg_quality = 15;
    config.fb_count     = 1;
    config.fb_location  = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAM] Camera init failed: 0x%x\n", err);
    return false;
  }

  // The OV2640 on these boards ships mirrored and a little flat.
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_hmirror(s, 1);
    s->set_vflip(s, 0);
    s->set_brightness(s, 1);
    s->set_saturation(s, 0);
  }
  return true;
}

// Take a frame and keep a copy of it as the current image.
//
// The copy matters: esp_camera_fb_return() hands the buffer straight back to
// the driver, which will overwrite it on the next capture. Serving the frame
// buffer directly would mean the web page could receive a half-overwritten
// image. So we copy into PSRAM, then release immediately.
static bool captureFrame() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[CAM] Capture failed");
    return false;
  }

  uint8_t *copy = (uint8_t *)(psramFound() ? ps_malloc(fb->len) : malloc(fb->len));
  if (!copy) {
    // Keep the previous frame rather than dropping to no image at all.
    Serial.printf("[CAM] Out of memory for a %u byte frame\n", (unsigned)fb->len);
    esp_camera_fb_return(fb);
    return false;
  }

  memcpy(copy, fb->buf, fb->len);
  size_t copyLen = fb->len;
  esp_camera_fb_return(fb);

  if (g_jpeg) free(g_jpeg);
  g_jpeg    = copy;
  g_jpegLen = copyLen;
  return true;
}

// ==========================================================================
// Inference -- PLACEHOLDER
//
// Replace the body with the real thing at Step 4 of the spec. The real version
// has to do what "Crop detection model/cropnet/preprocess.py" does on the
// desktop, or the model sees a different picture than it was trained on:
//
//   1. Decode the JPEG to RGB     -- jpg2rgb565() / fmt2rgb888() from
//                                    img_converters.h, or capture a second
//                                    frame in PIXFORMAT_RGB565 instead.
//   2. Centre-crop to a square.
//   3. Downscale to 96x96 (config.IMAGE_SIZE).
//   4. Quantize to int8 and run the model with esp-tflite-micro.
//   5. Return argmax and its softmax value as 0-100.
//
// Until then this rotates through the crop table so the whole pipeline --
// logging, web page, ESP-NOW -- can be tested end to end.
// ==========================================================================

static void runInference(const uint8_t *jpeg, size_t len,
                         uint8_t *cropId, uint8_t *confidence) {
  (void)jpeg;
  (void)len;

  static uint8_t next = 0;
  *cropId = next;
  next = (next + 1) % CROP_COUNT;

  // Unknown deliberately comes back below the threshold, so the "do not act on
  // a low-confidence reading" path on the main board gets exercised too.
  *confidence = (*cropId == CROP_UNKNOWN) ? 41 : 92;
}

// ==========================================================================
// ESP-NOW
// ==========================================================================

#if ENABLE_ESPNOW
// The send callback's first argument changed shape in ESP32 Arduino core 3.x
// (it now carries the whole tx_info instead of just the MAC). We only use the
// status, so both spellings do the same thing -- this just keeps the sketch
// compiling on whichever core version is installed.
#if ESP_ARDUINO_VERSION_MAJOR >= 3
static void onEspNowSent(const wifi_tx_info_t *info, esp_now_send_status_t status) {
  (void)info;
#else
static void onEspNowSent(const uint8_t *mac, esp_now_send_status_t status) {
  (void)mac;
#endif
  Serial.printf("[CAM] ESP-NOW packet %s\n",
                status == ESP_NOW_SEND_SUCCESS ? "delivered" : "NOT delivered");
}

static bool initEspNow() {
  if (esp_now_init() != ESP_OK) {
    Serial.println("[CAM] ESP-NOW init failed");
    return false;
  }
  esp_now_register_send_cb(onEspNowSent);

  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, MAIN_ESP_MAC, 6);
  peer.channel = WIFI_CHANNEL;   // must match the receiver's channel
  peer.encrypt = false;          // an ESP8266 receiver needs encryption off

  if (esp_now_add_peer(&peer) != ESP_OK) {
    Serial.println("[CAM] Failed to add the main board as a peer");
    return false;
  }
  Serial.println("[CAM] ESP-NOW ready");
  return true;
}

static void sendResult() {
  CropResult packet;
  packet.crop_id    = g_cropId;
  packet.confidence = g_confidence;
  packet.seq        = g_seq;
  esp_now_send(MAIN_ESP_MAC, (uint8_t *)&packet, sizeof(packet));
}
#endif  // ENABLE_ESPNOW

// ==========================================================================
// Desktop viewer
// ==========================================================================

#if ENABLE_WEB_VIEWER

static const char PAGE_HTML[] PROGMEM = R"HTML(
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FarmFrost CAM</title>
<style>
 body{margin:0;background:#101413;color:#e8efe9;
      font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
 .wrap{max-width:840px;margin:0 auto;padding:24px}
 h1{font-size:18px;letter-spacing:.14em;margin:0 0 4px;color:#8fd6a8}
 .sub{color:#7d8a82;margin:0 0 20px}
 .warn{background:#3a2f12;border:1px solid #6b5518;color:#f0d68a;
       padding:8px 12px;border-radius:6px;margin-bottom:16px}
 img{width:100%;border-radius:8px;background:#000;display:block}
 .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px}
 .card{background:#18201d;border:1px solid #26312c;border-radius:8px;padding:14px}
 .label{color:#7d8a82;font-size:11px;letter-spacing:.1em;text-transform:uppercase}
 .value{font-size:24px;font-weight:600;margin-top:4px}
 .low{color:#e6a15c}
 .foot{color:#5d6a63;font-size:12px;margin-top:16px}
</style></head><body><div class="wrap">
<h1>FARMFROST &mdash; ESP32-CAM</h1>
<p class="sub">Live capture from the board. No router, no internet.</p>
<div class="warn">PLACEHOLDER &mdash; the crop model is not flashed yet.
 The image is real; the crop and confidence below are fake test values.</div>
<img id="shot" src="/capture" alt="latest capture">
<div class="grid">
  <div class="card"><div class="label">Detected crop</div>
    <div class="value" id="crop">&mdash;</div></div>
  <div class="card"><div class="label">Confidence</div>
    <div class="value" id="conf">&mdash;</div></div>
  <div class="card"><div class="label">Frame</div>
    <div class="value" id="seq">&mdash;</div></div>
</div>
<p class="foot" id="foot"></p>
</div><script>
async function tick(){
  try{
    const s = await (await fetch('/status')).json();
    document.getElementById('crop').textContent = s.crop;
    const c = document.getElementById('conf');
    c.textContent = s.confidence + '%';
    c.className = 'value' + (s.confidence < s.threshold ? ' low' : '');
    document.getElementById('seq').textContent = '#' + s.seq;
    document.getElementById('foot').textContent =
      'frame ' + (s.image_bytes/1024).toFixed(1) + ' KB · new capture every ' +
      (s.interval_ms/1000) + ' s · fan acts at ' + s.threshold + '% and above';
    document.getElementById('shot').src = '/capture?s=' + s.seq;
  }catch(e){}
}
tick();
setInterval(tick, 1500);
</script></body></html>
)HTML";

static void handleRoot() {
  server.send_P(200, "text/html", PAGE_HTML);
}

// Serves the frame that was classified, so image and result always agree.
static void handleCapture() {
  if (!g_jpeg || g_jpegLen == 0) {
    server.send(503, "text/plain", "No frame captured yet");
    return;
  }
  server.sendHeader("Cache-Control", "no-store");
  server.setContentLength(g_jpegLen);
  server.send(200, "image/jpeg", "");
  server.sendContent((const char *)g_jpeg, g_jpegLen);
}

static void handleStatus() {
  char json[256];
  snprintf(json, sizeof(json),
           "{\"crop_id\":%u,\"crop\":\"%s\",\"confidence\":%u,\"seq\":%lu,"
           "\"image_bytes\":%u,\"interval_ms\":%u,\"threshold\":%u,"
           "\"placeholder\":true}",
           g_cropId, CROP_NAMES[g_cropId], g_confidence,
           (unsigned long)g_seq, (unsigned)g_jpegLen,
           DETECT_INTERVAL_MS, CONFIDENCE_THRESHOLD);
  server.send(200, "application/json", json);
}

static void initWebViewer() {
  WiFi.softAP(AP_SSID, AP_PASSWORD, WIFI_CHANNEL);

  server.on("/", handleRoot);
  server.on("/capture", handleCapture);
  server.on("/status", handleStatus);
  server.begin();

  Serial.println();
  Serial.println("[CAM] Desktop viewer ready");
  Serial.printf("[CAM]   1. Connect this computer to Wi-Fi \"%s\"", AP_SSID);
  if (strlen(AP_PASSWORD)) Serial.printf(" (password: %s)", AP_PASSWORD);
  Serial.println();
  Serial.print("[CAM]   2. Open http://");
  Serial.println(WiFi.softAPIP());
  Serial.println();
}
#endif  // ENABLE_WEB_VIEWER

// ==========================================================================
// Setup / loop
// ==========================================================================

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);   // see the include above

  Serial.begin(115200);
  delay(300);
  Serial.println();
  Serial.println("[CAM] FarmFrost ESP32-CAM starting");

  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);            // flash off
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, HIGH);          // active LOW, so HIGH is off

  Serial.printf("[CAM] PSRAM: %s\n", psramFound() ? "found" : "NOT found");

  if (!initCamera()) {
    // Nothing this sketch does is useful without a camera, so make the failure
    // impossible to miss instead of quietly looping.
    Serial.println("[CAM] Halted. Check the ribbon cable and the 5V supply.");
    while (true) {
      digitalWrite(STATUS_LED_PIN, LOW);  delay(150);
      digitalWrite(STATUS_LED_PIN, HIGH); delay(150);
    }
  }
  Serial.println("[CAM] Camera ready");

  // Wi-Fi has to be up before ESP-NOW starts. AP mode both serves the desktop
  // viewer and pins the radio to WIFI_CHANNEL, which is what ESP-NOW needs.
#if ENABLE_WEB_VIEWER
  WiFi.mode(WIFI_AP);
  initWebViewer();
#elif ENABLE_ESPNOW
  // No access point to pin the radio, so the channel has to be set by hand.
  // A station that is not associated to anything sits on channel 1 by default
  // but is free to move; ESP-NOW would then quietly stop working whenever the
  // two boards drifted apart.
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
#endif

#if ENABLE_ESPNOW
  Serial.print("[CAM] This board's MAC: ");
  Serial.println(WiFi.macAddress());
  initEspNow();
#endif
}

void loop() {
#if ENABLE_WEB_VIEWER
  server.handleClient();
#endif

  uint32_t now = millis();
  if (now - g_lastDetectMs < DETECT_INTERVAL_MS) return;
  g_lastDetectMs = now;

  if (!captureFrame()) return;

  digitalWrite(STATUS_LED_PIN, LOW);   // on while classifying
  runInference(g_jpeg, g_jpegLen, &g_cropId, &g_confidence);
  digitalWrite(STATUS_LED_PIN, HIGH);
  g_seq++;

  Serial.printf("[CAM] Image captured (%u bytes)\n", (unsigned)g_jpegLen);
  Serial.printf("[CAM] Crop: %s (placeholder)\n", CROP_NAMES[g_cropId]);
  Serial.printf("[CAM] Confidence: %u%%\n", g_confidence);

#if ENABLE_ESPNOW
  if (g_confidence >= CONFIDENCE_THRESHOLD) {
    sendResult();
  } else {
    // Sending it anyway would be fine -- the main board checks the threshold
    // too -- but not sending keeps the radio quiet on frames that cannot
    // change anything.
    Serial.printf("[CAM] Below %u%% threshold, not sent\n", CONFIDENCE_THRESHOLD);
  }
#endif
}
