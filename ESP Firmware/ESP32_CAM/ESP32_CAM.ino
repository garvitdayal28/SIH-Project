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
 *   The white LED lights for the capture itself and goes out again before
 *   inference starts, so every classified frame is lit the same way. Set
 *   FLASH_ENABLED to 0 if the subject is already well lit.
 *
 *   The state is exposed three ways:
 *     1. Serial monitor  -- log lines, always on.
 *     2. Desktop browser -- the board runs its own Wi-Fi access point and a
 *                           small web page. No router, no internet. This is
 *                           the "show it on the desktop" testing path.
 *     3. ESP-NOW         -- crop_id + confidence to the main board. Off until
 *                           you fill in the main board's MAC (see below).
 *
 * The model
 *
 *   Real inference, running on the board. The MobileNetV1 alpha=0.5 96x96
 *   int8 model trained in "Crop detection model" is compiled into the binary
 *   as model_data.cpp and run by TensorFlow Lite Micro. All of that lives in
 *   crop_model.cpp; this file only asks it for a crop and a confidence.
 *
 *   Two things it needs that a plain sketch does not: a TFLM library (see
 *   README section 2) and a ~176 KB tensor arena, which crop_model.cpp
 *   puts in internal SRAM when it fits and PSRAM otherwise.
 *
 * Flashing from the Arduino IDE
 *
 *   Boards Manager  : esp32 by Espressif Systems
 *   Board           : "AI Thinker ESP32-CAM"
 *   Partition Scheme: "Huge APP (3MB No OTA/1MB SPIFFS)"   <- required. The
 *                     model alone is ~970 KB; the default 1.2 MB app partition
 *                     will not hold it and the build fails at link time.
 *   PSRAM           : Enabled   <- required, the arena will not fit without it
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

// Nothing in this file uses TFLite directly -- crop_model.cpp does. The include
// has to be here anyway, because of how Arduino finds libraries: it compiles,
// looks for an include it could not satisfy, adds the library that provides it,
// and repeats. crop_model.cpp guards its include with __has_include, which by
// design never fails, so the resolver would never notice the library existed
// and the build would stop at "No TensorFlow Lite Micro library found".
//
// Install it from the Library Manager as "tflm_esp32". If you switch to a
// different TFLM port, this is the line to change -- crop_model.cpp already
// adapts to the two common API shapes on its own.
#include <tflm_esp32.h>

#include "crop_model.h"

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
#define ENABLE_ESPNOW       1

// The NodeMCU's SoftAP MAC. Taken from the BSSID its "FarmFrost" access point
// broadcasts, which for an ESP8266 softAP is the same address -- note the 8e
// against the station side's 8c, the AP-side offset described above.
//
// If you swap to a different NodeMCU, this must change. Read the new one from
// that board's boot log, or scan for its AP:
//   netsh wlan show networks mode=bssid        (Windows)
static uint8_t MAIN_ESP_MAC[6] = { 0x8E, 0xAA, 0xB5, 0x4F, 0xE4, 0x66 };

// ESP-NOW and the access point must sit on the same Wi-Fi channel, so this
// value is used for both. The main board has to listen on it too.
#define WIFI_CHANNEL        1

// --- Detection cycle ------------------------------------------------------
// Inference is not free -- expect roughly 1.5-4 s per frame on a plain ESP32
// (no vector unit, and the arena lives in the slower PSRAM). loop() is
// single-threaded, so the web server stops responding for that whole window
// and the page will occasionally skip a poll. That is expected on this chip.
//
// The real per-frame cost is printed every cycle as "prep Nms, infer Nms".
//
// This is idle time BETWEEN cycles, and the web server only gets served during
// it -- loop() is single-threaded, so while inference runs, every request to
// the page is queued or dropped.
//
// 0 therefore does not mean "fastest useful". It means the server never gets a
// gap at all and the viewer starves: the picture stops updating even though
// detection is running fine. 1000 ms leaves room for roughly one status poll
// plus one image fetch per cycle, which is what makes the page look alive.
//
// Raise it to be kinder to the flash LED and the 5 V rail; lower it only if
// you do not care about the viewer.
#define DETECT_INTERVAL_MS  1000

// Below this the main board keeps its previous fan state. Mirrors
// CONFIDENCE_THRESHOLD = 0.60 in "Crop detection model/config.py".
#define CONFIDENCE_THRESHOLD 60

// --- Board pins -----------------------------------------------------------
#define FLASH_LED_PIN        4    // the bright white LED
#define STATUS_LED_PIN      33    // the small red LED, active LOW

// --- Flash while classifying ---------------------------------------------
// The model was trained on well-lit photographs, so a dim frame is not a
// neutral loss of quality -- it desaturates exactly the brown/tan colour that
// separates a yellow lemon from a red tomato. Lighting the subject is
// the cheapest accuracy the board has.
//
// The LED is driven with digitalWrite rather than PWM on purpose. Brightness
// control would mean LEDC, and the camera driver already holds an LEDC timer
// for the 20 MHz XCLK -- it claims that timer down in the IDF, where the
// Arduino LEDC allocator cannot see it, so ledcAttach() is free to hand out
// the same timer and reconfigure it to a few kHz. That kills the pixel clock:
// the light works and the camera stops. Full brightness for a fraction of a
// second is the safe trade.
#define FLASH_ENABLED         1
// Time for the sensor's auto-exposure to adapt after the light comes on.
// Grabbing immediately gives a frame still exposed for the dark scene, which
// comes out blown out and is worse than no flash at all.
#define FLASH_SETTLE_MS     250
// The driver fills its buffers continuously, so the frames already queued when
// the LED lit are pre-flash. Drop them and take the next one.
#define FLASH_DISCARD_FRAMES  2

// ==========================================================================
// Crop table
//
// Comes from model_data.h, which export_c_array.py generates from the same
// labels.txt the model was trained against -- CROP_NAMES, CROP_COUNT and
// CROP_EMPTY below are all defined there.
//
// This used to be a hand-written list. It is generated now because it cannot
// be allowed to drift: the index IS the crop_id on the wire, so a table that
// disagreed with the model by one position would make the main board run the
// wrong fan profile for every crop, with nothing anywhere reporting an error.
//
// The main board has its own copy of the order in its fan-speed table. That
// one is still manual, so if the classes ever change, change it too.
// ==========================================================================

#include "model_data.h"

#define CROP_NAMES g_crop_labels

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
static uint16_t g_jpegW      = 0;         // its dimensions, needed to decode it
static uint16_t g_jpegH      = 0;
static uint8_t  g_cropId     = CROP_EMPTY;
static uint8_t  g_confidence = 0;
static uint32_t g_seq        = 0;         // detections, i.e. successful inferences
// Captures. Counted separately from g_seq because the page uses this as the
// cache-buster on the image: keying it to g_seq meant that whenever inference
// failed -- which is every single frame if the model did not load -- the URL
// never changed, the browser served the first photo from cache forever, and
// the viewer looked frozen while the serial log showed captures ticking by.
static uint32_t g_capSeq     = 0;
static uint32_t g_lastDetectMs = 0;
static bool     g_modelOk    = false;     // cropModelInit() succeeded

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

  // QVGA rather than something larger, because every captured frame now gets
  // decoded to RGB888 for the model. That buffer is width*height*3, so SVGA
  // would cost 1.4 MB of PSRAM and a much slower decode, all to feed a 96x96
  // input. QVGA centre-crops to 240x240 -- a 2.5x reduction to 96x96, which
  // is a comfortable ratio for the area-average resize in crop_model.cpp.
  //
  // It is also the picture the desktop viewer shows. 320x240 is small but
  // perfectly legible for checking what the model is looking at. Raising it
  // costs inference latency, not accuracy: the model sees 96x96 either way.
  if (psramFound()) {
    config.frame_size   = FRAMESIZE_QVGA;   // 320x240
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
#if FLASH_ENABLED
  digitalWrite(FLASH_LED_PIN, HIGH);
  delay(FLASH_SETTLE_MS);
  for (int i = 0; i < FLASH_DISCARD_FRAMES; i++) {
    camera_fb_t *stale = esp_camera_fb_get();
    if (stale) esp_camera_fb_return(stale);
  }
#endif

  camera_fb_t *fb = esp_camera_fb_get();

#if FLASH_ENABLED
  // Off the moment the frame is in hand. The LED runs hot and draws hard
  // enough to matter on a board whose brownout detector is already disabled,
  // so it stays lit for the capture and nothing else -- in particular not for
  // the seconds of inference that follow.
  digitalWrite(FLASH_LED_PIN, LOW);
#endif

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
  size_t   copyLen = fb->len;
  uint16_t copyW   = fb->width;
  uint16_t copyH   = fb->height;
  esp_camera_fb_return(fb);

  if (g_jpeg) free(g_jpeg);
  g_jpeg    = copy;
  g_jpegLen = copyLen;
  g_jpegW   = copyW;
  g_jpegH   = copyH;
  g_capSeq++;
  return true;
}

// ==========================================================================
// Inference
//
// The work is all in crop_model.cpp -- decode, centre-crop, resize to 96x96,
// quantize, invoke. This is just the call plus the failure policy.
// ==========================================================================

static bool runInference(const uint8_t *jpeg, size_t len,
                         uint16_t width, uint16_t height,
                         uint8_t *cropId, uint8_t *confidence) {
  if (!g_modelOk) return false;

  // On failure the previous crop and confidence stay put. A dropped frame
  // should look identical to no new frame having arrived yet -- the main board
  // already holds the last valid state, and inventing an UNKNOWN here would
  // throw that away over what is usually a transient decode error.
  return cropModelClassify(jpeg, len, width, height, cropId, confidence);
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
  static const uint8_t placeholder[6] = { 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF };
  if (memcmp(MAIN_ESP_MAC, placeholder, 6) == 0) {
    Serial.println("[CAM] ***********************************************");
    Serial.println("[CAM] MAIN_ESP_MAC is still the placeholder.");
    Serial.println("[CAM] Flash Main_ESP8266.ino, copy the SoftAP MAC it");
    Serial.println("[CAM] prints at boot into MAIN_ESP_MAC, and reflash.");
    Serial.println("[CAM] Detection still works; the fan will never move.");
    Serial.println("[CAM] ***********************************************");
  }

  if (esp_now_init() != ESP_OK) {
    Serial.println("[CAM] ESP-NOW init failed");
    return false;
  }
  esp_now_register_send_cb(onEspNowSent);

  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, MAIN_ESP_MAC, 6);
  peer.channel = WIFI_CHANNEL;   // must match the receiver's channel
  peer.encrypt = false;          // an ESP8266 receiver needs encryption off

  // Which radio interface the packet leaves by. Zero-initialising the struct
  // sets this to WIFI_IF_STA, and with ENABLE_WEB_VIEWER the sketch runs
  // WiFi.mode(WIFI_AP) -- so the station interface is never started and every
  // send fails with ESP_ERR_ESPNOW_IF. The symptom is a board that classifies
  // perfectly and never delivers anything, with "This board's MAC" printing
  // as 00:00:00:00:00:00 because the STA interface has no address.
#if ENABLE_WEB_VIEWER
  peer.ifidx = WIFI_IF_AP;
#else
  peer.ifidx = WIFI_IF_STA;
#endif

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
  esp_err_t err = esp_now_send(MAIN_ESP_MAC, (uint8_t *)&packet, sizeof(packet));
  if (err != ESP_OK) {
    // Distinct from the send *callback*, which reports whether the frame was
    // acknowledged. This is the local call failing outright -- wrong interface,
    // peer not added, ESP-NOW not started.
    Serial.printf("[CAM] esp_now_send failed locally: %s\n", esp_err_to_name(err));
  }
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
<div class="warn" id="warn" hidden>The model failed to load &mdash; check the
 serial monitor for a [CV] line. Nothing below is a real detection.</div>
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
<p class="foot" id="diag"></p>
<p class="foot">Mean RGB is the average colour of the 96&times;96 tensor the
 model actually saw. Point the camera at something strongly red: R should be
 clearly above B. If they are swapped, flip CROP_SWAP_RB in crop_model.h.</p>
</div><script>
async function tick(){
  try{
    const s = await (await fetch('/status')).json();
    document.getElementById('crop').textContent = s.crop;
    const c = document.getElementById('conf');
    c.textContent = s.confidence + '%';
    c.className = 'value' + (s.confidence < s.threshold ? ' low' : '');
    document.getElementById('seq').textContent = '#' + s.seq;
    document.getElementById('warn').hidden = s.model_ok;
    document.getElementById('foot').textContent =
      'frame ' + (s.image_bytes/1024).toFixed(1) + ' KB · capture #' + s.cap +
      ' · ' + (s.interval_ms ? 'every ' + (s.interval_ms/1000) + ' s'
                             : 'back to back, ' + (1000/Math.max(1,s.prep_ms+s.infer_ms)).toFixed(2) + ' fps') +
      ' · fan acts at ' + s.threshold + '% and above';
    document.getElementById('diag').textContent =
      'prep ' + s.prep_ms + ' ms · inference ' + s.infer_ms + ' ms · arena ' +
      (s.arena_used/1024).toFixed(0) + ' KB · mean RGB ' +
      s.mean_r + ', ' + s.mean_g + ', ' + s.mean_b;
    // Keyed to the capture counter, not the detection counter, so a frame that
    // failed to classify still refreshes the picture instead of freezing it.
    document.getElementById('shot').src = '/capture?s=' + s.cap;
  }catch(e){}
}
tick();
// Polling faster than the board can answer just piles up requests that time
// out during the inference window and makes the page look more broken, not
// less. This roughly matches one cycle.
setInterval(tick, 1200);
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
  uint8_t mr, mg, mb;
  cropModelLastMeanRGB(&mr, &mg, &mb);

  // Sized with room to spare: snprintf truncates rather than overflows, but a
  // truncated response is invalid JSON and the page just silently stops
  // updating, which looks like a hung board rather than a full buffer.
  char json[512];
  snprintf(json, sizeof(json),
           "{\"crop_id\":%u,\"crop\":\"%s\",\"confidence\":%u,\"seq\":%lu,"
           "\"cap\":%lu,"
           "\"image_bytes\":%u,\"interval_ms\":%u,\"threshold\":%u,"
           "\"model_ok\":%s,\"prep_ms\":%lu,\"infer_ms\":%lu,"
           "\"arena_used\":%u,\"mean_r\":%u,\"mean_g\":%u,\"mean_b\":%u}",
           g_cropId, CROP_NAMES[g_cropId], g_confidence,
           (unsigned long)g_seq, (unsigned long)g_capSeq, (unsigned)g_jpegLen,
           DETECT_INTERVAL_MS, CONFIDENCE_THRESHOLD,
           g_modelOk ? "true" : "false",
           (unsigned long)cropModelLastPrepMs(),
           (unsigned long)cropModelLastInferMs(),
           (unsigned)cropModelArenaUsed(), mr, mg, mb);
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

  // Load the model before Wi-Fi comes up. The arena is a single 600 KB PSRAM
  // allocation and it is the largest one this sketch makes -- taking it while
  // the heap is still unfragmented is the difference between it succeeding and
  // failing intermittently.
  //
  // A failure here is not fatal: the camera, the viewer and the serial log all
  // still work, which is exactly what you need in order to debug why the model
  // would not load. The page says so rather than showing a stale verdict.
  g_modelOk = cropModelInit();
  if (!g_modelOk) {
    Serial.println("[CAM] Running WITHOUT a model -- images only, no detection");
  }

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
  // Print the address of the interface ESP-NOW actually transmits on. The
  // station MAC reads as all zeros in AP mode and is misleading here.
#if ENABLE_WEB_VIEWER
  Serial.print("[CAM] This board's SoftAP MAC: ");
  Serial.println(WiFi.softAPmacAddress());
#else
  Serial.print("[CAM] This board's station MAC: ");
  Serial.println(WiFi.macAddress());
#endif
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
  bool ok = runInference(g_jpeg, g_jpegLen, g_jpegW, g_jpegH,
                         &g_cropId, &g_confidence);
  digitalWrite(STATUS_LED_PIN, HIGH);

  Serial.printf("[CAM] Image captured (%ux%u, %u bytes)\n",
                g_jpegW, g_jpegH, (unsigned)g_jpegLen);

  if (!ok) {
    // Frame kept, result not updated. The web page will show the new picture
    // beside the previous verdict, which is honest -- nothing was classified.
    Serial.println("[CAM] Inference failed, keeping previous result");
    return;
  }

  g_seq++;

  uint8_t mr, mg, mb;
  cropModelLastMeanRGB(&mr, &mg, &mb);
  uint8_t sr, sg, sb, sc;
  cropModelLastSubjectRGB(&sr, &sg, &sb, &sc);

  Serial.printf("[CAM] Crop: %s\n", CROP_NAMES[g_cropId]);
  Serial.printf("[CAM] Confidence: %u%%\n", g_confidence);
  // Two colour readings, and the second is the useful one.
  //
  // "frame" is the mean over the whole 96x96 tensor. It is dominated by the
  // white box, so a red tomato moves it by a couple of units -- less than the
  // sensor's own cast. It tells you the picture is dark; it cannot tell you
  // the item is red.
  //
  // "subject" is the mean of the tenth of pixels with the most chroma, i.e.
  // the item. On a tomato R should be far above G and B. If B leads instead,
  // CROP_SWAP_RB is set the wrong way round. If chroma is under ~15 on a
  // coloured item, the frame is washed out and the colour the model relies on
  // is not in the picture at all.
  Serial.printf("[CAM] prep %lums, infer %lums | frame RGB %u,%u,%u"
                " | subject RGB %u,%u,%u chroma %u\n",
                (unsigned long)cropModelLastPrepMs(),
                (unsigned long)cropModelLastInferMs(),
                mr, mg, mb, sr, sg, sb, sc);

#if ENABLE_ESPNOW
  if (g_cropId == CROP_EMPTY) {
    // An empty box is a confident, correct answer with nothing to act on. The
    // fan should hold whatever the last real crop set rather than be driven by
    // a reading that names no crop at all.
    Serial.println("[CAM] Box is empty, nothing to send");
  } else if (g_confidence >= CONFIDENCE_THRESHOLD) {
    sendResult();
  } else {
    // Sending it anyway would be fine -- the main board checks the threshold
    // too -- but not sending keeps the radio quiet on frames that cannot
    // change anything.
    Serial.printf("[CAM] Below %u%% threshold, not sent\n", CONFIDENCE_THRESHOLD);
  }
#endif
}
