/*
 * crop_model.cpp -- TensorFlow Lite Micro inference for the ESP32-CAM.
 *
 * The pipeline here mirrors cropnet/preprocess.py step for step:
 *
 *     JPEG frame -> RGB888 -> centre-crop to a square -> 96x96 -> int8
 *
 * There is deliberately no normalisation. The model carries a Rescaling layer
 * internally, so its input tensor takes raw pixel bytes; with the measured
 * quantization of scale=1.0, zero_point=-128 the whole conversion is
 * "pixel - 128". Keeping it that way is what stops the board and the desktop
 * from disagreeing about what an image looks like.
 */

#include <Arduino.h>
#include <math.h>
#include <esp_heap_caps.h>    // heap_caps_malloc(), to place the arena by hand
#include "crop_model.h"
#include "model_data.h"
#include "img_converters.h"   // fmt2rgb888(), from the esp32-camera driver

// --------------------------------------------------------------------------
// TFLite Micro library
//
// Two Arduino ports are in circulation and their interpreter constructors
// differ. Rather than make you edit includes, detect which one is installed.
// See section 2 of the README for which to install.
// --------------------------------------------------------------------------

#if __has_include(<tflm_esp32.h>)
  // eloquentarduino/tflm_esp32 -- the one this was built and compile-tested
  // against. Install it from the Library Manager as "tflm_esp32".
  #include <tflm_esp32.h>
  #define TFLM_NEEDS_ERROR_REPORTER 0
#elif __has_include(<TensorFlowLite.h>)
  // tflite-micro-arduino-examples and its forks -- same modern API.
  #include <TensorFlowLite.h>
  #define TFLM_NEEDS_ERROR_REPORTER 0
#elif __has_include(<TensorFlowLite_ESP32.h>)
  // tanakamasayuki/Arduino_TensorFlowLite_ESP32 -- an older snapshot that
  // wants an explicit ErrorReporter. Untested here, and it predates ESP32
  // core 3.x; treat this branch as a fallback, not a recommendation.
  #include <TensorFlowLite_ESP32.h>
  #include "tensorflow/lite/micro/micro_error_reporter.h"
  #define TFLM_NEEDS_ERROR_REPORTER 1
#else
  #error "No TensorFlow Lite Micro library found. See ESP32_CAM/README.md section 2."
#endif

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// --------------------------------------------------------------------------
// State
// --------------------------------------------------------------------------

namespace {

constexpr int kInputSize = CROP_MODEL_INPUT_WIDTH;   // 96, square
constexpr int kNumClasses = CROP_MODEL_NUM_CLASSES;  // 5

const tflite::Model     *g_model       = nullptr;
tflite::MicroInterpreter *g_interpreter = nullptr;
TfLiteTensor            *g_input       = nullptr;
TfLiteTensor            *g_output      = nullptr;

uint8_t *g_arena = nullptr;   // PSRAM
uint8_t *g_rgb   = nullptr;   // PSRAM, decoded frame, width*height*3
size_t   g_rgbCapacity = 0;

bool   g_ready = false;
size_t g_arenaUsed = 0;

uint32_t g_prepMs  = 0;
uint32_t g_inferMs = 0;
uint8_t  g_meanR = 0, g_meanG = 0, g_meanB = 0;

// Colour of the item itself, as opposed to the whole frame. See measureSubject().
uint8_t  g_subjR = 0, g_subjG = 0, g_subjB = 0, g_subjChroma = 0;
void measureSubject(const uint8_t *tensor);

// Input quantization, read from the model rather than hardcoded, so a
// re-export with different parameters cannot silently corrupt the input.
float g_inScaleInv = 1.0f;
int   g_inZeroPoint = -128;
float g_outScale = 1.0f / 256.0f;
int   g_outZeroPoint = -128;

#if TFLM_NEEDS_ERROR_REPORTER
tflite::MicroErrorReporter g_errorReporter;
#endif

// The exact operator set this model uses, from
// `python -c "...interpreter._get_ops_details()"` on model_int8.tflite:
//   ADD, CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, MEAN, MUL, SOFTMAX
//
// ADD and MUL are the Rescaling layer. RESHAPE is not in the current export
// but costs almost nothing and is the op most likely to appear after a
// backbone change, so it is registered pre-emptively.
//
// Listing ops explicitly rather than using AllOpsResolver saves both flash and
// arena -- AllOpsResolver drags in every kernel TFLM has.
constexpr int kOpCount = 8;
tflite::MicroMutableOpResolver<kOpCount> g_resolver;

bool registerOps() {
  if (g_resolver.AddAdd()             != kTfLiteOk) return false;
  if (g_resolver.AddConv2D()          != kTfLiteOk) return false;
  if (g_resolver.AddDepthwiseConv2D() != kTfLiteOk) return false;
  if (g_resolver.AddFullyConnected()  != kTfLiteOk) return false;
  if (g_resolver.AddMean()            != kTfLiteOk) return false;
  if (g_resolver.AddMul()             != kTfLiteOk) return false;
  if (g_resolver.AddSoftmax()         != kTfLiteOk) return false;
  if (g_resolver.AddReshape()         != kTfLiteOk) return false;
  return true;
}

// Area-average downscale of a square RGB888 region to kInputSize x kInputSize.
//
// Not plain bilinear, on purpose. preprocess.py resizes with Pillow, whose
// BILINEAR filter is antialiased -- for a 240->96 reduction it averages
// roughly a 5x5 neighbourhood per output pixel. Naive bilinear samples only
// 2x2 and throws the rest away, which produces visibly noisier, aliased input
// than the model was trained on. Box averaging over the full source rectangle
// is the cheap approximation that stays close to Pillow's result.
void resizeSquareToInput(const uint8_t *src, int srcStride,
                         int srcX, int srcY, int srcSide,
                         uint8_t *dst) {
  uint32_t sumR = 0, sumG = 0, sumB = 0;

  for (int oy = 0; oy < kInputSize; oy++) {
    const int y0 = srcY + (oy * srcSide) / kInputSize;
    int y1 = srcY + ((oy + 1) * srcSide) / kInputSize;
    if (y1 <= y0) y1 = y0 + 1;

    for (int ox = 0; ox < kInputSize; ox++) {
      const int x0 = srcX + (ox * srcSide) / kInputSize;
      int x1 = srcX + ((ox + 1) * srcSide) / kInputSize;
      if (x1 <= x0) x1 = x0 + 1;

      uint32_t accA = 0, accG = 0, accC = 0;   // A and C are the outer channels
      uint32_t count = 0;

      for (int y = y0; y < y1; y++) {
        const uint8_t *row = src + (size_t)y * srcStride + (size_t)x0 * 3;
        for (int x = x0; x < x1; x++) {
          accA += row[0];
          accG += row[1];
          accC += row[2];
          row += 3;
          count++;
        }
      }

      const uint8_t a = (uint8_t)(accA / count);
      const uint8_t g = (uint8_t)(accG / count);
      const uint8_t c = (uint8_t)(accC / count);

      // The decoder gives B,G,R; the model wants R,G,B.
      uint8_t *out = dst + ((size_t)oy * kInputSize + ox) * 3;
#if CROP_SWAP_RB
      out[0] = c; out[1] = g; out[2] = a;
#else
      out[0] = a; out[1] = g; out[2] = c;
#endif
      sumR += out[0];
      sumG += out[1];
      sumB += out[2];
    }
  }

  const uint32_t pixels = (uint32_t)kInputSize * kInputSize;
  g_meanR = (uint8_t)(sumR / pixels);
  g_meanG = (uint8_t)(sumG / pixels);
  g_meanB = (uint8_t)(sumB / pixels);

  measureSubject(dst);
}

// Mean colour of the most colourful pixels in the tensor, i.e. the item.
//
// The whole-frame mean is close to useless for diagnosing colour here, and
// that cost real debugging time: the box is a large white surface and the item
// is a small part of the frame, so a red tomato moved the frame mean by about
// two units per channel -- less than the sensor's own cast. The frame mean can
// tell you the picture is dark. It cannot tell you the item is red.
//
// So: take the pixels whose chroma (max channel - min channel) is in the top
// tenth, and average those. On an empty box nothing is colourful and the
// result is neutral, which is itself the correct answer.
void measureSubject(const uint8_t *tensor) {
  const uint32_t pixels = (uint32_t)kInputSize * kInputSize;

  // Chroma histogram, so the threshold needs no sorting or second buffer.
  uint16_t hist[256] = {0};
  for (uint32_t i = 0; i < pixels; i++) {
    const uint8_t *p = tensor + i * 3;
    uint8_t hi = p[0] > p[1] ? p[0] : p[1]; if (p[2] > hi) hi = p[2];
    uint8_t lo = p[0] < p[1] ? p[0] : p[1]; if (p[2] < lo) lo = p[2];
    hist[hi - lo]++;
  }

  const uint32_t want = pixels / 10;
  uint32_t seen = 0;
  int threshold = 255;
  for (int c = 255; c >= 0; c--) {
    seen += hist[c];
    if (seen >= want) { threshold = c; break; }
  }

  uint32_t sr = 0, sg = 0, sb = 0, n = 0, sc = 0;
  for (uint32_t i = 0; i < pixels; i++) {
    const uint8_t *p = tensor + i * 3;
    uint8_t hi = p[0] > p[1] ? p[0] : p[1]; if (p[2] > hi) hi = p[2];
    uint8_t lo = p[0] < p[1] ? p[0] : p[1]; if (p[2] < lo) lo = p[2];
    if ((int)(hi - lo) >= threshold) {
      sr += p[0]; sg += p[1]; sb += p[2]; sc += (hi - lo); n++;
    }
  }
  if (n == 0) n = 1;
  g_subjR = (uint8_t)(sr / n);
  g_subjG = (uint8_t)(sg / n);
  g_subjB = (uint8_t)(sb / n);
  g_subjChroma = (uint8_t)(sc / n);
}

}  // namespace

// --------------------------------------------------------------------------
// Public interface
// --------------------------------------------------------------------------

bool cropModelInit(void) {
  if (g_ready) return true;

  if (!psramFound()) {
    // Not a recoverable situation: the arena alone is larger than the free
    // internal heap once Wi-Fi and the camera driver have taken their share.
    Serial.println("[CV] No PSRAM. This model cannot run on this module.");
    return false;
  }

  // Internal SRAM first. Every intermediate activation is read and written
  // through this buffer, and PSRAM is reached over a much slower bus, so where
  // the arena lands dominates inference time -- far more than it looks like it
  // should. Fall back to PSRAM rather than refusing to run.
  g_arena = (uint8_t *)heap_caps_malloc(CROP_ARENA_BYTES,
                                        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (g_arena) {
    Serial.printf("[CV] Arena: %u bytes in internal SRAM (fast)\n",
                  (unsigned)CROP_ARENA_BYTES);
  } else {
    g_arena = (uint8_t *)ps_malloc(CROP_ARENA_BYTES);
    if (g_arena) {
      Serial.printf("[CV] Arena: %u bytes in PSRAM -- internal SRAM was full, "
                    "expect much slower inference\n", (unsigned)CROP_ARENA_BYTES);
    }
  }
  if (!g_arena) {
    Serial.printf("[CV] Could not allocate a %u byte arena\n",
                  (unsigned)CROP_ARENA_BYTES);
    return false;
  }

  g_model = tflite::GetModel(g_crop_model_data);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[CV] Model schema %lu, library expects %d. Re-export the "
                  "model or change library version.\n",
                  (unsigned long)g_model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  if (!registerOps()) {
    Serial.println("[CV] Op resolver full -- raise kOpCount");
    return false;
  }

  static tflite::MicroInterpreter interpreter(
      g_model, g_resolver, g_arena, CROP_ARENA_BYTES
#if TFLM_NEEDS_ERROR_REPORTER
      , &g_errorReporter
#endif
  );
  g_interpreter = &interpreter;

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[CV] AllocateTensors failed -- raise CROP_ARENA_BYTES");
    return false;
  }

  g_input  = g_interpreter->input(0);
  g_output = g_interpreter->output(0);

  // Fail loudly on a mismatch rather than producing confident nonsense.
  if (g_input->dims->size != 4 ||
      g_input->dims->data[1] != kInputSize ||
      g_input->dims->data[2] != kInputSize ||
      g_input->dims->data[3] != CROP_MODEL_INPUT_CHANNELS ||
      g_input->type != kTfLiteInt8) {
    Serial.println("[CV] Unexpected input tensor shape or type");
    return false;
  }
  if (g_output->dims->data[g_output->dims->size - 1] != kNumClasses) {
    Serial.println("[CV] Output class count does not match model_data.h");
    return false;
  }

  // Measured on this export: input scale 1.0 / zp -128, output scale 1/256 /
  // zp -128. Read rather than assumed, so a re-export stays correct.
  g_inScaleInv   = (g_input->params.scale != 0.0f) ? 1.0f / g_input->params.scale : 1.0f;
  g_inZeroPoint  = g_input->params.zero_point;
  g_outScale     = g_output->params.scale;
  g_outZeroPoint = g_output->params.zero_point;

  g_arenaUsed = g_interpreter->arena_used_bytes();
  g_ready = true;

  Serial.printf("[CV] Model loaded: %u bytes of weights\n",
                (unsigned)g_crop_model_data_len);
  Serial.printf("[CV] Arena used: %u of %u bytes\n",
                (unsigned)g_arenaUsed, (unsigned)CROP_ARENA_BYTES);
  Serial.printf("[CV] Input  int8 %dx%dx%d, scale %.6f zp %d\n",
                kInputSize, kInputSize, CROP_MODEL_INPUT_CHANNELS,
                g_input->params.scale, g_inZeroPoint);
  Serial.printf("[CV] Output int8 %d classes, scale %.6f zp %d\n",
                kNumClasses, g_outScale, g_outZeroPoint);
  return true;
}

bool cropModelReady(void) { return g_ready; }

bool cropModelClassify(const uint8_t *jpeg, size_t len,
                       uint16_t width, uint16_t height,
                       uint8_t *cropId, uint8_t *confidence) {
  if (!g_ready) return false;

  const uint32_t tPrep = millis();

  // One decode buffer, allocated on first use and reused. Allocating per frame
  // would fragment PSRAM over hours of running.
  const size_t needed = (size_t)width * height * 3;
  if (needed > g_rgbCapacity) {
    if (g_rgb) free(g_rgb);
    g_rgb = (uint8_t *)ps_malloc(needed);
    g_rgbCapacity = g_rgb ? needed : 0;
    if (!g_rgb) {
      Serial.printf("[CV] Could not allocate a %u byte decode buffer\n",
                    (unsigned)needed);
      return false;
    }
  }

  if (!fmt2rgb888(jpeg, len, PIXFORMAT_JPEG, g_rgb)) {
    Serial.println("[CV] JPEG decode failed");
    return false;
  }

  // Centre-crop the largest square, exactly as center_crop_to_square() does.
  // The camera looks down at one item, so the centre is the subject and the
  // edges are tray -- squashing the frame instead would distort every shape
  // the model relies on.
  const int side = (width < height) ? width : height;
  const int left = (width - side) / 2;
  const int top  = (height - side) / 2;

  // Scratch for the 96x96x3 RGB image before quantization. 27 KB on the stack
  // would overflow the Arduino task, so it is static.
  static uint8_t rgbInput[kInputSize * kInputSize * 3];
  resizeSquareToInput(g_rgb, width * 3, left, top, side, rgbInput);

  // uint8 pixel -> int8 tensor. With scale 1.0 and zp -128 this is a subtract
  // of 128; the general form is kept so a re-quantized export still works.
  int8_t *in = g_input->data.int8;
  for (size_t i = 0; i < sizeof(rgbInput); i++) {
    int32_t q = (int32_t)lroundf(rgbInput[i] * g_inScaleInv) + g_inZeroPoint;
    if (q < -128) q = -128;
    if (q > 127)  q = 127;
    in[i] = (int8_t)q;
  }

  g_prepMs = millis() - tPrep;

  const uint32_t tInfer = millis();
  if (g_interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[CV] Invoke failed");
    return false;
  }
  g_inferMs = millis() - tInfer;

  // Softmax is inside the model, so these dequantize straight to probabilities.
  const int8_t *out = g_output->data.int8;
  int best = 0;
  for (int i = 1; i < kNumClasses; i++) {
    if (out[i] > out[best]) best = i;
  }

  float p = (out[best] - g_outZeroPoint) * g_outScale;
  if (p < 0.0f) p = 0.0f;
  if (p > 1.0f) p = 1.0f;

  *cropId     = (uint8_t)best;
  *confidence = (uint8_t)lroundf(p * 100.0f);
  return true;
}

void cropModelLastSubjectRGB(uint8_t *r, uint8_t *g, uint8_t *b, uint8_t *chroma) {
  if (r) *r = g_subjR;
  if (g) *g = g_subjG;
  if (b) *b = g_subjB;
  if (chroma) *chroma = g_subjChroma;
}

uint32_t cropModelLastPrepMs(void)  { return g_prepMs; }
uint32_t cropModelLastInferMs(void) { return g_inferMs; }
size_t   cropModelArenaUsed(void)   { return g_arenaUsed; }

void cropModelLastMeanRGB(uint8_t *r, uint8_t *g, uint8_t *b) {
  if (r) *r = g_meanR;
  if (g) *g = g_meanG;
  if (b) *b = g_meanB;
}
