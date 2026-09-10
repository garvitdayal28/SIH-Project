/*
 * crop_model.h -- on-device crop classification for the ESP32-CAM.
 *
 * Wraps TensorFlow Lite Micro and the preprocessing that has to match
 * "Crop detection model/cropnet/preprocess.py" exactly. Everything about the
 * model lives behind this interface, so ESP32_CAM.ino never has to know that
 * TFLite exists.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

// Tensor arena. Holds the model's intermediate activations -- not the weights,
// which are read in place from flash.
//
// Measured need for MobileNetV1 alpha=0.5 at 96x96 is well under this; the
// slack is deliberate because AllocateTensors() failing is a hard stop. After
// the first successful boot the serial log prints the real figure
// ("arena used: N bytes") and this can be trimmed to that plus ~10%.
//
// It is allocated in PSRAM, so being generous costs nothing on a 4 MB module.
#define CROP_ARENA_BYTES (600 * 1024)

// esp32-camera's JPEG decoder writes its output in B,G,R order rather than
// R,G,B -- a consequence of that converter being shared with the BMP writer,
// where BGR is the format on disk. The model was trained on RGB.
//
// Getting this wrong does not fail loudly. It silently swaps red and blue,
// which is close to worst case for this particular model: config.py notes that
// onion, potato and ginger are separated mainly BY their brown/tan colour.
//
// Leave at 1 unless the mean-RGB check in the README says otherwise.
#define CROP_SWAP_RB 1

#ifdef __cplusplus
extern "C" {
#endif

// Load the model and allocate tensors. Call once from setup(), after the
// camera is up. Returns false if the arena is too small, PSRAM is missing, or
// the model does not match the compiled-in op resolver.
bool cropModelInit(void);

// True once cropModelInit() has succeeded.
bool cropModelReady(void);

// Classify one JPEG frame straight from the camera.
//
//   jpeg/len   the frame as captured
//   width/hgt  its dimensions, from camera_fb_t
//   cropId     out: index into g_crop_labels, i.e. the crop_id on the wire
//   confidence out: 0-100
//
// Returns false if decoding or inference failed, in which case the outputs are
// untouched and the caller should keep its previous result.
bool cropModelClassify(const uint8_t *jpeg, size_t len,
                       uint16_t width, uint16_t height,
                       uint8_t *cropId, uint8_t *confidence);

// How long the last cropModelClassify() took, in milliseconds, split into the
// JPEG decode + resize stage and the inference itself. Useful for deciding
// what DETECT_INTERVAL_MS can realistically be.
uint32_t cropModelLastPrepMs(void);
uint32_t cropModelLastInferMs(void);

// Mean R, G and B of the 96x96 tensor that was actually fed to the model.
//
// This exists to catch a channel swap. Point the camera at something strongly
// red: r should come back well above b. If they are reversed, flip
// CROP_SWAP_RB. There is no other symptom -- the model just quietly gets worse.
void cropModelLastMeanRGB(uint8_t *r, uint8_t *g, uint8_t *b);

// Peak arena usage reported by the interpreter after allocation, for tuning
// CROP_ARENA_BYTES.
size_t cropModelArenaUsed(void);

#ifdef __cplusplus
}
#endif
