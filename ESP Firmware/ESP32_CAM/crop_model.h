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
// Measured on this board: "[CV] Arena used: 159548 bytes". 176 KB gives that
// about 10% headroom.
//
// The size matters for more than memory. At 600 KB the arena could only ever
// live in PSRAM; at this size it fits in internal SRAM, which is several times
// faster for the scattered reads and writes inference does. cropModelInit()
// tries internal first and falls back to PSRAM, and says which it got.
#define CROP_ARENA_BYTES (176 * 1024)

// esp32-camera's JPEG decoder writes its output in B,G,R order rather than
// R,G,B -- a consequence of that converter being shared with the BMP writer,
// where BGR is the format on disk. The model was trained on RGB.
//
// Getting this wrong does not fail loudly. It silently swaps red and blue,
// which is close to worst case for this particular model: config.py notes that
// a yellow lemon and a red tomato are separated mainly BY their colour.
//
// Set to 0 on 2026-09-11, from evidence on the real board. It had been 1.
//
// Symptom: tomato read as lemon essentially every time, while lemon, onion and
// empty were all correct. Two independent things point at the same cause.
//
//   The serial log's frame mean had red as the LOWEST channel and blue the
//   highest on frames containing a red tomato -- 136,142,144 and 126,131,132
//   and 130,129,134. That is backwards for a red subject.
//
//   Replaying the real rig photographs through the model with R and B
//   exchanged, at the tighter framing this camera gives, reproduces the
//   symptom and nothing else does: tomato 9/23 with lemon as the largest
//   failure bucket, while lemon stays 23/24 and onion 24/25. With the channels
//   correct, all three are perfect.
//
// So this driver build already hands back R,G,B and the swap was corrupting it.
// The BGR quirk described above is real but evidently version-dependent, which
// is why this is a switch and not an assumption.
//
// Confirm on the board with the "subject RGB" figure in the detection log --
// see cropModelLastSubjectRGB(). On a tomato R should now lead by a wide
// margin. If instead B leads, put this back to 1.
#define CROP_SWAP_RB 0

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

// Mean colour of the ITEM, not the whole frame -- the average of the tenth of
// pixels with the most chroma, plus that chroma value.
//
// This is the number to read when a class is being confused with another. The
// whole-frame mean cannot answer it: the box is a big white surface, so a red
// tomato shifts the frame mean by a couple of units, less than the sensor's
// own colour cast. The subject mean shifts by a hundred.
//
// Expected, roughly:  tomato  R much greater than G and B, chroma > 40
//                     lemon   R and G high, B lower, chroma > 40
//                     onion   all low, chroma modest
//                     empty   near-neutral, chroma < 15
//
// If a tomato reports B above R, CROP_SWAP_RB is set the wrong way.
// If chroma is under ~15 on a coloured item, the frame is desaturated and the
// colour the model relies on is simply not there.
void cropModelLastSubjectRGB(uint8_t *r, uint8_t *g, uint8_t *b, uint8_t *chroma);

// Peak arena usage reported by the interpreter after allocation, for tuning
// CROP_ARENA_BYTES.
size_t cropModelArenaUsed(void);

#ifdef __cplusplus
}
#endif
