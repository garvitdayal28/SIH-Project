"""
Custom augmentation layers.

Only one lives here, and only because Keras has no equivalent: a random
white-balance shift.

Why not RandomHue
-----------------
The ESP32-CAM's frames run warm and washed out (my images/from ESPCAM/), so the
model needs to tolerate a shifted white point. The obvious layer for "shift the
colours" is RandomHue, and it was measured and rejected -- see AUG_HUE in
config.py, where hue=0.03 cost 15 points of test accuracy.

The two operations are not interchangeable. A hue rotation moves red towards
green, which destroys the colour identity that onion, potato and ginger are
separated by. A white-balance gain scales the red and blue channels in opposite
directions, which moves the white point while leaving the *ordering* of object
colours intact: under a warm cast a tomato is still the reddest thing in frame,
and ginger is still browner than a lemon.

That is what a camera's auto-white-balance actually gets wrong, so it is also the
more faithful simulation.

Lifetime
--------
This layer is training-only, and it is only ever used inside the Sequential named
"augmentation" in scripts/train.py. scripts/export_tflite.py strips that
Sequential out before conversion, so nothing here reaches the .tflite file or the
ESP32 -- it costs no flash and no inference time.

It does get written into models/model.keras, so anything calling load_model on
that file has to have imported this module first. train.py and export_tflite.py
both do. The @register_keras_serializable decorator is what makes the reload work.
"""

from __future__ import annotations

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="cropnet")
class RandomWhiteBalance(keras.layers.Layer):
    """Randomly shift the white point by scaling red up and blue down, or vice versa.

    Args:
        factor: maximum gain applied in either direction. 0.12 means red is
            scaled by up to 1.12 while blue is scaled by 0.88, and the reverse
            for a cool shift. 0 disables the layer.
        value_range: the range the incoming pixels occupy. This layer sits before
            the model's Rescaling layer, so it sees raw 0-255 pixels.
        seed: seed for the internal generator.

    Input and output shape are both (batch, height, width, 3).
    """

    def __init__(
        self,
        factor: float = 0.12,
        value_range: tuple[float, float] = (0.0, 255.0),
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if factor < 0:
            raise ValueError(f"factor must be >= 0, got {factor}")
        self.factor = float(factor)
        self.value_range = tuple(float(v) for v in value_range)
        self.seed = seed
        self.generator = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=False):
        # Inactive at inference, like every other Keras random layer. Without
        # this the exported model would still be randomised if the augmentation
        # block were ever left in the graph.
        if not training or self.factor == 0.0:
            return inputs

        # One shift per image, not per batch: a batch-wide shift would correlate
        # the augmentation across samples and waste most of its value.
        batch = ops.shape(inputs)[0]
        shift = keras.random.uniform(
            (batch, 1, 1, 1),
            minval=-self.factor,
            maxval=self.factor,
            dtype=inputs.dtype,
            seed=self.generator,
        )

        ones = ops.ones_like(shift)
        # Green is left alone. Scaling red and blue in opposite directions about
        # a fixed green keeps overall luminance roughly constant, so this stays
        # independent of RandomBrightness rather than partly duplicating it.
        gains = ops.concatenate([ones + shift, ones, ones - shift], axis=-1)

        return ops.clip(inputs * gains, self.value_range[0], self.value_range[1])

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "factor": self.factor,
            "value_range": self.value_range,
            "seed": self.seed,
        })
        return config
