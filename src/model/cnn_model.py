"""Model layer — the CNN architecture and persistence.

A convolutional neural network built from scratch (no transfer learning) to
classify galaxy images into the three morphological classes.
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from keras.models import load_model as _keras_load_model

from .. import config


def build_model(
    input_shape=config.INPUT_SHAPE,
    num_classes: int = config.NUM_CLASSES,
) -> tf.keras.Model:
    """Build and compile the CNN classifier."""
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                64, (3, 3), activation="relu", input_shape=input_shape
            ),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def save_model(model: tf.keras.Model, path: Path = config.MODEL_PATH) -> None:
    """Persist a trained model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}")


def load_trained_model(path: Path = config.MODEL_PATH) -> tf.keras.Model:
    """Load a previously trained model from disk."""
    return _keras_load_model(path)
