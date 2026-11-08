"""Model layer — data preparation.

Responsibilities:
    * Extract the raw Galaxy Zoo archives.
    * Load the solutions CSV and derive morphological labels.
    * Split images into train/ and validation/ class folders.
    * Build augmented Keras data generators.
    * Provide flattened image arrays for PCA-based EDA.
"""

from __future__ import annotations

import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image as keras_image

from .. import config


# --- Raw data --------------------------------------------------------------
def extract_archives(zip_paths: List[Path], dest_dir: Path) -> None:
    """Extract each zip archive in ``zip_paths`` into ``dest_dir``."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
        print(f"Extracted {zip_path} -> {dest_dir}")


def load_solutions(csv_path: Path = config.TRAINING_SOLUTIONS_CSV) -> pd.DataFrame:
    """Load the training-solutions CSV and normalise column names.

    The original columns are named ``ClassN.M``; they are renamed to
    ``QN.M`` (question/answer) to match the labelling thresholds.
    """
    df = pd.read_csv(csv_path)
    df.columns = [col.replace("Class", "Q") for col in df.columns]
    return df


# --- Labelling -------------------------------------------------------------
def filter_labels(df: pd.DataFrame) -> Dict[str, List[int]]:
    """Return the galaxy IDs belonging to each morphological class.

    Labels are derived from Galaxy Zoo vote fractions using the thresholds in
    :data:`config.LABEL_THRESHOLDS`.
    """
    galaxy_ids: Dict[str, List[int]] = {}
    for class_name, thresholds in config.LABEL_THRESHOLDS.items():
        mask = pd.Series(True, index=df.index)
        for column, minimum in thresholds.items():
            mask &= df[column] > minimum
        galaxy_ids[class_name] = df.loc[mask, "GalaxyID"].tolist()
        print(f"Total {class_name} examples: {len(galaxy_ids[class_name])}")
    return galaxy_ids


# --- Train / validation split ---------------------------------------------
def _split_class(
    source_images_dir: Path,
    output_dir: Path,
    label: str,
    galaxy_ids: List[int],
    train_split: float,
) -> None:
    """Copy one class's images into train/ and validation/ subfolders."""
    train_dest = Path(output_dir) / "train" / label
    val_dest = Path(output_dir) / "validation" / label
    train_dest.mkdir(parents=True, exist_ok=True)
    val_dest.mkdir(parents=True, exist_ok=True)

    ids = list(galaxy_ids)
    random.shuffle(ids)
    split_index = int(len(ids) * train_split)

    for galaxy_id in ids[:split_index]:
        shutil.copyfile(
            Path(source_images_dir) / f"{galaxy_id}.jpg",
            train_dest / f"{galaxy_id}.jpg",
        )
    for galaxy_id in ids[split_index:]:
        shutil.copyfile(
            Path(source_images_dir) / f"{galaxy_id}.jpg",
            val_dest / f"{galaxy_id}.jpg",
        )
    print(f"{label}: {split_index} train / {len(ids) - split_index} validation")


def build_dataset(
    galaxy_ids: Dict[str, List[int]],
    source_images_dir: Path = config.TRAINING_IMAGES_DIR,
    output_dir: Path = config.PROCESSED_DIR,
    train_split: float = config.TRAIN_SPLIT,
) -> None:
    """Build the processed train/validation directory tree for all classes."""
    for label, ids in galaxy_ids.items():
        _split_class(source_images_dir, output_dir, label, ids, train_split)


def count_images(split_dir: Path) -> Dict[str, int]:
    """Count images per class inside a split directory (train or validation)."""
    counts = {}
    for class_name in config.CLASSES:
        class_dir = Path(split_dir) / class_name
        counts[class_name] = len(os.listdir(class_dir)) if class_dir.exists() else 0
    return counts


# --- Generators ------------------------------------------------------------
def build_generators(
    train_dir: Path = config.TRAIN_DIR,
    validation_dir: Path = config.VALIDATION_DIR,
    img_size=config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
):
    """Create augmented training and validation data generators."""
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.2,
    )
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
    )

    train_generator = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        class_mode="categorical",
    )
    validation_generator = validation_datagen.flow_from_directory(
        str(validation_dir),
        target_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode="categorical",
    )
    return train_generator, validation_generator


# --- Flattened arrays for PCA / classical EDA ------------------------------
def load_flattened_images(image_directory: Path, img_size=config.IMG_SIZE) -> np.ndarray:
    """Load every image in a directory as a flattened, stacked array."""
    data = []
    for filename in os.listdir(image_directory):
        img = keras_image.load_img(
            Path(image_directory) / filename, target_size=img_size
        )
        data.append(keras_image.img_to_array(img).flatten())
    return np.array(data)


def load_single_image(image_path: Path, img_size=config.IMG_SIZE) -> np.ndarray:
    """Load and normalise a single image into a model-ready batch of size 1."""
    img = keras_image.load_img(image_path, target_size=img_size)
    img_array = keras_image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)
