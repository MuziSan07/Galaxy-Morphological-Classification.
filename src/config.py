"""Shared configuration: paths and hyperparameters.

Edit the paths here to match your environment. The defaults assume the
Galaxy Zoo files have been downloaded and extracted into ``<project>/data/raw``.
"""

from pathlib import Path

# --- Directories -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                 # extracted Galaxy Zoo files live here
PROCESSED_DIR = DATA_DIR / "processed"     # train/ and validation/ are built here

TRAIN_DIR = PROCESSED_DIR / "train"
VALIDATION_DIR = PROCESSED_DIR / "validation"

TRAINING_SOLUTIONS_CSV = RAW_DIR / "training_solutions_rev1.csv"
TRAINING_IMAGES_DIR = RAW_DIR / "images_training_rev1"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "galaxy_classifier.h5"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"     # saved figures / reports

# --- Classes ---------------------------------------------------------------
CLASSES = ["elliptical", "lenticular", "spiral"]
NUM_CLASSES = len(CLASSES)

# --- Image / training hyperparameters --------------------------------------
IMG_SIZE = (150, 150)
INPUT_SHAPE = (150, 150, 3)
BATCH_SIZE = 32
EPOCHS = 100
TRAIN_SPLIT = 0.7          # fraction of each class used for training

# Thresholds used to derive morphological labels from Galaxy Zoo vote fractions
LABEL_THRESHOLDS = {
    "elliptical": {"Q1.1": 0.8, "Q7.1": 0.4},
    "lenticular": {"Q1.1": 0.8, "Q7.2": 0.4},
    "spiral": {"Q1.2": 0.8, "Q2.1": 0.4},
}
