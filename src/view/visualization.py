"""View layer — all plots and figures.

These functions only present data; they contain no training or data-preparation
logic. Each accepts an optional ``save_path`` so figures can be written to disk
(e.g. into ``outputs/``) for the README.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.preprocessing import image as keras_image

from .. import config


def _finish(save_path: Optional[Path], show: bool) -> None:
    """Shared save/show/close handling for a figure."""
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_correlation_matrix(
    df, save_path: Optional[Path] = None, show: bool = True
) -> None:
    """Heatmap of the feature correlation matrix."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title("Correlation Matrix")
    _finish(save_path, show)


def show_sample_images(
    class_dir: Path,
    n: int = 3,
    img_size=config.IMG_SIZE,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Display the first ``n`` images from a class directory."""
    files = os.listdir(class_dir)[:n]
    plt.figure(figsize=(4 * n, 4))
    for i, filename in enumerate(files):
        img = keras_image.load_img(Path(class_dir) / filename, target_size=img_size)
        plt.subplot(1, n, i + 1)
        plt.axis("off")
        plt.imshow(img)
    _finish(save_path, show)


def plot_pca(
    points: np.ndarray,
    title: str,
    color: str = "blue",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Scatter plot of the first two PCA components."""
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], c=color, label=title)
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    _finish(save_path, show)


def plot_training_history(
    history, save_path: Optional[Path] = None, show: bool = True
) -> None:
    """Plot training/validation accuracy and loss curves side by side."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(acc))

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    _finish(save_path, show)


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_labels: List[str],
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Annotated heatmap of a confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    _finish(save_path, show)


def show_prediction(
    image_path: Path,
    predicted_class: str,
    img_size=config.IMG_SIZE,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Display a single image annotated with its predicted class."""
    img = keras_image.load_img(image_path, target_size=img_size)
    plt.imshow(img)
    plt.title(f"Predicted Class: {predicted_class}")
    plt.axis("off")
    _finish(save_path, show)
