"""Controller layer — orchestrates the end-to-end workflow.

The controller is the only layer that knows about *both* the model and the view.
It wires data preparation, model building, training, evaluation, and prediction
together, delegating all domain work to :mod:`model` and all presentation to
:mod:`view`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from .. import config
from ..model import cnn_model, data_model
from ..view import visualization


class GalaxyPipeline:
    """End-to-end controller for the galaxy classification workflow."""

    def __init__(self) -> None:
        self.model = None

    # -- Data ---------------------------------------------------------------
    def prepare_data(self) -> None:
        """Load solutions, derive labels, and build the train/val dataset."""
        df = data_model.load_solutions()
        galaxy_ids = data_model.filter_labels(df)
        data_model.build_dataset(galaxy_ids)

        print("Train counts:", data_model.count_images(config.TRAIN_DIR))
        print("Validation counts:", data_model.count_images(config.VALIDATION_DIR))

    def run_eda(self, show: bool = True) -> None:
        """Produce exploratory visualisations (correlation, samples, PCA)."""
        df = data_model.load_solutions()
        visualization.plot_correlation_matrix(
            df, save_path=config.OUTPUTS_DIR / "correlation_matrix.png", show=show
        )

        for class_name in config.CLASSES:
            visualization.show_sample_images(
                config.TRAIN_DIR / class_name,
                save_path=config.OUTPUTS_DIR / f"samples_{class_name}.png",
                show=show,
            )

        self._pca_visualization(show=show)

    def _pca_visualization(self, show: bool = True) -> None:
        """Flatten images, reduce with PCA, and plot the first two components."""
        train_arrays = [
            data_model.load_flattened_images(config.TRAIN_DIR / c)
            for c in config.CLASSES
        ]
        val_arrays = [
            data_model.load_flattened_images(config.VALIDATION_DIR / c)
            for c in config.CLASSES
        ]
        x_train = np.concatenate(train_arrays, axis=0)
        x_val = np.concatenate(val_arrays, axis=0)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)

        pca = PCA(n_components=100)
        x_train_pca = pca.fit_transform(x_train_scaled)
        x_val_pca = pca.transform(x_val_scaled)

        visualization.plot_pca(
            x_train_pca, "PCA Visualization - Training Data", color="blue",
            save_path=config.OUTPUTS_DIR / "pca_train.png", show=show,
        )
        visualization.plot_pca(
            x_val_pca, "PCA Visualization - Validation Data", color="red",
            save_path=config.OUTPUTS_DIR / "pca_validation.png", show=show,
        )

    # -- Training -----------------------------------------------------------
    def train(self, epochs: int = config.EPOCHS, show: bool = True):
        """Build the model, train it, plot curves, and persist it."""
        train_generator, validation_generator = data_model.build_generators()

        self.model = cnn_model.build_model()
        self.model.summary()

        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            verbose=1,
        )

        visualization.plot_training_history(
            history, save_path=config.OUTPUTS_DIR / "training_history.png", show=show
        )
        cnn_model.save_model(self.model)
        return history

    # -- Evaluation ---------------------------------------------------------
    def evaluate(self, show: bool = True) -> dict:
        """Evaluate the trained model on the validation set."""
        model = self._ensure_model()
        _, validation_generator = data_model.build_generators()

        predictions = model.predict(validation_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = validation_generator.classes
        class_labels = list(validation_generator.class_indices.keys())

        conf_matrix = confusion_matrix(true_classes, predicted_classes)
        report = classification_report(
            true_classes, predicted_classes, target_names=class_labels
        )
        metrics = {
            "accuracy": accuracy_score(true_classes, predicted_classes),
            "precision": precision_score(
                true_classes, predicted_classes, average="weighted"
            ),
            "recall": recall_score(
                true_classes, predicted_classes, average="weighted"
            ),
            "f1": f1_score(true_classes, predicted_classes, average="weighted"),
        }

        print("Confusion Matrix:\n", conf_matrix)
        print("\nClassification Report:\n", report)
        for name, value in metrics.items():
            print(f"{name.capitalize()}: {value:.4f}")

        visualization.plot_confusion_matrix(
            conf_matrix, class_labels,
            save_path=config.OUTPUTS_DIR / "confusion_matrix.png", show=show,
        )
        return metrics

    # -- Prediction ---------------------------------------------------------
    def predict(self, image_path: Path, show: bool = True) -> str:
        """Predict the morphological class of a single galaxy image."""
        model = self._ensure_model()
        batch = data_model.load_single_image(image_path)
        prediction = model.predict(batch)
        predicted_class = config.CLASSES[int(np.argmax(prediction))]

        print(f"Predicted class for {image_path}: {predicted_class}")
        visualization.show_prediction(image_path, predicted_class, show=show)
        return predicted_class

    # -- Helpers ------------------------------------------------------------
    def _ensure_model(self):
        """Return a loaded model, loading from disk if not already in memory."""
        if self.model is None:
            self.model = cnn_model.load_trained_model()
        return self.model
