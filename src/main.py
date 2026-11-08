"""Command-line entry point for the Galaxy Morphological Classification pipeline.

Usage examples
--------------
    python -m src.main prepare
    python -m src.main eda
    python -m src.main train --epochs 100
    python -m src.main evaluate
    python -m src.main predict --image path/to/galaxy.jpg

Run ``python -m src.main --help`` for the full list of commands.
"""

from __future__ import annotations

import argparse

from .config import EPOCHS
from .controller.pipeline import GalaxyPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Galaxy Morphological Classification (MVC)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="Load data, derive labels, build train/val split")
    sub.add_parser("eda", help="Generate exploratory visualisations")

    train_parser = sub.add_parser("train", help="Train the CNN classifier")
    train_parser.add_argument("--epochs", type=int, default=EPOCHS)

    sub.add_parser("evaluate", help="Evaluate the trained model")

    predict_parser = sub.add_parser("predict", help="Predict a single image's class")
    predict_parser.add_argument("--image", required=True, help="Path to a galaxy image")

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without opening interactive windows",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    show = not args.no_show
    pipeline = GalaxyPipeline()

    if args.command == "prepare":
        pipeline.prepare_data()
    elif args.command == "eda":
        pipeline.run_eda(show=show)
    elif args.command == "train":
        pipeline.train(epochs=args.epochs, show=show)
    elif args.command == "evaluate":
        pipeline.evaluate(show=show)
    elif args.command == "predict":
        pipeline.predict(args.image, show=show)


if __name__ == "__main__":
    main()
