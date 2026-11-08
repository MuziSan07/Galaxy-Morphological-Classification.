# Galaxy Morphological Classification

> A convolutional neural network that classifies galaxies into **Spiral**, **Elliptical**, and **Lenticular** types from telescope images — automating a task astronomers traditionally do by hand.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](notebooks/galaxy_classification.ipynb)

<p align="center">
  <img src="assets/poster.png" alt="Galaxy Morphological Classification Poster" width="600"/>
</p>

---

## Overview

Galaxy morphological classification is the process of grouping galaxies by their **shape and structure**. These shapes carry deep physical meaning — they help astrophysicists study:

- **Galactic evolution** — how galaxies form and change over time
- **The large-scale structure of the universe**
- **Star formation** history and rates
- **Cosmic evolution** and the role of **dark matter**

Doing this by hand is slow and labour-intensive. This project trains a **Convolutional Neural Network (CNN)** to classify galaxies automatically into the three main morphological types:

| Type | Description |
|------|-------------|
| **Spiral** | Rotating disks with spiral arms |
| **Elliptical** | Smooth, featureless ellipsoids |
| **Lenticular** | Disk galaxies without prominent spiral arms (between spiral and elliptical) |

---

## Dataset

- **Source:** [Galaxy Zoo](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) (Kaggle)
- The raw labels were **cleaned and filtered** down to the three main galaxy types (this step requires domain knowledge of galaxy morphology).
- **Data augmentation** was applied to expand the training set and improve generalisation.

> **Note:** The dataset is not included in this repository due to its size. Download it from the link above and place it in a `data/` folder before running the notebook.

---

## Method

1. **Data cleaning & label filtering** — isolate Spiral, Elliptical, and Lenticular classes.
2. **Exploratory Data Analysis (EDA)** — understand class balance and image characteristics.
3. **Data augmentation** — rotations, flips, and shifts to enlarge the training set.
4. **CNN built from scratch** — a custom convolutional architecture (no transfer learning).

**Training configuration:**

| Setting | Value |
|---------|-------|
| Loss function | Categorical Cross-Entropy |
| Optimizer | Adam |
| Metric | Accuracy |
| Epochs | 100 |

---

## Results

The model achieved **~84% accuracy** on the classification task.

> _Tip: Add your confusion matrix and accuracy/loss curves here — export them from the notebook into `assets/` and embed them like the poster above._

---

## Getting Started

### Prerequisites

- Python 3.9+
- (Recommended) a GPU for faster training

### Installation

```bash
# Clone the repository
git clone https://github.com/MuziSan07/Galaxy-Morphological-Classification.git
cd Galaxy-Morphological-Classification

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

1. Download the [Galaxy Zoo dataset](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) and place it in a `data/` folder.

**Option A — Jupyter notebook** (interactive, with inline plots):

```bash
jupyter notebook notebooks/galaxy_classification.ipynb
```

**Option B — Command-line pipeline** (modular MVC package in `src/`):

```bash
python -m src.main prepare              # build train/validation split from raw data
python -m src.main eda                  # generate exploratory figures into outputs/
python -m src.main train --epochs 100   # train the CNN and save the model
python -m src.main evaluate             # confusion matrix + classification report
python -m src.main predict --image path/to/galaxy.jpg
```

> Review the paths in [`src/config.py`](src/config.py) before running. The defaults expect the extracted Galaxy Zoo files in `data/raw/`.

---

## Project Structure

The `src/` package follows the **Model–View–Controller (MVC)** pattern:

```
Galaxy-Morphological-Classification/
├── src/
│   ├── config.py                     # Paths and hyperparameters
│   ├── main.py                       # CLI entry point
│   ├── model/                        # MODEL: data + CNN (domain logic)
│   │   ├── data_model.py             #   loading, labelling, generators
│   │   └── cnn_model.py              #   CNN architecture + persistence
│   ├── view/                         # VIEW: presentation
│   │   └── visualization.py          #   all plots and figures
│   └── controller/                   # CONTROLLER: orchestration
│       └── pipeline.py               #   prepare -> train -> evaluate -> predict
├── notebooks/
│   └── galaxy_classification.ipynb   # Original exploratory notebook
├── assets/
│   ├── poster.png                    # Project poster
│   └── intro_video.mp4               # Short introduction video
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
└── README.md
```

**How the layers map to this project:**

| Layer | Responsibility | Files |
|-------|----------------|-------|
| **Model** | Data preparation and the CNN itself | `model/data_model.py`, `model/cnn_model.py` |
| **View** | Every plot and figure (no logic) | `view/visualization.py` |
| **Controller** | Wires model + view into a workflow | `controller/pipeline.py` |
| **Entry point** | Parses CLI commands | `main.py`, `config.py` |

---

## Future Work

- Extend the classifier to **more galaxy sub-types** (irregular, barred spiral, etc.).
- Experiment with **transfer learning** (ResNet, EfficientNet) to push accuracy higher.
- Add **model evaluation visuals** (confusion matrix, per-class precision/recall).
- Package the trained model behind a simple **web demo** for live predictions.

---

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Galaxy Zoo](https://www.galaxyzoo.org/) and the Kaggle community for the dataset.
- The astrophysics and open-source AI communities.
