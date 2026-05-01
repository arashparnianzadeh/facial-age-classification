# shared paths + constants
# kept this file small on purpose so I can tweak things in one place

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "CW_Dataset"
MODELS_DIR   = PROJECT_ROOT / "Models"
RESULTS_DIR  = PROJECT_ROOT / "Code" / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
METRICS_DIR  = RESULTS_DIR / "metrics"

# label map from CW_Dataset/README.txt
LABEL_TO_NAME = {
    0: "Child",
    1: "Young",
    2: "Middle-Aged",
    3: "Senior",
}

# 96x96 seems to work fine for both HOG and CNN so reusing it everywhere
HOG_IMAGE_SIZE = (96, 96)
RANDOM_STATE = 42
