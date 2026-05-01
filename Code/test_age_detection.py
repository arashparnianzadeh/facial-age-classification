# quick terminal sanity check for AgeDetection(path)
# just runs it on the Personal_Dataset folder and prints the predictions
# (the real demo is in test_function.ipynb for the presentation)

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from Code.age_detection import AgeDetection
    from Code.config import FIGURES_DIR, PROJECT_ROOT
else:
    from .age_detection import AgeDetection
    from .config import FIGURES_DIR, PROJECT_ROOT


def main():
    personal_dataset = PROJECT_ROOT / "Personal_Dataset"
    results = AgeDetection(personal_dataset, save_figure=True)
    print(json.dumps(results, indent=2))
    print(f"Saved figure to {FIGURES_DIR / 'age_detection_demo.png'}")


if __name__ == "__main__":
    main()
