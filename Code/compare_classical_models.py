# older version that only compared SVM vs MLP
# kept around from before the CNN was added — still useful if I just want the classical table

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from Code.config import METRICS_DIR
else:
    from .config import METRICS_DIR


SUMMARY_FILES = [
    "hog_svm_summary.json",
    "hog_mlp_summary.json",
]


def _load_summary(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    rows = []
    for filename in SUMMARY_FILES:
        summary_path = METRICS_DIR / filename
        if not summary_path.exists():
            continue
        summary = _load_summary(summary_path)
        rows.append({
            "model_name": summary["model_name"],
            "image_size": "x".join(str(v) for v in summary["image_size"]),
            "validation_accuracy": summary["validation"]["accuracy"],
            "test_accuracy": summary["test"]["accuracy"],
            "avg_inference_ms_per_image": summary["test"]["avg_inference_ms_per_image"],
            "model_size_mb": summary["model_size_mb"],
            "model_path": summary["model_path"],
            "qualitative_figure": summary["qualitative_figure"],
        })

    output_path = METRICS_DIR / "classical_model_comparison.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name",
                "image_size",
                "validation_accuracy",
                "test_accuracy",
                "avg_inference_ms_per_image",
                "model_size_mb",
                "model_path",
                "qualitative_figure",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote comparison table to {output_path}")


if __name__ == "__main__":
    main()
