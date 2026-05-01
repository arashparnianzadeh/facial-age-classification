# builds the big comparison CSV across all 3 models for the report
# just reads the per-model *_summary.json files and flattens a few fields

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
    "cnn_summary.json",
]


def _load_summary(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _val_accuracy(summary):
    # classical models store val accuracy under "validation", CNN under "validation_best_accuracy"
    if "validation" in summary:
        return float(summary["validation"]["accuracy"])
    return float(summary["validation_best_accuracy"])


def _test_accuracy(summary):
    return float(summary["test"]["accuracy"])


def _inference_ms(summary):
    return float(summary["test"]["avg_inference_ms_per_image"])


def _model_path(summary):
    # classical: "model_path", CNN: "best_model_path"
    return summary.get("model_path", summary.get("best_model_path", ""))


def main():
    rows = []
    for filename in SUMMARY_FILES:
        summary_path = METRICS_DIR / filename
        if not summary_path.exists():
            # e.g. run this before one of the trainers finished — just skip
            continue
        summary = _load_summary(summary_path)
        rows.append({
            "model_name": summary["model_name"],
            "image_size": "x".join(str(v) for v in summary["image_size"]),
            "validation_accuracy": _val_accuracy(summary),
            "test_accuracy": _test_accuracy(summary),
            "avg_inference_ms_per_image": _inference_ms(summary),
            "model_size_mb": summary["model_size_mb"],
            "model_path": _model_path(summary),
            "qualitative_figure": summary["qualitative_figure"],
        })

    output_path = METRICS_DIR / "all_model_comparison.csv"
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
