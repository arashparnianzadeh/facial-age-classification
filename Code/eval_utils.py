# eval / reporting helpers
# accuracy + confusion matrix + timing so I can compare all 3 models fairly

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .config import FIGURES_DIR, LABEL_TO_NAME, METRICS_DIR


def evaluate_classifier(model, features, labels):
    # time the .predict call so I can report ms/image in the comparison table
    start = time.perf_counter()
    preds = model.predict(features)
    elapsed = time.perf_counter() - start

    # print(preds[:10])  # quick sanity check while debugging
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "avg_inference_ms_per_image": float((elapsed / max(len(labels), 1)) * 1000.0),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "classification_report": classification_report(
            labels,
            preds,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics


def model_size_mb(model_path):
    # size on disk in MB, for the comparison table
    return Path(model_path).stat().st_size / (1024.0 * 1024.0)


def save_metrics(metrics, filename):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / filename
    with out.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return out


def save_qualitative_grid(image_paths, true_labels, predicted_labels, filename, num_examples=8):
    # random 2x4 grid of test images w/ predictions for the report
    # seeded so it's the same every run
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(image_paths), size=min(num_examples, len(image_paths)), replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for ax, i in zip(axes, idx):
        img = plt.imread(image_paths[i])
        ax.imshow(img, cmap="gray")
        ax.set_title(
            f"Pred: {LABEL_TO_NAME[int(predicted_labels[i])]}\n"
            f"True: {LABEL_TO_NAME[int(true_labels[i])]}",
            fontsize=9,
        )
        ax.axis("off")

    # turn off any unused axes (safety if num_examples < 8)
    for ax in axes[len(idx):]:
        ax.axis("off")

    fig.tight_layout()
    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def save_model(model, model_path):
    # joblib is what sklearn recommends for this
    t = Path(model_path)
    t.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, t)
    return t
