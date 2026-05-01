# Model 2: HOG features + MLP classifier
# reusing the same HOG features so the comparison with the SVM is fair
# MLP idea is from Lab 6

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from Code.config import HOG_IMAGE_SIZE, MODELS_DIR
    from Code.data_utils import batch_load_images, load_dataset_split, make_train_val_split
    from Code.eval_utils import evaluate_classifier, model_size_mb, save_metrics, save_model, save_qualitative_grid
    from Code.feature_utils import extract_hog_features
else:
    from .config import HOG_IMAGE_SIZE, MODELS_DIR
    from .data_utils import batch_load_images, load_dataset_split, make_train_val_split
    from .eval_utils import evaluate_classifier, model_size_mb, save_metrics, save_model, save_qualitative_grid
    from .feature_utils import extract_hog_features

from sklearn.neural_network import MLPClassifier


def train_hog_mlp(x_train, y_train):
    # 2 hidden layers, ReLU, adam — standard stuff
    # tried (512,) once but it overfit faster, (256,128) was more stable
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        batch_size=128,
        max_iter=50,
        random_state=42,
    )
    clf.fit(x_train, y_train)
    return clf


def main():
    full_train = load_dataset_split("train")
    test_split = load_dataset_split("test")
    train_split, val_split = make_train_val_split(full_train, val_size=0.2)

    train_images = batch_load_images(train_split.image_paths, image_size=HOG_IMAGE_SIZE, grayscale=True)
    val_images   = batch_load_images(val_split.image_paths,   image_size=HOG_IMAGE_SIZE, grayscale=True)
    test_images  = batch_load_images(test_split.image_paths,  image_size=HOG_IMAGE_SIZE, grayscale=True)

    x_train = extract_hog_features(train_images)
    x_val   = extract_hog_features(val_images)
    x_test  = extract_hog_features(test_images)

    model = train_hog_mlp(x_train, train_split.labels)
    model_path = save_model(model, MODELS_DIR / "hog_mlp.joblib")

    val_metrics  = evaluate_classifier(model, x_val,  val_split.labels)
    test_metrics = evaluate_classifier(model, x_test, test_split.labels)
    test_predictions = model.predict(x_test)

    qualitative_path = save_qualitative_grid(
        image_paths=test_split.image_paths,
        true_labels=test_split.labels,
        predicted_labels=test_predictions,
        filename="hog_mlp_test_examples.png",
    )

    summary = {
        "model_name": "HOG + MLP",
        "image_size": list(HOG_IMAGE_SIZE),
        "train_samples": int(len(train_split.labels)),
        "validation_samples": int(len(val_split.labels)),
        "test_samples": int(len(test_split.labels)),
        "model_path": str(model_path),
        "model_size_mb": model_size_mb(model_path),
        "validation": val_metrics,
        "test": test_metrics,
        "qualitative_figure": str(qualitative_path),
    }

    save_metrics(summary, "hog_mlp_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
