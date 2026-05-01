# Model 3: small CNN trained on resized RGB face images
# following the Lab 8 PyTorch training loop pattern

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from Code.cnn_utils import AgeGroupImageDataset, CnnSettings, SimpleAgeCnn, device_for_training
    from Code.config import LABEL_TO_NAME, METRICS_DIR, MODELS_DIR
    from Code.data_utils import load_dataset_split, make_train_val_split
    from Code.eval_utils import save_metrics, save_qualitative_grid
else:
    from .cnn_utils import AgeGroupImageDataset, CnnSettings, SimpleAgeCnn, device_for_training
    from .config import LABEL_TO_NAME, MODELS_DIR
    from .data_utils import load_dataset_split, make_train_val_split
    from .eval_utils import save_metrics, save_qualitative_grid

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader


def run_epoch(model, dataloader, criterion, optimizer, device, train):
    # one pass over the loader, either training or evaluating
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in dataloader:
        inputs  = inputs.to(device)
        targets = targets.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

    avg_loss = total_loss / max(len(dataloader.dataset), 1)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, float(acc)


def predict_dataset(model, dataloader, device):
    # run through the whole loader once, collecting preds + timing for ms/image
    model.eval()
    preds = []
    targets_all = []

    start = time.perf_counter()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds.extend(outputs.argmax(dim=1).cpu().tolist())
            targets_all.extend(targets.tolist())
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / max(len(targets_all), 1)) * 1000.0
    return preds, targets_all, float(avg_ms)


def main():
    settings = CnnSettings()
    device = device_for_training()
    # print("device:", device)  # left this in while testing on my laptop

    full_train = load_dataset_split("train")
    test_split = load_dataset_split("test")
    train_split, val_split = make_train_val_split(full_train, val_size=0.2)

    train_dataset = AgeGroupImageDataset(train_split.image_paths, train_split.labels, settings.image_size)
    val_dataset   = AgeGroupImageDataset(val_split.image_paths,   val_split.labels,   settings.image_size)
    test_dataset  = AgeGroupImageDataset(test_split.image_paths,  test_split.labels,  settings.image_size)

    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=settings.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=settings.batch_size, shuffle=False)

    model = SimpleAgeCnn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )

    # keep per-epoch history so I can plot/report later if needed
    history = []
    best_val_acc = -1.0
    best_model_path = MODELS_DIR / "simple_cnn_best.pt"

    for epoch in range(settings.num_epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        history.append({
            "epoch": epoch + 1,
            "train_loss": tr_loss,
            "train_accuracy": tr_acc,
            "val_loss": vl_loss,
            "val_accuracy": vl_acc,
        })

        # checkpoint whenever val acc improves (simple early-stopping-ish idea)
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": settings.image_size,
                    "label_to_name": LABEL_TO_NAME,
                },
                best_model_path,
            )

    # reload best weights before the test-set eval
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_preds, test_targets, avg_inference_ms = predict_dataset(model, test_loader, device=device)

    qualitative_path = save_qualitative_grid(
        image_paths=test_split.image_paths,
        true_labels=test_split.labels,
        predicted_labels=test_preds,
        filename="cnn_test_examples.png",
    )

    summary = {
        "model_name": "Simple CNN",
        "image_size": list(settings.image_size),
        "device": str(device),
        "train_samples": int(len(train_dataset)),
        "validation_samples": int(len(val_dataset)),
        "test_samples": int(len(test_dataset)),
        "num_epochs": settings.num_epochs,
        "batch_size": settings.batch_size,
        "learning_rate": settings.learning_rate,
        "best_model_path": str(best_model_path),
        "model_size_mb": best_model_path.stat().st_size / (1024.0 * 1024.0),
        "history": history,
        "validation_best_accuracy": best_val_acc,
        "test": {
            "accuracy": float(accuracy_score(test_targets, test_preds)),
            "avg_inference_ms_per_image": avg_inference_ms,
            "confusion_matrix": confusion_matrix(test_targets, test_preds).tolist(),
            "classification_report": classification_report(
                test_targets,
                test_preds,
                output_dict=True,
                zero_division=0,
            ),
        },
        "qualitative_figure": str(qualitative_path),
    }

    save_metrics(summary, "cnn_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
