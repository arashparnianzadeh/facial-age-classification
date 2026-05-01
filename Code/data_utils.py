# dataset loading + preprocessing
# mostly adapted from Lab 6 (the train_test_split pattern is basically from there)
# label file format is "filename label" per line, see CW_Dataset/README.txt

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from .config import DATASET_ROOT, RANDOM_STATE


@dataclass(frozen=True)
class DatasetSplit:
    # just a little container so I don't keep passing two arrays around
    image_paths: np.ndarray
    labels: np.ndarray


def _read_label_file(label_file: Path):
    # parse the "filename <space> label" pairs
    # skipping any blank/malformed lines just in case
    records = []
    with label_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 2:
                continue  # TODO: log a warning here maybe? ignoring for now
            records.append((parts[0], int(parts[1])))
    return records


def load_dataset_split(split_name, dataset_root=None):
    # split_name should be "train" or "test"
    root = Path(dataset_root) if dataset_root is not None else DATASET_ROOT
    split_dir = root / split_name
    label_file = split_dir / f"{split_name}_labels.txt"

    records = _read_label_file(label_file)
    image_paths = np.array([str(split_dir / f) for f, _ in records], dtype=object)
    labels = np.array([lbl for _, lbl in records], dtype=np.int64)
    return DatasetSplit(image_paths=image_paths, labels=labels)


def make_train_val_split(train_split, val_size=0.2, random_state=RANDOM_STATE):
    # stratified split so every age group shows up in val too
    # (Lab 6 did this with stratify=y)
    tr_p, val_p, tr_y, val_y = train_test_split(
        train_split.image_paths,
        train_split.labels,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
        stratify=train_split.labels,
    )
    return DatasetSplit(tr_p, tr_y), DatasetSplit(val_p, val_y)


def load_image(image_path, image_size, grayscale=True):
    # images in the dataset have different sizes so resize to a fixed one
    # grayscale=True for HOG (needs single channel), False for CNN
    img = Image.open(image_path)
    img = img.convert("L" if grayscale else "RGB")
    img = img.resize(image_size, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def batch_load_images(image_paths: Iterable, image_size, grayscale=True):
    # just a loop, nothing fancy
    imgs = [load_image(p, image_size=image_size, grayscale=grayscale) for p in image_paths]
    return np.stack(imgs, axis=0)
