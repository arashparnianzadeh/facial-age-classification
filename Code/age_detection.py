# AgeDetection(path) — required entry point for the coursework
# loads the best saved CNN and predicts age group for 4 random images in a folder
# face detection bit uses the OpenCV Haar cascade (Viola-Jones) — same one from the labs

from __future__ import annotations

import random
import sys
from pathlib import Path

# same shim as the training scripts so this works both as package and direct
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from Code.cnn_utils import SimpleAgeCnn
    from Code.config import FIGURES_DIR, LABEL_TO_NAME, MODELS_DIR
else:
    from .cnn_utils import SimpleAgeCnn
    from .config import FIGURES_DIR, LABEL_TO_NAME, MODELS_DIR

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# cv2 might not be installed on every machine, so fall back gracefully
try:
    import cv2
except ImportError:
    cv2 = None  # no OpenCV -> skip face detection, just centre-crop


BEST_MODEL_PATH = MODELS_DIR / "simple_cnn_best.pt"

# cache the model so calling AgeDetection(path) multiple times doesn't reload it every time
_CACHED_MODEL = None
_CACHED_IMAGE_SIZE = None


def _load_checkpoint(model_path=BEST_MODEL_PATH):
    # loads once, returns cached copy after that
    global _CACHED_MODEL, _CACHED_IMAGE_SIZE

    if _CACHED_MODEL is not None:
        return _CACHED_MODEL, _CACHED_IMAGE_SIZE

    ckpt = torch.load(model_path, map_location="cpu")
    model = SimpleAgeCnn(num_classes=len(LABEL_TO_NAME))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _CACHED_MODEL = model
    _CACHED_IMAGE_SIZE = tuple(ckpt["image_size"])
    return _CACHED_MODEL, _CACHED_IMAGE_SIZE


def _preprocess_image(image_path, image_size):
    # try Viola-Jones face detection first
    # if it finds a face -> crop around it (with a little padding so we get the full head)
    # otherwise -> centre square crop
    # then resize to whatever size the model was trained on
    img = Image.open(image_path).convert("RGB")

    if cv2 is not None:
        bgr  = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        if len(faces) > 0:
            # biggest face in case there's more than one
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            # 15% padding — tried 10% and it felt too tight on foreheads/chins
            pad_x = int(0.15 * w)
            pad_y = int(0.15 * h)
            left   = max(x - pad_x, 0)
            top    = max(y - pad_y, 0)
            right  = min(x + w + pad_x, img.width)
            bottom = min(y + h + pad_y, img.height)
            img = img.crop((left, top, right, bottom))
        else:
            # no face found -> just centre crop
            W, H = img.size
            s = min(W, H)
            left = (W - s) // 2
            top  = (H - s) // 2
            img = img.crop((left, top, left + s, top + s))
    else:
        # same centre-crop path when cv2 isn't available
        W, H = img.size
        s = min(W, H)
        left = (W - s) // 2
        top  = (H - s) // 2
        img = img.crop((left, top, left + s, top + s))

    img = img.resize(image_size, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def predict_image(image_path):
    # single-image prediction, used by AgeDetection below
    model, image_size = _load_checkpoint()
    inputs = _preprocess_image(image_path, image_size)
    with torch.no_grad():
        outputs = model(inputs)
    label_index = int(outputs.argmax(dim=1).item())
    return label_index, LABEL_TO_NAME[label_index]


def AgeDetection(path, save_figure=True):
    # the function the coursework spec asks for: AgeDetection(path)
    # picks 4 random images from the given folder, shows them + predictions in a 2x2 grid
    folder = Path(path)
    image_paths = sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])

    if len(image_paths) < 4:
        raise ValueError("AgeDetection(path) requires at least 4 images in the folder.")

    chosen = random.sample(image_paths, k=4)
    results = []

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for ax, p in zip(axes, chosen):
        label_index, label_name = predict_image(p)
        img = plt.imread(p)
        ax.imshow(img)
        ax.set_title(f"{p.name}\nPrediction: {label_name}", fontsize=10)
        ax.axis("off")
        results.append({
            "image_path": str(p),
            "predicted_label_index": label_index,
            "predicted_label_name": label_name,
        })

    fig.tight_layout()

    if save_figure:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "age_detection_demo.png", dpi=150, bbox_inches="tight")

    plt.show()
    return results
