# Age Group Detection

Computer vision pipeline for classifying facial images into four age groups (Child, Young, Middle-Aged, Senior). Compares hand-crafted feature approaches (HOG + SVM, HOG + MLP) with a small CNN, then applies the best-performing model to in-the-wild images using a Viola-Jones face detection pipeline.

## Models

- **HOG + Linear SVM** — `skimage.feature.hog` descriptors (9 orientations, 8×8 cells, 2×2 blocks, L2-Hys) on 96×96 grayscale crops, fed into a linear-kernel `sklearn.svm.SVC`.
- **HOG + MLP** — same HOG features into an `sklearn.neural_network.MLPClassifier` with two hidden layers (256, 128), ReLU activation, Adam optimiser.
- **Simple CNN** — three Conv→ReLU→MaxPool blocks (32 → 64 → 128 channels) followed by two fully-connected layers (256 → num_classes), trained on 96×96 RGB inputs with cross-entropy loss and Adam (lr 1e-3, weight_decay 1e-4) for 10 epochs. Best validation checkpoint is saved.

## Pipeline

1. **Load split** — read `train_labels.txt` / `test_labels.txt`, build a stratified 80/20 train/validation split (seed 42).
2. **Preprocess** — resize to 96×96, grayscale for HOG, RGB for the CNN, normalise to `[0, 1]`.
3. **Feature extraction / model forward pass** — HOG descriptors for the classical models, raw tensors for the CNN.
4. **Classify** — predict one of four age groups.
5. **In-the-wild inference** (`AgeDetection(path)` in [Code/age_detection.py](Code/age_detection.py)) — for each image in a folder, run OpenCV's Viola-Jones Haar cascade (`haarcascade_frontalface_default.xml`) to locate the largest face, crop with 15% padding, fall back to a centre-crop if no face is detected, then run the trained CNN. Plots a 2×2 grid of four randomly-sampled images with their predicted labels.

## Results

Test-set performance on the 850-image held-out split (numbers from [Code/results/metrics/all_model_comparison.csv](Code/results/metrics/all_model_comparison.csv)):

| Model              | Test accuracy | Inference (ms/image) | Model size |
|--------------------|---------------|----------------------|------------|
| HOG + Linear SVM   | 69.5%         | 17.87                | 188.7 MB   |
| HOG + MLP          | 72.0%         | 0.09                 | 17.5 MB    |
| Simple CNN         | **79.9%**     | 5.10                 | 18.4 MB    |

The CNN is the best model and is what `AgeDetection(path)` loads at inference time.

## Tech Stack

Python · NumPy · OpenCV · scikit-learn · scikit-image · PyTorch · Pillow · Matplotlib · joblib

## Notes on the dataset

The training dataset (~14,000 facial images across four age-group classes — 13,300 train / 850 test) was provided as part of coursework and is **not redistributed in this repository**. The `Personal_Dataset/` folder contains royalty-free images sourced from Pexels and Unsplash for in-the-wild testing of `AgeDetection(path)`.

The `Models/hog_svm.joblib` artefact (~189 MB) is also **not committed** — it exceeds GitHub's 100 MB per-file limit. It can be regenerated from the training data by running `python -m Code.train_hog_svm`. The HOG + MLP and CNN checkpoints are committed.

## Project structure

```
.
├── Code/                       # Python package: training, evaluation, inference
│   ├── age_detection.py        # AgeDetection(path) — coursework entry point
│   ├── cnn_utils.py            # SimpleAgeCnn + Dataset + device helper
│   ├── compare_all_models.py   # Builds CSV comparison across all 3 models
│   ├── compare_classical_models.py  # SVM-vs-MLP only comparison (legacy)
│   ├── config.py               # Paths, label map, hyperparameter constants
│   ├── data_utils.py           # Split loading, train/val split, image loader
│   ├── eval_utils.py           # Accuracy, confusion matrix, qualitative grid
│   ├── feature_utils.py        # HOG feature extractor
│   ├── test_age_detection.py   # Terminal sanity-check for AgeDetection()
│   ├── train_cnn.py            # CNN training loop
│   ├── train_hog_mlp.py        # HOG + MLP trainer
│   ├── train_hog_svm.py        # HOG + Linear SVM trainer
│   └── results/
│       ├── figures/            # Qualitative prediction grids (PNG)
│       └── metrics/            # Per-model JSON summaries + comparison CSV
├── Models/                     # Trained model checkpoints
│   ├── hog_mlp.joblib          # HOG + MLP, ~18 MB
│   └── simple_cnn_best.pt      # Best-val CNN checkpoint, ~19 MB
│   # hog_svm.joblib is gitignored — too large for GitHub; regenerate with train_hog_svm.py
├── Personal_Dataset/           # 12 royalty-free in-the-wild test images (Pexels/Unsplash)
├── test_function.ipynb         # Live demo notebook for AgeDetection(path)
├── .gitignore
└── README.md
```
