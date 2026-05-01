# HOG features for the classical baselines
# using skimage.feature.hog — covered in Lab 5

import numpy as np
from skimage.feature import hog


def extract_hog_features(images):
    # expects [N, H, W] grayscale images with values already in [0,1]
    # default HOG params (9 orientations, 8x8 cells, 2x2 blocks) — same as lab notes
    feats = []
    for img in images:
        d = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        feats.append(d.astype(np.float32))
    return np.stack(feats, axis=0)
