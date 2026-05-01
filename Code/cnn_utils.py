# CNN pieces: dataset + model + device helper
# following the Lab 8 PyTorch pipeline pretty closely

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from .config import LABEL_TO_NAME


@dataclass(frozen=True)
class CnnSettings:
    # hyperparams — tried a few and these were fine
    image_size: tuple = (96, 96)
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # small L2, helps a tiny bit


class AgeGroupImageDataset(Dataset):
    # torch Dataset wrapper for the face images
    def __init__(self, image_paths, labels, image_size):
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.open(self.image_paths[i]).convert("RGB")
        img = img.resize(self.image_size, Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # HWC -> CHW for torch
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        label = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return tensor, label


class SimpleAgeCnn(nn.Module):
    """small CNN. 3 conv blocks then 2 FC layers.

    input 96x96 -> after 3 max-pools it's 12x12 so FC in is 128*12*12.
    nothing fancy, kept it close to the lab 8 structure.
    """

    def __init__(self, num_classes=len(LABEL_TO_NAME)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def device_for_training():
    # use GPU if there's one, otherwise CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
