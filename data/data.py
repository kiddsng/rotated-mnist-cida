import torch
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.transforms.functional import rotate
from torchvision.datasets import MNIST

import numpy as np


class RotatedMNIST(Dataset):
    """Create the Rotated MNIST dataset"""

    def __init__(self, rotation_range):
        transform = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        self.data = MNIST(
            root='data',
            train=True,
            download=True,
            transform=transform
        )
        self.rotation_range = rotation_range

    def __getitem__(self, index):
        image, label = self.data[index]
        image = transforms.ToPILImage()(image)

        angle_min, angle_max = self.rotation_range
        angle = np.random.rand() * (angle_max - angle_min) + angle_min

        image = rotate(image, angle)
        image = transforms.ToTensor()(image).to(torch.float)
        angle = np.array([angle / 360.0], dtype=np.float32)
        domain = angle

        return image, label, angle, domain

    def __len__(self):
        return len(self.data)
