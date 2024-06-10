import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from skimage.color import rgb2lab
from PIL import Image
import utils


class Colorization(Dataset):
    def __init__(self, path, size, split="train"):
        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((size, size), Image.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        elif split == "test":
            self.transform = transforms.Compose(
                [transforms.Resize((size, size), Image.BICUBIC)]
            )
        self.split = split
        self.size = size
        self.path = path

    def __getitem__(self, index):
        img = Image.open(self.path[index]).convert("RGB")
        img = self.transform(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50.0 - 1.0  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.0  # Between -1 and 1
        return {"L": L, "ab": ab}

    def __len__(self):
        return len(self.path)


def datalaoder(dataset, BATCH_SIZE, shuffle):
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    return dataloader
