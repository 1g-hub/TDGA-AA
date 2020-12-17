import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import os
from PIL import Image
from natsort import natsorted

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ComicDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.images = []
        self.targets = []
        self.transform = transform

        for label, title in enumerate(sorted(os.listdir(root))):
            title_path = os.path.join(root, title)
            for idx, img in enumerate(natsorted(os.listdir(title_path))):
                img_path = os.path.join(title_path, img)
                pil_img = Image.open(img_path)
                if split == "train" and idx < int(len(os.listdir(title_path))*0.9):
                    self.images.append(pil_img)
                    self.targets.append(label)
                elif split != "train" and idx >= int(len(os.listdir(title_path))*0.9):
                    self.images.append(pil_img)
                    self.targets.append(label)

    def __getitem__(self, index):
        img, target = self.images[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.images)
