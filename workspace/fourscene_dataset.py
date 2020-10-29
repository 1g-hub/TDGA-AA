import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import os
from PIL import Image
from natsort import natsorted
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FourSceneDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.stories = []
        self.targets = []
        self.transform = transform

        for title_idx, title in enumerate(os.listdir(root)):
            title_path = os.path.join(root, title)
            num_data = len(os.listdir(title_path)) // 4
            img_list = natsorted(os.listdir(title_path))
            for idx, img in enumerate(img_list):
                img_path = os.path.join(title_path, img)
                pil_img = Image.open(img_path)

                # label 0: 入れ替えなし
                if idx % 4 == 0:
                    story_0 = []

                story_0.append(pil_img)

                if idx % 4 == 3:
                    if split == "train" and idx//4 <= int(num_data*0.9):
                        self.stories.append(story_0)
                        self.targets.append(0)

                    elif split != "train" and idx//4 > int(num_data*0.9):
                        self.stories.append(story_0)
                        self.targets.append(0)

                # label 1: 同タイトルと入れ替え
                if idx % 4 == 0:
                    story_1 = []

                story_1.append(pil_img)

                if idx % 4 == 3:
                    other_img = img_list[(idx+4) % num_data]  # 次の画像
                    other_img_path = os.path.join(title_path, other_img)
                    other_pli_img = Image.open(other_img_path)
                    story_1[-1] = other_pli_img

                    if split == "train" and idx // 4 <= int(num_data * 0.9):
                        self.stories.append(story_1)
                        self.targets.append(1)

                    elif split != "train" and idx // 4 > int(num_data * 0.9):
                        self.stories.append(story_1)
                        self.targets.append(1)

                # label 2: 異タイトルと入れ替え
                if idx % 4 == 0:
                    story_2 = []

                story_2.append(pil_img)

                if idx % 4 == 3:
                    titles = sorted(os.listdir(root))
                    other_title = random.choice(titles[:title_idx]+titles[title_idx+1:])
                    other_title_path = os.path.join(root, other_title)
                    other_title_image_path = os.path.join(other_title_path, random.choice(os.listdir(other_title_path)))
                    other_title_pil_img = Image.open(other_title_image_path)
                    story_2[-1] = other_title_pil_img

                    if split == "train" and idx // 4 <= int(num_data * 0.9):
                        self.stories.append(story_2)
                        self.targets.append(2)

                    elif split != "train" and idx // 4 > int(num_data * 0.9):
                        self.stories.append(story_2)
                        self.targets.append(2)

    def __getitem__(self, index):
        story, target = self.stories[index], self.targets[index]

        if self.transform is not None:
            t = random.choice(self.transform.transforms)
            if not isinstance(t, list):  # ベース拡張
                t = self.transform

            for i in range(4):
                story[i] = t(story[i])
                story[i] = story[i].view((1,)+story[i].size())
        return torch.cat(story, dim=0), target

    def __len__(self):
        return len(self.stories)
