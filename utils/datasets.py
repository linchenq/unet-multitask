import os
import numpy as np
import random
import hdf5storage
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from cfgs.config import cfg

class SpineSegDataset:
    def __init__(self, list_path):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        mat_data = hdf5storage.loadmat(img_path, options=hdf5storage.Options(matlab_compatible=True))

        image = np.zeros((cfg.IN_CH, cfg.H, cfg.W))
        if mat_data['I'].ndim == 2:
            for i in range(cfg.IN_CH):
                image[i, ...] = mat_data['I']
        else:
            raise NotImplementedError

        mask = np.zeros((cfg.SEG.OUT_CH, cfg.H, cfg.W))
        for i in range(cfg.SEG.OUT_CH):
            mask[i, ...] = mat_data[cfg.SEG.REP[i]]

        return image, mask

class SpineLocDataset:
    def __init__(self, list_path):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = transforms.ToTensor()(Image.open(img_path).convert('L'))
        _, w, h = img.shape

        # # Handle images with less than three channels
        # if len(img.shape) != 3:
        #     img = img.unsqueeze(0)
        #     img = img.expand((3, img.shape[1:]))

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

            # Extract coordinates for unpadded + unscaled image
            x1 = w * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h * (boxes[:, 2] + boxes[:, 4] / 2)

            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2.) / w
            boxes[:, 2] = ((y1 + y2) / 2.) / h
            boxes[:, 3] *= w / w
            boxes[:, 4] *= h / h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        return img, targets

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))

        targets = [boxes for boxes in targets if boxes is not None]

        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)

        imgs = torch.stack(imgs)

        return imgs, targets

class SpineLoadImageDataset:
    def __init__(self, list_path):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = transforms.ToTensor()(Image.open(img_path).convert('L'))
        img_name = img_path.split("/")[-1].split(".")[0]

        return img, img_name






