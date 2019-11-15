import os
import numpy as np
import hdf5storage
from torch.utils.data import Dataset

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

        return [image, mask]





