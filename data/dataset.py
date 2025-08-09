import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class ProtoDataset(Dataset):
    def __init__(self, feature, config):
        self.feature = feature
        self.train_set = os.path.join(Path.cwd(),
                                      config.dataset.devset,
                                      f"Training_Set/train_{feature}.h5")
        self.label = h5py.File(self.train_set, 'r')['label']

        self.dataset = None
        # x, y = self.get_samples(self.train_set)

    def __len__(self):
        return self.label.size

    # TODO: define function __getitem__ based on corresponding csv file
    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.train_set, 'r')["feature"]

        return self.dataset[idx], self.label[idx]
