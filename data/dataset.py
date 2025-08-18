import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict
from scripts import utils


class ProtoDataset(Dataset):
    def __init__(self, feature, config):
        self.feature = feature
        self.train_set = os.path.join(Path.cwd(),
                                      config.dataset.devset,
                                      f"Training_Set/train_{feature}.h5")
        self.labels = h5py.File(self.train_set, 'r')['labels']
        self.unique_labels = h5py.File(self.train_set, 'r')['unique_labels']
        # self.label = list(h5py.File(self.train_set, 'r').keys())
        # self.label, self.label_map = utils.map_label_toint(self.label)

        self.dataset = None
        # x, y = self.get_samples(self.train_set)

    def __len__(self):
        return self.label.size

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.train_set, 'r')['features']
            # self.dataset = defaultdict()
            # for label in self.label:
            #     self.dataset[label] = np.array(h5py.File(self.train_set,
            #                                              'r').get(label))

        return self.dataset[idx], self.labels[idx]
