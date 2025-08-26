import os
import h5py
from torch.utils.data import Dataset
# from pathlib import Path
# from collections import defaultdict
# from scripts import utils


class ProtoDataset(Dataset):
    def __init__(self, dataset_path, feature: str, config):
        """
        Dataset object

        Params:
            dataset_path (str | Path): Path to dataset. Point this to either the
                Training Set or Validation Set.
            feature (str): Audio feature to be used for training/validation.
            config: config.yaml object.
        """

        assert (feature in ["pcen", "melspec", "logmel"],
                "Invalid feature. Must be either 'pcen', 'melspec', or 'logmel'")
        self.dataset_path = dataset_path
        self.feature = feature
        self.train_set = os.path.join(self.dataset_path,
                                      f"{feature}.h5")
        self.unique_labels, self.labels = self.get_labels()
        self.unique_labels = [label.decode() for label in self.unique_labels]

        # dict approach
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

    def get_labels(self):
        labels = list(h5py.File(self.train_set, 'r')['labels'])
        unique_labels = list(h5py.File(self.train_set,
                                       'r')['unique_labels'])
        return unique_labels, labels
