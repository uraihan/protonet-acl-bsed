import torch
import random
import numpy as np
from collections import defaultdict


class FewShotSampler:
    def __init__(self, dataset_labels,
                 n_way: int,
                 k_shot: int,
                 include_query: bool = False,
                 shuffle: bool = True):
        self.dataset_labels = dataset_labels
        self.n_way = n_way
        self.include_query = include_query
        self.shuffle = shuffle
        if self.include_query:
            self.k_shot = k_shot * 2
        else:
            self.k_shot = k_shot

        self.batch_size = self.n_way * self.k_shot

        self.classes = torch.unique(self.dataset_labels).tolist()
        self.num_classes = len(self.classes)
        self.indices_per_class = {}
        self.batches_per_class = {}
        for c in self.classes:
            self.indices_per_class[c] = torch.where(self.dataset_labels ==
                                                    c)[0]
            self.batches_per_class[c] = self.indices_per_class[c].shape[0] // self.k_shot

        self.iterations = sum(self.batches_per_class.values()) // self.n_way
        self.class_list = [c for c in self.classes for _ in range(
            self.batches_per_class[c])]

        if self.shuffle:
            self.shuffle_data()
        else:
            sort_idxs = [
                i + p * self.num_classes
                for i, c in enumerate(self.classes)
                for p in range(self.batches_per_class[c])
            ]
            self.class_list = np.array(self.class_list)[
                np.argsort(sort_idxs)].tolist()

    def __iter__(self):
        start_idx = defaultdict(int)
        for it in range(self.iterations):
            # Select N classes for the batch
            class_batch = self.class_list[it *
                                          self.n_way: (it + 1) * self.n_way]
            index_batch = []
            for c in class_batch:
                index_batch.extend(
                    self.indices_per_class[c][start_idx[c]: start_idx[c] +
                                              self.k_shot])
                start_idx[c] += self.k_shot

            if self.include_query:
                index_batch = index_batch[::2] + index_batch[1::2]

            yield index_batch

    def __len__(self):
        return self.iterations

    def shuffle_data(self):
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]

        random.shuffle(self.class_list)
