import torch
import random
import numpy as np


class FewShotSampler:
    def __init__(self, dataset, n_way: int, k_shot: int, n_query: int):
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.dataset = dataset

        self.labels_list = torch.Tensor(self.dataset.get_labels())
        self.classes = torch.unique(self.labels_list).tolist()
        self.class_indices = {}
        for c in self.classes:
            self.class_indices[c] = torch.where(self.labels_list ==
                                                c)[0].tolist()

        self.iterations = len(self.labels_list) // (self.n_way * self.k_shot)

    def __iter__(self):
        for _ in range(self.iterations):
            # yield
            # torch.cat([torch.tensor(random.sample(self.items_per_label[label], self.n_shot + self.n_query))
            #           for label in random.sample(sorted(self.items_per_label.keys()), self.n_way)]).tolist()

            batch = []
            for label in random.sample(sorted(self.class_indices.keys()),
                                       self.n_way):
                selected_idx = random.sample(
                    self.class_indices[label], self.k_shot + self.n_query)
                batch.extend(selected_idx)

            yield batch

    def collate_fn(self, input_data):
        features, labels = zip(*input_data)
        features = torch.from_numpy(np.array(features)).unsqueeze(1)
        # labels = list(labels)

        support_set = []
        support_labels = []
        query_set = []
        query_labels = []
        for idx in range(0, len(labels), self.k_shot*2):
            support_set.extend(features[idx: idx+self.k_shot])
            support_labels.extend(labels[idx: idx+self.k_shot])
            query_set.extend(features[idx+self.k_shot: idx+self.k_shot*2])
            query_labels.exted(labels[idx+self.k_shot: idx+self.k_shot*2])

        return torch.cat(support_set), torch.Tensor(support_labels),
        torch.cat(query_set), torch.Tensor(query_labels)
