import torch
import torch.nn.functional as F
import torch.nn as nn


class ProtoNet(torch.nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 num_prototypes: int,
                 n_way: int,
                 k_shot: int,
                 embedding_dim: int = None,
                 prototypes=None,
                 squared: bool = True,
                 dist: str = "euclidean",
                 normalize: bool = False,
                 device: str = "cuda"):
        """
        Prototypical Network layer. Insert with feature embedding after encoder.

        Args:
            encoder (nn.Module): feature extracting network
            n_prototypes (int): number of prototypes to use
            n_ways (int): Number of class in a single batch
            k_shot (int): Number of example per class
            embedding_dim (int): dimension of the embedding space
            prototypes (tensor): Prototype tensor of shape (n_prototypes x embedding_dim),
            squared (bool): Whether to use the squared Euclidean distance or not
            dist (str): default 'euclidean', other possibility 'cosine'
            normalize (bool): l2 normalization of the features
            device (str): device on which to declare the prototypes (cpu/cuda)
        """
        super(ProtoNet, self).__init__()
        self.encoder = encoder
        self.num_prototypes = num_prototypes
        self.squared = squared
        self.dist = dist
        self.normalize = normalize
        self.n_way = n_way
        self.k_shot = k_shot
        # self.prototypes = (nn.Parameter(torch.rand((num_prototypes, embedding_dim), device=device)).requires_grad_(True)
        #                    if prototypes is None else nn.Parameter(prototypes).requires_grad_(False))

    def forward(self, batch):
        support_set, support_labels, query_set, query_labels = batch
        if torch.cuda.is_available():
            support_set = support_set.cuda()
            query_set = query_set.cuda()
            support_labels = support_labels.cuda()
            query_labels = query_labels.cuda()

        support_set = support_set.unsqueeze(1)
        query_set = query_set.unsqueeze(1)

        sup_embedding = self.encoder(support_set)
        query_embedding = self.encoder(query_set)

        if self.normalize:
            sup_embedding = F.normalize(sup_embedding, dim=1)

        sup_embedding = sup_embedding.squeeze(1)

        prototypes = self.get_prototypes(sup_embedding)

        dist = torch.cdist(query_embedding, prototypes)

        return -dist, query_labels

    def get_prototypes(self, support):
        prototypes = support.reshape(self.n_way, self.k_shot,
                                     -1).mean(dim=1)
        return prototypes

    def get_distance(self, query, prototypes):
        if (len(query.shape) >= 3) and (query.size(1) == 1):
            query.squeeze(1)

        if (len(prototypes.shape) >= 3) and (prototypes.size(1) == 1):
            query.squeeze(1)

        assert (len(query.shape) == 2 and len(prototypes.shape) == 2,
                "Shape of query embedding or prototypes is more than 3")

        query_class = query.size(0)
        proto_class = prototypes.size(0)
        query_dim = query.size(1)
        assert (query_dim == prototypes.size(1),
                "Feature maps of query embedding and prototype does not match")

        query = query.unsqueeze(1).expand(query_class, proto_class,
                                          query_dim)
        prototypes = prototypes.unsqueeze(0).expand(query_class,
                                                    proto_class,
                                                    query_dim)

        return torch.pow(query - prototypes, 2).sum(2)
