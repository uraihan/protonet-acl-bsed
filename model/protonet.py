import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class ProtoNet(torch.nn.Module):
    def __init__(self,
                 encoder,
                 num_prototypes: int,
                 embedding_dim: int,
                 prototypes=None,
                 squared: bool = True,
                 dist: str = "euclidean",
                 normalize: bool = False,
                 device: str = "cuda"):
        """
        Prototypical Network layer. Insert with feature embedding after encoder.

        Args:
            model (nn.Module): feature extracting network
            n_prototypes (int): number of prototypes to use
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
        self.prototypes = (nn.Parameter(torch.rand((num_prototypes, embedding_dim), device=device)).requires_grad_(True)
                           if prototypes is None else nn.Parameter(prototypes).requires_grad_(False))

    def forward(self, data):
        embedding = self.encoder(data)

        if self.normalize:
            embedding = F.normalize(embedding, dim=1)

        batch, channel, height, width = embedding.shape
        embedding = embedding.view(batch, channel, height*width) \
                             .transpose(1, 2) \
                             .contiguous() \
                             .view(batch*height*width, channel)

        if self.dist == "euclidean":
            dist = torch.norm(embedding[:, None, :] -
                              self.prototypes[None, :, :], dim=-1)
        if self.dist == "cosine":
            dist = 1 - nn.CosineSimilarity(dim=-1)(embedding[:, None, :],
                                                   self.prototypes[None, :, :])

        if self.squared:
            dist = dist ** 2

        dist = dist.view(batch, height * width, self.n_prototypes) \
                   .transpose(1, 2) \
                   .contiguous() \
                   .view(batch, self.n_prototypes, height, width)

        return -dist
