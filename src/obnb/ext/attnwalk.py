"""AttentionWalk embeddings.

Implementation adapted from https://github.com/benedekrozemberczki/AttentionWalk

@article{abu2018watch,
  title={Watch your step: Learning node embeddings via graph attention},
  author={Abu-El-Haija, Sami and Perozzi, Bryan and Al-Rfou, Rami and Alemi, Alexander A},
  journal={Advances in neural information processing systems},
  volume={31},
  year={2018}
}

"""
from typing import Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from obnb import logger
from obnb.feature import FeatureVec
from obnb.graph import SparseGraph
from obnb.util.io import sparse_graph_to_nx

EPS = 1e-15


def attnwalk_embed(
    g: SparseGraph,
    *,
    dim: int = 128,
    walk_length: int = 80,
    window_size: int = 5,
    beta: float = 0.5,
    gamma: float = 0.5,
    epochs: int = 200,
    lr: float = 0.01,
    verbose: bool = False,
    device: str = "auto",
    as_array: bool = False,
    return_attn: bool = False,
):
    """Generate AttentionWalk embedding.

    Args:
        g: Target graph to embed.
        dim: Embedding dimension.
        walk_length: Random walk length.
        window_size: Wiindow size.
        beta: Attention l2 regularization parameter.
        gamma: Embedding l2 regularization parameter.
        epochs: Training epochs.
        lr: Learning rate.
        device: Compute device.
        verbose: Show training progress.
        as_array: If set to True, then return the embeddings as a 2-d numpy
            array (node x dim). Otherwise, return as a :class:`FeatureVec`
            object.
        return_attn: If set to True, then return attention as a 1-d numpy array
            in addition to the embeddings.

    """
    if isinstance(g, SparseGraph):
        nx_g = sparse_graph_to_nx(g)
    else:
        raise TypeError(f"Input graph must be SparseGraph, got {type(g)}: {g!r}")

    awe = AttentionWalkEmbedding(
        nx_g,
        dim=dim,
        walk_length=walk_length,
        window_size=window_size,
        beta=beta,
        gamma=gamma,
        epochs=epochs,
        lr=lr,
        verbose=verbose,
        device=device,
    )
    emb = awe.get_emb(as_numpy=True)

    if not as_array:
        emb = FeatureVec.from_mat(emb, list(g.node_ids))

    if return_attn:
        attn = awe.attn_weights.softmax(0).ravel().detach().cpu().numpy()
        return emb, attn
    else:
        return emb


def get_device(device: str) -> str:
    """Optionally auto get device."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


class AttentionWalkEmbedding(nn.Module):
    """Attention Walk Layer.

    Args:
        g: Target graph to embed.
        dim: Embedding dimension.
        walk_length: Random walk length.
        window_size: Wiindow size.
        beta: Attention l2 regularization parameter.
        gamma: Embedding l2 regularization parameter.
        epochs: Training epochs.
        lr: Learning rate.

    """

    def __init__(
        self,
        g: nx.Graph,
        *,
        dim: int = 128,
        walk_length: int = 80,
        window_size: int = 5,
        beta: float = 0.5,
        gamma: float = 0.5,
        epochs: int = 200,
        lr: float = 0.01,
        verbose: bool = False,
        device: str = "auto",
    ):
        super().__init__()
        self.g = g
        self.dim = dim
        self.walk_length = walk_length
        self.window_size = window_size
        self.beta = beta
        self.gamma = gamma
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.device = get_device(device)

        self.prepare_features()
        self.define_weights()
        self.initialize_weights()
        self.to(self.device)

        self.fit()

    def prepare_features(self):
        """Calculate feature tensor and no-edge adjacency indicator.

        Feature tensor is computed by powering the normalized adjacency matrix,
        where the first dimension is the powering number.

        """
        adj_mat = nx.adjacency_matrix(self.g, dtype=np.float32).toarray()

        inv_sqrt_degs = 1 / np.sqrt(adj_mat.sum(0, keepdims=True))
        norm_adj_mat = adj_mat * inv_sqrt_degs * inv_sqrt_degs.T

        out = [norm_adj_mat]
        if self.window_size > 1:
            for _ in trange(
                self.window_size - 1,
                desc="Adjacency matrix powers",
                disable=not self.verbose,
            ):
                out.append(out[-1] @ norm_adj_mat)
        out = np.array(out, dtype=np.float32)

        self.register_buffer("target_tensor", torch.from_numpy(out))
        self.shapes = list(self.target_tensor.shape)

        self.register_buffer("adj_opposite", torch.from_numpy(adj_mat == 0))

    def define_weights(self):
        half_dim = int(self.dim / 2)
        self.left = nn.Parameter(torch.Tensor(self.shapes[1], half_dim))
        self.right = nn.Parameter(torch.Tensor(half_dim, self.shapes[1]))
        self.attn_weights = nn.Parameter(torch.Tensor(self.shapes[0], 1, 1))

    def initialize_weights(self):
        nn.init.uniform_(self.left, -0.01, 0.01)
        nn.init.uniform_(self.right, -0.01, 0.01)
        nn.init.uniform_(self.attn_weights, -0.01, 0.01)

    def forward(self):
        target_tensor = self.target_tensor
        adj_opposite = self.adj_opposite

        attn = self.attn_weights.softmax(0)
        target_mat = (target_tensor * attn).sum(0)

        pred = torch.mm(self.left, self.right).sigmoid().clamp(EPS, 1 - EPS)
        pos_loss = -(target_mat * torch.log(pred))
        neg_loss = -(adj_opposite * torch.log(1 - pred))
        nlgl = (self.walk_length * target_mat.shape[0] * pos_loss + neg_loss).mean()

        attn_reg = self.beta * self.attn_weights.norm(2).pow(2)
        emb_reg = self.gamma * (self.left.abs().mean() + self.right.abs().mean())

        loss = nlgl + attn_reg + emb_reg

        return loss

    def fit(self):
        self.train()
        logger.info("Start training AttentionWalk embeddings.")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        pbar = trange(self.epochs, desc="Loss", disable=not self.verbose)
        for _ in pbar:
            loss = self()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4f}")

    @torch.no_grad()
    def get_emb(self, as_numpy: bool = True) -> Union[torch.Tensor, np.ndarray]:
        emb = torch.cat((self.left, self.right.T), dim=1).cpu()
        return emb.numpy() if as_numpy else emb
