from typing import get_args

import numpy as np
from pecanpy import pecanpy

from obnb.feature import FeatureVec
from obnb.graph import DenseGraph, SparseGraph
from obnb.alltypes import Literal, Optional, Union

PECANPY_MODES = Literal[
    "PreComp",
    "SparseOTF",
    "DenseOTF",
    "PreCompFirstOrder",
    "FirstOrderUnweighted",
]


def pecanpy_embed(
    g: Union[DenseGraph, SparseGraph],
    *,
    mode: PECANPY_MODES = "SparseOTF",
    p: float = 1,
    q: float = 1,
    extend: bool = False,
    gamma: float = 0,
    dim: int = 128,
    num_walks: int = 10,
    walk_length: int = 80,
    window_size: int = 10,
    epochs: int = 1,
    workers: int = 1,
    verbose: bool = False,
    random_state: Optional[int] = None,
    as_array: bool = False,
) -> Union[FeatureVec, np.ndarray]:
    """Generate node2vec(+) embeddings using PecanPy.

    Args:
        g: Input graph object.
        mode: Different implementations of th PecanPy that are more optimized
            for different scenarios (memory-/runtime-wise). See
            https://github.com/krishnanlab/PecanPy for more info.
        p: The return paarmeter for the biased random walk.
        q: The in-out parameter for the biased random walk.
        extend: Use [node2vec+](https://doi.org/10.1093/bioinformatics/btad047)
            if set to True.
        gamma: Thresholding parameter for node2vec+.
        dim: Embedding dimension.
        num_walks: Number of walks to generate starting from each node.
        walk_length: Length of the ranom walks.
        window_size: Window size for the skip-gram to train.
        epochs: Skip-gram training epochs.
        workers: Number of workers for the skip-gram training.
        verbose: Whether to report progress.
        random_state: Random state to control the random walks. Note that this
            does not guarantee reproducibility due to the randomness of
            gensim's skip-gram.
        as_array: If set to True, then return the embeddings as a 2-d numpy
            array (node x dim). Otherwise, return as a :class:`FeatureVec`
            object.

    """
    # Obtain dense adjacency matrix
    if isinstance(g, SparseGraph):
        adj = g.to_dense_graph().mat
    elif isinstance(g, DenseGraph):
        adj = g.mat
    else:
        raise TypeError(
            f"Input graph must be type DenseGraph or SparseGraph, got {type(g)}: {g!r}",
        )

    # Initialize PecanPy graph object
    if not isinstance(mode, str):
        raise TypeError(f"mode must be string type, got {type(mode)}: {mode!r}")
    elif mode not in (opts := get_args(PECANPY_MODES)):
        raise ValueError(f"Unknown mode {mode!r}, available options are: {opts}")
    pecanpy_mode = getattr(pecanpy, mode)
    pcg = pecanpy_mode.from_mat(
        adj,
        g.node_ids,
        p=p,
        q=q,
        extend=extend,
        gamma=gamma,
        workers=workers,
        random_state=random_state,
        verbose=verbose,
    )

    # Generate embeddings
    emd = pcg.embed(
        dim=dim,
        num_walks=num_walks,
        walk_length=walk_length,
        window_size=window_size,
        epochs=epochs,
        verbose=verbose,
    )

    return emd if as_array else FeatureVec.from_mat(emd, list(g.node_ids))
