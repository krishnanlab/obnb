from typing import get_args

import numpy as np
import sknetwork.embedding

from nleval.feature import FeatureVec
from nleval.graph import DenseGraph, SparseGraph
from nleval.typing import Literal, Union

SKNETWORK_EMBEDDINGS = Literal[
    "Spectral",
    "SVD",
    "GSVD",
    "PCA",
    "RandomProjection",
    "LouvainNE",
]


def sknetwork_embed(
    g: Union[DenseGraph, SparseGraph],
    embedding_model: SKNETWORK_EMBEDDINGS,
    *,
    dim: int = 128,
    as_array: bool = False,
    **kwargs,
) -> Union[FeatureVec, np.ndarray]:
    """Embed a graph using scikit-network embedding methods.

    Args:
        g: Input graph object.
        embedding_model: Name of the scikit-network embedding model to use.
            See https://scikit-network.readthedocs.io/en/latest/reference/embedding.html
            for more info
        dim: Embedding dimensions.
        as_array: If set to True, then return the embeddings as a 2-d numpy
            array (node x dim). Otherwise, return as a :class:`FeatureVec`
            object.
        **kwargs: Other kwargs for the ``embedding_model``.

    """
    # TODO: SparseGraph to scipy csr
    if isinstance(g, SparseGraph):
        adj = g.to_dense_graph().mat
    elif isinstance(g, DenseGraph):
        adj = g.mat
    else:
        raise ValueError(f"Unknown type for input graph {type(g)}")

    if embedding_model not in (opts := get_args(SKNETWORK_EMBEDDINGS)):
        raise ValueError(
            f"Unknown embedding_mode {embedding_model!r}, available "
            f"options are: \n{opts}",
        )

    mdl_cls = getattr(sknetwork.embedding, embedding_model)
    mdl = mdl_cls(n_components=dim, **kwargs)
    emd = mdl.fit_transform(adj)

    return emd if as_array else FeatureVec.from_mat(emd, list(g.node_ids))
