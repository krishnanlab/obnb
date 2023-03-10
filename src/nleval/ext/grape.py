from pprint import pformat

import grape
import numpy as np
from embiggen.utils.abstract_models.abstract_embedding_model import (
    AbstractEmbeddingModel,
)

from nleval.feature import FeatureVec
from nleval.graph.sparse import SparseGraph
from nleval.typing import Type, Union

# Tested methods, see test/ext/test_grape.py
VALIDATED_EMBEDDERS = [
    "FirstOrderLINEEnsmallen",
    "SecondOrderLINEEnsmallen",
    "DeepWalkCBOWEnsmallen",
    "DeepWalkGloVeEnsmallen",
    "DeepWalkSkipGramEnsmallen",
    "HOPEEnsmallen",
    "LaplacianEigenmapsEnsmallen",
    "Node2VecCBOWEnsmallen",
    "Node2VecGloVeEnsmallen",
    "Node2VecSkipGramEnsmallen",
    "SocioDimEnsmallen",
    "UnstructuredEnsmallen",
    "WalkletsCBOWEnsmallen",
    "WalkletsGloVeEnsmallen",
    "WalkletsSkipGramEnsmallen",
    "WeightedSPINE",
    "DegreeSPINE",
    "DegreeWINE",
    "ScoreSPINE",
    "ScoreWINE",
]


# TODO: from dense graph object (edge_gen -> only nonzeros)
def grape_graph_from_nleval_sparse(g: SparseGraph) -> grape.Graph:
    """Convert nleval SparseGraph to a GRAPE graph object."""
    ggb = grape.GraphBuilder()

    # Add nodes
    for node in g.node_ids:
        ggb.add_node(node)

    # Add edges
    for src, dst, weight in g.edge_gen():
        ggb.add_edge(src, dst, weight=weight)

    # Build graph
    g = ggb.build()

    return g


def grape_embed(
    g: Union[SparseGraph, grape.Graph],
    embedding_model: Union[str, Type[AbstractEmbeddingModel]],
    *,
    as_array: bool = False,
    grape_enable: bool = False,
    _test_mode: bool = False,
    **kwargs,
) -> Union[FeatureVec, np.ndarray]:
    """Embed a graph using GRAPE Embiggen supported embedding methods.

    Args:
        g: Input graph object.
        embedding_model: Embedding model to use, see
            https://anacletolab.github.io/grape/grape/embiggen.html for more
            info.
        as_array: If set to True, then return the embeddings as a 2-d numpy
            array (node x dim). Otherwise, return as a :class:`FeatureVec`
            object.
        grape_enable: Enable GRAPE graph extra perks to improve computation
            time with the cost of extra memory usage.
        _test_mode: Bypass the constriaint of only allowing instantiating
            validated GRAPE embedders.
        **kwargs: Other kwaargs for the ``embedding_model``. Only used when the
            ``embedding_model`` is passed as a string.

    """
    # Convert graph to GRAPE grape if necessary
    if isinstance(g, grape.Graph):
        gpg = g
    elif not isinstance(g, SparseGraph):
        raise TypeError(
            "Input graph must be either a GRAPE Graph object or "
            f"a nleval SparseGraph object, got {type(g)} instead.",
        )
    else:
        gpg = grape_graph_from_nleval_sparse(g)

    if grape_enable:
        gpg.enable()

    # Instantiate the embedder if necessary
    if not isinstance(embedding_model, AbstractEmbeddingModel):
        if not isinstance(embedding_model, str):
            raise TypeError(
                "embedding_model must be a string or a GRAPE embedding method "
                f"object, got {type(embedding_model)}: {embedding_model!r}",
            )
        elif (not _test_mode) and (embedding_model not in VALIDATED_EMBEDDERS):
            raise ValueError(
                f"{embedding_model!r} is not validated yet, if you still want "
                "to use this method, pass `_test_mode=True`. Currently "
                f"supported options are:\n{pformat(VALIDATED_EMBEDDERS)}",
            )
        embedder = getattr(grape.embedders, embedding_model)(**kwargs)
    else:
        embedder = embedding_model

    # Generate embeddings
    gpe = grape.embedders.embed_graph(gpg, embedder, return_dataframe=False)
    emd = np.hstack(gpe.get_all_node_embedding())

    return emd if as_array else FeatureVec.from_mat(emd, gpg.get_node_names())
