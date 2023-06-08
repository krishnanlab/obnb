from pprint import pformat

# NOTE: do not import GRAPE directly, which occupies modules like 'utils'
import numpy as np
from embiggen import embedders
from embiggen.utils.abstract_models.abstract_embedding_model import (
    AbstractEmbeddingModel,
)
from ensmallen import Graph, GraphBuilder

from obnb.feature import FeatureVec
from obnb.graph.sparse import SparseGraph
from obnb.typing import Type, Union

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
def grape_graph_from_obnb_sparse(g: SparseGraph) -> Graph:
    """Convert obnb SparseGraph to a GRAPE graph object."""
    ggb = GraphBuilder()

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
    g: Union[SparseGraph, Graph],
    embedding_model: Union[str, Type[AbstractEmbeddingModel]],
    *,
    dim: int = 128,
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
        dim: Embedding dimensions.
        as_array: If set to True, then return the embeddings as a 2-d numpy
            array (node x dim). Otherwise, return as a :class:`FeatureVec`
            object.
        grape_enable: Enable GRAPE graph extra perks to improve computation
            time with the cost of extra memory usage.
        _test_mode: Bypass the constriaint of only allowing instantiating
            validated GRAPE embedders.
        **kwargs: Other kwargs for the ``embedding_model``. Only used when the
            ``embedding_model`` is passed as a string.

    """
    # Convert graph to GRAPE grape if necessary
    if isinstance(g, Graph):
        gpg = g
    elif not isinstance(g, SparseGraph):
        raise TypeError(
            "Input graph must be either a GRAPE Graph object or "
            f"a obnb SparseGraph object, got {type(g)} instead.",
        )
    else:
        gpg = grape_graph_from_obnb_sparse(g)

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
        embedder = getattr(embedders, embedding_model)(embedding_size=dim, **kwargs)
    else:
        embedder = embedding_model

    # Generate embeddings
    gpe = embedders.embed_graph(gpg, embedder, return_dataframe=False)
    emd = np.hstack(gpe.get_all_node_embedding())

    featvec = FeatureVec.from_mat(emd, gpg.get_node_names())
    if featvec.ids != g.node_ids:
        featvec.align_to_ids(list(g.node_ids))

    return featvec.mat if as_array else featvec
