import networkx as nx

from obnb.graph import SparseGraph


def sparse_graph_to_nx(
    g: SparseGraph,
    weighted: bool = True,
    weight_key: str = "weight",
) -> nx.Graph:
    """Convert an OBNB sparse graph object to a networkx Graph object.

    Args:
        g: A :cls:`obnb.graph.SparseGraph` object.
        weighted: If set to True, then create networkx graph with edge weight,
            otherwise, only use edge precense and ignore edge weights.
        weight_key: Key of the edge weight to use.

    """
    nx_g = nx.Graph()
    for u, v, w in g.edge_gen():
        opt = {} if not weighted else {weight_key: w}
        nx_g.add_edge(u, v, **opt)
    return nx_g
