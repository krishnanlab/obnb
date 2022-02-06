import os.path as osp

import numpy as np
from NLEval.graph import DenseGraph
from NLEval.graph import SparseGraph
from NLEval.label import filters
from NLEval.label import LabelsetCollection


def load_data(
    network: str = "STRING-EXP",
    label: str = "KEGGBP",
    sparse: bool = False,
    filter_negative: bool = True,
):
    """Load graph and node labels.

    Args:
        network (str): Name of the network to use (default: "STRING-EXP").
        label (str): Name of the geneset collection to use (default: "KEGGBP").
        sparse (bool): Whether or not to load the network as sparse graph
            instead of dense graph (default: :obj:`False`).
        filter_negative (bool): Whether or not to filter negative genes based
            on hypergeometric test (default: :obj:`True`).

    """
    data_dir = osp.join(osp.pardir, "data")
    graph_path = osp.join(data_dir, "networks", f"{network}.edg")
    label_path = osp.join(data_dir, "labels", f"{label}.gmt")
    property_path = osp.join(data_dir, "properties", "PubMedCount.txt")

    print(f"{network=}\n{label=}")

    # Load data
    graph_factory = SparseGraph if sparse else DenseGraph
    g = graph_factory.from_edglst(graph_path, weighted=True, directed=False)
    lsc = LabelsetCollection.from_gmt(label_path)

    # Filter labels
    print(f"Number of labelsets before filtering: {len(lsc.label_ids)}")
    lsc.iapply(filters.EntityExistenceFilter(g.idmap.lst))
    lsc.iapply(filters.LabelsetRangeFilterSize(min_val=50))
    if filter_negative:
        lsc.iapply(filters.NegativeGeneratorHypergeom(p_thresh=0.05))
    print(f"Number of labelsets after filtering: {len(lsc.label_ids)}")

    # Load gene properties for study-bias holdout
    # Note: wait after filtering is done to reduce time for filtering
    lsc.load_entity_properties(property_path, "PubMed Count", 0, int)

    return g, lsc
