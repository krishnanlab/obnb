import os.path as osp

import NLEval.data
from NLEval.label.filters import (
    Compose,
    EntityExistenceFilter,
    LabelsetRangeFilterSize,
    NegativeGeneratorHypergeom,
)
from NLEval.typing import LogLevel


def load_data(
    network: str = "BioPlex",
    label: str = "GOBP",
    sparse: bool = False,
    filter_negative: bool = True,
    log_level: LogLevel = "WARNING",
    progress_bar: bool = False,
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
    data_version = "nledata-v0.1.0-dev"
    save_dir = "datasets"
    print(f"{network=}\n{label=}")

    # Load data
    g = getattr(NLEval.data, network)(save_dir, version=data_version)
    if not sparse:
        g = g.to_dense_graph()

    filter_list = [
        EntityExistenceFilter(list(g.node_ids), log_level=log_level),
        LabelsetRangeFilterSize(min_val=50, max_val=200, log_level=log_level),
    ]

    if filter_negative:
        filter_list.append(
            NegativeGeneratorHypergeom(p_thresh=0.05, log_level=log_level),
        )

    lsc = getattr(NLEval.data, label)(
        save_dir,
        version=data_version,
        transform=Compose(*filter_list, log_level=log_level),
    )

    # Load gene properties for study-bias holdout
    # Note: wait after filtering is done to reduce time for filtering
    property_path = osp.join(osp.pardir, "data", "properties", "PubMedCount.txt")
    lsc.load_entity_properties(property_path, "PubMed Count", 0, int)

    return g, lsc


if __name__ == "__main__":
    load_data(log_level="DEBUG")
    load_data(progress_bar=True)
