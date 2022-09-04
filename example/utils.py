import os.path as osp

import NLEval.data
from NLEval.label.filters import (
    Compose,
    EntityExistenceFilter,
    LabelsetRangeFilterSize,
    LabelsetRangeFilterSplit,
    NegativeGeneratorHypergeom,
)
from NLEval.label.split import RatioPartition
from NLEval.typing import LogLevel
from NLEval.util.converter import GenePropertyConverter


def load_data(
    network: str = "BioPlex",
    label: str = "GOBP",
    sparse: bool = False,
    filter_negative: bool = True,
    filter_holdout_split: bool = False,
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

    # Load graph data
    g = getattr(NLEval.data, network)(save_dir, version=data_version)
    if not sparse:
        g = g.to_dense_graph()

    # Construct filters for processing label set collection
    filter_list = [
        EntityExistenceFilter(list(g.node_ids), log_level=log_level),
        LabelsetRangeFilterSize(min_val=50, max_val=200, log_level=log_level),
    ]

    if filter_negative:
        filter_list.append(
            NegativeGeneratorHypergeom(p_thresh=0.05, log_level=log_level),
        )

    if filter_holdout_split:
        pmdcnt_converter = GenePropertyConverter(
            root=save_dir,
            name="PubMedCount",
            log_level=log_level,
        )
        splitter = RatioPartition(
            *(0.6, 0.2, 0.2),
            ascending=False,
            property_converter=pmdcnt_converter,
        )
        filter_list.append(LabelsetRangeFilterSplit(min_val=20, splitter=splitter))

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


def print_expected(*to_print, header: str = "Expected outcome", width: int = 80):
    break_line = "-" * width
    print()
    print("\n".join([header, break_line, *to_print, break_line]))


if __name__ == "__main__":
    load_data(progress_bar=True)
    load_data(log_level="DEBUG")
    load_data(log_level="INFO", filter_negative=False, filter_holdout_split=True)
