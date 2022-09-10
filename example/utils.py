import nleval.data
from nleval.label.filters import (
    Compose,
    EntityExistenceFilter,
    LabelsetRangeFilterSize,
    LabelsetRangeFilterSplit,
    NegativeGeneratorHypergeom,
)
from nleval.label.split import RatioPartition
from nleval.typing import LogLevel
from nleval.util.converter import GenePropertyConverter


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
    data_version = "nledata-v0.1.0-dev2"
    save_dir = "datasets"
    print(f"{network=}\n{label=}")

    # Load graph data
    g = getattr(nleval.data, network)(save_dir, version=data_version)
    if not sparse:
        g = g.to_dense_graph()

    # Construct filters for processing label set collection
    # TODO: EntityExistenceFilter convert to list using property
    filter_list = [
        EntityExistenceFilter(list(g.node_ids), log_level=log_level),
        LabelsetRangeFilterSize(min_val=100, max_val=200, log_level=log_level),
    ]

    if filter_negative:
        filter_list.append(
            NegativeGeneratorHypergeom(p_thresh=0.05, log_level=log_level),
        )

    pmdcnt_converter = GenePropertyConverter(
        root=save_dir,
        name="PubMedCount",
        log_level=log_level,
    )
    if filter_holdout_split:
        splitter = RatioPartition(
            *(0.6, 0.2, 0.2),
            ascending=False,
            property_converter=pmdcnt_converter,
        )
        filter_list.append(LabelsetRangeFilterSplit(min_val=20, splitter=splitter))

    lsc = getattr(nleval.data, label)(
        save_dir,
        version=data_version,
        transform=Compose(*filter_list, log_level=log_level),
    )

    return g, lsc, pmdcnt_converter


def print_expected(
    *to_print,
    header: str = "Expected outcome",
    width: int = 80,
    blank_lines: int = 2,
):
    head_line = f"{header:=^{width}}"
    end_line = "=" * width
    _ = [print() for i in range(blank_lines)]
    print("\n".join([head_line, *to_print, end_line]))


if __name__ == "__main__":
    load_data(progress_bar=True)
    load_data(log_level="DEBUG")
    load_data(log_level="INFO", filter_negative=False, filter_holdout_split=True)
