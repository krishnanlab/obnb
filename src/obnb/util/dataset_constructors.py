"""Helper functions to construct processed datasets."""
import obnb.data
import obnb.label.split
from obnb import Dataset
from obnb.label import filters
from obnb.typing import List, LogLevel, Optional
from obnb.util.converter import GenePropertyConverter


def default_constructor(
    root: str,
    graph_name: str,
    label_name: str,
    *,
    version: str = "current",
    graph_as_feature: bool = False,
    use_dense_graph: bool = False,
    min_size: int = 50,
    min_size_split: int = 5,
    negatives_p_thresh: float = 0.05,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    selected_genes: Optional[List[str]] = None,
    log_level: LogLevel = "INFO",
):
    """Default dataset constructor using study-bias holdout splitting scheme.

    Args:
        root: Directory where the data will be saved.
        version: Archive data version to use. "current" uses the most recent
            processed archive data. "latest" download the latest data from
            source direction and process it from scratch.
        graph_name: Name of the biological network to use.
        label_name: Name of the label sets to use.
        graph_as_feature: If set to True, then set the dataset feature as the
            adjacency matrix of the network.
        use_dense_graph: If set to True, then use dense graph data type for
            the graph in the dataset.
        min_size: Minimum number of positive genes below which the gene set is
            discarded.
        min_size_split: Minimum number of positive genes in any of the
            train/val/test split below which the gene set is discarded.
        negatives_p_thresh: P-value threshold for excluding neutral genes
            determining from negatives.
        selected_genes: List of gene ids to be used in addition to the network
            gene ids for filtering. More specifically, only genes that are
            present in the network and in the provided selected gene list will
            be used. Only use network genes if this is list is not provided.
        log_level: Logging level.

    """
    # Download network data
    graph_cls = getattr(obnb.data, graph_name)
    graph = graph_cls(root, version=version, log_level=log_level)

    # Set up study-bias holdout data splitter
    train_ratio = round(1 - val_ratio - test_ratio, 4)
    if train_ratio < 0:
        raise ValueError("val_ratio and test_ratio must sum below 1")
    elif val_ratio < 0 or test_ratio < 0:
        raise ValueError("val_ratio and test_ratio must be non-negative")
    pubmedcnt_converter = GenePropertyConverter(
        root,
        name="PubMedCount",
        log_level=log_level,
    )
    splitter = obnb.label.split.RatioPartition(
        train_ratio,
        val_ratio,
        test_ratio,
        ascending=False,
        property_converter=pubmedcnt_converter,
    )

    # List of genes of interest
    genes_to_use = list(graph.node_ids)
    if selected_genes is not None:
        orig_num_genes = len(genes_to_use)
        genes_to_use = list(set(genes_to_use) & set(selected_genes))
        new_num_genes = len(genes_to_use)
        obnb.logger.info(
            f"{new_num_genes:,} genes intersecting network genes "
            f"(n={orig_num_genes:,}) and the provided gene list "
            f"(n={len(selected_genes):,})",
        )

    # Download and process the label data
    label = getattr(obnb.data, label_name)(
        root,
        version=version,
        transform=filters.Compose(
            filters.EntityExistenceFilter(genes_to_use),
            filters.LabelsetRangeFilterSize(min_val=min_size),
            filters.LabelsetRangeFilterSplit(min_val=min_size_split, splitter=splitter),
            # XXX: make sure to filter out tasks with insufficient negatives
            filters.NegativeGeneratorHypergeom(p_thresh=negatives_p_thresh),
            log_level=log_level,
        ),
        log_level=log_level,
    )

    # Perform necessary data conversion
    if graph_as_feature or use_dense_graph:
        dense_graph = graph.to_dense_graph()
    graph = dense_graph if use_dense_graph else graph
    feature = dense_graph.to_feature() if graph_as_feature else None

    return Dataset(graph=graph, feature=feature, label=label, splitter=splitter)
