"""Helper functions to construct processed datasets."""
import nleval.data
import nleval.label.split
from nleval import Dataset
from nleval.label import filters
from nleval.typing import LogLevel
from nleval.util.converter import GenePropertyConverter


def default_constructor(
    root: str,
    version: str,
    graph_name: str,
    label_name: str,
    graph_as_feature: bool = False,
    use_dense_graph: bool = False,
    log_level: LogLevel = "INFO",
):
    """Default dataset constructor using study-bias holdout splitting scheme.

    Args:
        root: Directory where the data will be saved.
        version: Archive data version to use. If set to "latest", then download
            and process the latest data directly from the source.
        graph_name: Name of the biological network to use.
        label_name: Name of the label sets to use.
        graph_as_feature: If set to True, then set the dataset feature as the
            adjacency matrix of the network.
        use_dense_graph: If set to True, then use dense graph data type for
            the graph in the dataset.
        log_level: Logging level.

    """
    # Download network data
    graph = getattr(nleval.data, graph_name)(root, version=version)

    # Set up data splitter
    pubmedcnt_converter = GenePropertyConverter(root, name="PubMedCount")
    splitter = nleval.label.split.RatioPartition(
        *(0.6, 0.2, 0.2),
        ascending=False,
        property_converter=pubmedcnt_converter,
    )

    # Download and process the label data
    label = getattr(nleval.data, label_name)(
        root,
        version=version,
        transform=filters.Compose(
            filters.EntityExistenceFilter(list(graph.node_ids)),
            filters.LabelsetRangeFilterSize(min_val=50),
            filters.LabelsetRangeFilterSplit(min_val=10, splitter=splitter),
            log_level=log_level,
        ),
    )

    # Perform necessary data conversion
    if graph_as_feature or use_dense_graph:
        dense_graph = graph.to_dense_graph()
    graph = dense_graph if use_dense_graph else graph
    feature = dense_graph.to_feature() if graph_as_feature else None

    return Dataset(graph=graph, feature=feature, label=label, splitter=splitter)
