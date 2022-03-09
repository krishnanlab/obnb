from .base import BaseNdexData


class HIPPIE(BaseNdexData):
    """The HIPPIE Human scored Protein Protein Interaction network.

    Note: the inferred PPI directionality is disregarded, i.e. the resulting
        network is undirected.

    """

    cx_uuid = "89dd3925-3718-11e9-9f06-0ac135e8bacf"

    def __init__(self, root: str, **kwargs):
        """Initialize the HIPPIE network data."""
        super().__init__(
            root,
            weighted=True,
            directed=False,
            interaction_types=["interacts-with"],
            node_id_prefix="ncbigene",
            default_edge_weight=0.0,
            edge_weight_attr_name="Confidence Value",
            reduction="max",
            **kwargs,
        )
