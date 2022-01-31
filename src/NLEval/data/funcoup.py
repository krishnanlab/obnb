from ..util.converter import MyGeneInfoConverter
from .base import BaseNdexData


class FunCoup(BaseNdexData):
    """The FunCoup funcional asssociation network.

    The edge weights are PFC values, which is a probablistic estimation about
    whether a pair of genes are functionally coupled.

    https://funcoup5.scilifelab.se/help/#Citation

    """

    cx_uuid = "172990f7-102f-11ec-b666-0ac135e8bacf"

    def __init__(self, root: str):
        """Initialize the FunCoup network data."""
        super().__init__(
            root,
            weighted=True,
            directed=False,
            interaction_types=["has functional association with"],
            node_id_prefix="ensembl",
            node_id_converter=MyGeneInfoConverter(),
            default_edge_weight=0.0,
            edge_weight_attr_name="#0:PFC",
            reduction="max",
        )
