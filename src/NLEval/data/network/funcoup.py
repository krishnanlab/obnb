from NLEval.util.converter import MyGeneInfoConverter
from NLEval.data.network.base import BaseNdexData


class FunCoup(BaseNdexData):
    """The FunCoup funcional asssociation network.

    The edge weights are PFC values, which is a probablistic estimation about
    whether a pair of genes are functionally coupled.

    https://funcoup5.scilifelab.se/help/#Citation

    """

    cx_uuid = "e5122c98-a17d-11ec-b3be-0ac135e8bacf"

    def __init__(self, root: str, **kwargs):
        """Initialize the FunCoup network data."""
        super().__init__(
            root,
            weighted=True,
            directed=False,
            largest_comp=True,
            cx_kwargs={
                "interaction_types": ["has functional association with"],
                "node_id_prefix": "ensembl",
                "node_id_converter": MyGeneInfoConverter(root=root),
                "default_edge_weight": 0.0,
                "edge_weight_attr_name": "#0:PFC",
                "reduction": "max",
            },
            **kwargs,
        )
