from nleval.data.network.base import BaseNDExData
from nleval.typing import Converter


class FunCoup(BaseNDExData):
    """The FunCoup funcional asssociation network.

    The edge weights are PFC values, which is a probablistic estimation about
    whether a pair of genes are functionally coupled.

    https://funcoup5.scilifelab.se/help/#Citation

    """

    cx_uuid = "e5122c98-a17d-11ec-b3be-0ac135e8bacf"

    def __init__(
        self,
        root: str,
        *,
        weighted: bool = True,
        directed: bool = False,
        largest_comp: bool = True,
        gene_id_converter: Converter = "HumanEntrez",
        **kwargs,
    ):
        """Initialize the FunCoup network data."""
        super().__init__(
            root,
            weighted=True,
            directed=False,
            largest_comp=True,
            gene_id_converter="HumanEntrez",
            cx_kwargs={
                "interaction_types": ["has functional association with"],
                "node_id_prefix": "ensembl",
                "default_edge_weight": 0.0,
                "edge_weight_attr_name": "#0:PFC",
                "reduction": "max",
            },
            **kwargs,
        )
