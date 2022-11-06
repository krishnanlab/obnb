from nleval.data.network.base import BaseNDExData


class HumanNet(BaseNDExData):
    """The HumanNet-FN functional interaction network."""

    cx_uuid = "fbc750ac-a17d-11ec-b3be-0ac135e8bacf"

    def __init__(self, root: str, **kwargs):
        """Initialize the HumanNet network data."""
        super().__init__(
            root,
            weighted=True,
            directed=False,
            largest_comp=True,
            gene_id_converter="HumanEntrez",
            cx_kwargs={
                "interaction_types": ["associates-with"],
                "node_id_prefix": "ncbigene",
                "default_edge_weight": 0.0,
                "edge_weight_attr_name": "LLS",
                "reduction": None,
            },
            **kwargs,
        )
