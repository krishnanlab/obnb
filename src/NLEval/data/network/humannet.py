from ...util.converter import MyGeneInfoConverter
from .base import BaseNdexData


class HumanNet(BaseNdexData):
    """The HumanNet-FN functional interaction network."""

    cx_uuid = "fbc750ac-a17d-11ec-b3be-0ac135e8bacf"

    def __init__(self, root: str, **kwargs):
        """Initialize the HumanNet network data."""
        super().__init__(
            root,
            weighted=True,
            directed=False,
            largest_comp=True,
            cx_kwargs={
                "interaction_types": ["associates-with"],
                "node_id_prefix": "ncbigene",
                "node_id_converter": MyGeneInfoConverter(),
                "default_edge_weight": 0.0,
                "edge_weight_attr_name": "LLS",
                "reduction": None,
            },
            **kwargs,
        )
