from nleval.data.network.base import BaseNDExData
from nleval.typing import Converter


class HumanNet(BaseNDExData):
    """The HumanNet-FN functional interaction network."""

    cx_uuid = "40913318-3a9c-11ed-ac45-0ac135e8bacf"

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
        """Initialize the HumanNet network data."""
        super().__init__(
            root,
            weighted=weighted,
            directed=directed,
            largest_comp=largest_comp,
            gene_id_converter=gene_id_converter,
            cx_kwargs={
                "interaction_types": ["has functional association with"],
                "node_id_prefix": "ncbigene",
                "default_edge_weight": 0.0,
                "edge_weight_attr_name": "LLS",
                "reduction": "max",
            },
            **kwargs,
        )
