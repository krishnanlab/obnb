from obnb.data.network.base import BaseNDExData
from obnb.typing import Converter


class ProteomeHD(BaseNDExData):
    """The ProteomeHD Protein Protein Interaction network."""

    cx_uuid = "4cb4b0f3-83da-11e9-848d-0ac135e8bacf"

    def __init__(
        self,
        root: str,
        *,
        weighted: bool = False,
        directed: bool = False,
        largest_comp: bool = True,
        gene_id_converter: Converter = "HumanEntrez",
        **kwargs,
    ):
        """Initialize the ProteomeHD network data."""
        super().__init__(
            root,
            weighted=weighted,
            directed=directed,
            largest_comp=largest_comp,
            gene_id_converter=gene_id_converter,
            cx_kwargs={
                "interaction_types": ["correlates-with"],
                "node_id_prefix": "ncbigene",
                "default_edge_weight": 1.0,
                "edge_weight_attr_name": "score",
                "use_node_alias": True,
            },
            **kwargs,
        )
