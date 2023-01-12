from nleval.data.network.base import BaseNDExData
from nleval.typing import Converter


class HIPPIE(BaseNDExData):
    """The HIPPIE Human scored Protein Protein Interaction network.

    Note: the inferred PPI directionality is disregarded, i.e. the resulting
        network is undirected.

    """

    cx_uuid = "89dd3925-3718-11e9-9f06-0ac135e8bacf"

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
        """Initialize the HIPPIE network data."""
        super().__init__(
            root,
            weighted=weighted,
            directed=directed,
            largest_comp=largest_comp,
            gene_id_converter=gene_id_converter,
            cx_kwargs={
                "interaction_types": ["interacts-with"],
                "node_id_prefix": "entrez gene",
                "default_edge_weight": 0.0,
                "edge_weight_attr_name": "Confidence Value",
                "reduction": "max",
            },
            **kwargs,
        )
