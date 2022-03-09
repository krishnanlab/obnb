from .base import BaseNdexData


class STRING(BaseNdexData):
    """The STRING Human Protein Protein Interaction network."""

    cx_uuid = "d14db454-3d18-11e8-a935-0ac135e8bacf"

    def __init__(self, root: str, **kwargs):
        """Initialize the STRING network data."""
        super().__init__(
            root,
            weighted=True,
            directed=False,
            cx_kwargs={
                "interaction_types": ["interacts-with"],
                "node_id_prefix": "ncbigene",
                "default_edge_weight": 0.0,
                "edge_weight_attr_name": "combined_score",
                "reduction": "max",
                "use_node_alias": True,
            },
            **kwargs,
        )
