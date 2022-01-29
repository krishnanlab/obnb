from .base import BaseNdexData


class HumanNet(BaseNdexData):
    """The HumanNet-FN functional interaction network."""

    cx_uuid = "0d05756c-3553-11e9-9f06-0ac135e8bacf"

    def __init__(self, root: str):
        """Initialize the HumanNet network data."""
        super().__init__(
            root,
            weighted=True,
            directed=False,
            interaction_types=["associates-with"],
            node_id_prefix="ncbigene",
            default_edge_weight=0.0,
            edge_weight_attr_name="LLS",
            reduction=None,
        )
