from .base import BaseNdexData


class BioGRID(BaseNdexData):
    """The BioGRID Protein Protein Interaction network."""

    cx_uuid = "36f7d8fd-23dc-11e8-b939-0ac135e8bacf"

    def __init__(self, root: str, **kwargs):
        """Initialize the BioGRID network data."""
        super().__init__(
            root,
            weighted=False,
            directed=False,
            interaction_types=["interacts-with"],
            node_id_prefix="ncbigene",
            node_id_entry="r",
            **kwargs,
        )
