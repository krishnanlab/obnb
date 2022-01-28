from .base import BaseNdexData


class BioPlex(BaseNdexData):
    """The BioPlex3-shared Protein Protein Interaction network."""

    cx_uuid = "f7a218c0-2376-11ea-bb65-0ac135e8bacf"

    def __init__(self, root: str):
        """Initialize the BioPlex network data."""
        super().__init__(
            root,
            weighted=False,
            directed=False,
            interaction_types=["interacts-with"],
            node_id_prefix="ncbigene",
            node_id_entry="r",
        )
