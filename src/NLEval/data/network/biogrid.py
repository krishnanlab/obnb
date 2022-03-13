from .base import BaseNdexData


class BioGRID(BaseNdexData):
    """The BioGRID Protein Protein Interaction network."""

    cx_uuid = "ca656884-a17d-11ec-b3be-0ac135e8bacf"

    def __init__(self, root: str, **kwargs):
        """Initialize the BioGRID network data."""
        super().__init__(
            root,
            weighted=False,
            directed=False,
            largest_comp=True,
            cx_kwargs={
                "interaction_types": ["interacts-with"],
                "node_id_prefix": "ncbigene",
                "node_id_entry": "r",
            },
            **kwargs,
        )
