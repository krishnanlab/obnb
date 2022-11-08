from nleval.data.network.base import BaseNDExData
from nleval.typing import Converter


class BioGRID(BaseNDExData):
    """The BioGRID Protein Protein Interaction network."""

    cx_uuid = "ca656884-a17d-11ec-b3be-0ac135e8bacf"

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
        """Initialize the BioGRID network data."""
        super().__init__(
            root,
            weighted=weighted,
            directed=directed,
            largest_comp=largest_comp,
            gene_id_converter=gene_id_converter,
            cx_kwargs={
                "interaction_types": ["interacts-with"],
                "node_id_prefix": "ncbigene",
                "node_id_entry": "r",
            },
            **kwargs,
        )
