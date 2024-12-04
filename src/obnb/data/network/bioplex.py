from obnb.alltypes import Converter
from obnb.data.network.base import BaseNDExData


class BioPlex(BaseNDExData):
    """The BioPlex3-shared Protein Protein Interaction network."""

    cx_uuid = "f7a218c0-2376-11ea-bb65-0ac135e8bacf"

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
        """Initialize the BioPlex network data."""
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
