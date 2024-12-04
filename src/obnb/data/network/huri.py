from obnb.alltypes import Converter
from obnb.data.network.base import BaseNDExData


class HuRI(BaseNDExData):
    """The Human Reference Interactome."""

    cx_uuid = "73bc2c06-5fb2-11e9-9f06-0ac135e8bacf"

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
        """Initialize the HuRI network data."""
        super().__init__(
            root,
            weighted=weighted,
            directed=directed,
            largest_comp=largest_comp,
            gene_id_converter=gene_id_converter,
            cx_kwargs={
                "interaction_types": ["interacts with"],
                "node_id_prefix": "ensembl",
                "node_id_entry": "n",
            },
            **kwargs,
        )
