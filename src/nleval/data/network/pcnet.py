from nleval.data.network.base import BaseNDExData
from nleval.typing import Converter


class PCNet(BaseNDExData):
    """The PCNet (v1.3) Parsimonious Composite human gene interaction network."""

    cx_uuid = "7a686aa6-c494-11ec-b397-0ac135e8bacf"

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
        """Initialize the PCNet network data."""
        super().__init__(
            root,
            weighted=weighted,
            directed=directed,
            largest_comp=largest_comp,
            gene_id_converter=gene_id_converter,
            cx_kwargs={
                "interaction_types": ["neighbor-of"],
                "node_id_entry": "n",
                "node_id_prefix": None,
            },
            **kwargs,
        )
