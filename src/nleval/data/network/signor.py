from nleval.data.network.base import BaseNDExData
from nleval.typing import Converter


class SIGNOR(BaseNDExData):
    """The SIGnaling Network Open Resource human gene interaction network."""

    cx_uuid = "523fff27-afe8-11e9-8bb4-0ac135e8bacf"

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
        """Initialize the SIGNOR network data."""
        super().__init__(
            root,
            weighted=weighted,
            directed=directed,
            largest_comp=largest_comp,
            gene_id_converter=gene_id_converter,
            cx_kwargs={
                "interaction_types": [
                    "down-regulates activity",
                    "down-regulates quantity by destabilization",
                    "down-regulates quantity by repression",
                    "down-regulates quantity",
                    "down-regulates",
                    "form complex",
                    "unknown",
                    "up-regulates activity",
                    "up-regulates quantity by expression",
                    "up-regulates quantity by stabilization",
                    "up-regulates quantity",
                    "up-regulates",
                ],
                "node_id_prefix": "uniprot",
                "node_id_entry": "r",
            },
            **kwargs,
        )
