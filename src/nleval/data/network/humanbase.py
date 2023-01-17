from nleval.data.network.base import BaseURLSparseGraphData
from nleval.typing import Converter


class HumanBaseTopGlobal(BaseURLSparseGraphData):
    """The HumanBase-global network (top edges)."""

    url = "https://s3-us-west-2.amazonaws.com/humanbase/networks/global_top.gz"

    def __init__(
        self,
        root: str,
        *,
        weighted: bool = True,
        directed: bool = False,
        largest_comp: bool = True,
        gene_id_converter: Converter = "HumanEntrez",
        **kwargs,
    ):
        """Initialize the HumanBase-global network with top edges."""
        super().__init__(
            root,
            weighted=weighted,
            directed=directed,
            largest_comp=largest_comp,
            gene_id_converter=gene_id_converter,
            **kwargs,
        )
