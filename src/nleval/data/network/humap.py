from nleval.data.network.base import BaseURLSparseGraphData
from nleval.typing import Converter


class HuMAP(BaseURLSparseGraphData):
    """The hu.MAP 2.0 protein interaction network."""

    url = "http://humap2.proteincomplexes.org/static/downloads/humap2/humap2_ppis_geneid_20200821.pairsWprob.gz"

    def __init__(
        self,
        root: str,
        *,
        weighted: bool = True,
        directed: bool = False,
        largest_comp: bool = True,
        gene_id_converter: Converter = None,  # already in Entrez space
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
