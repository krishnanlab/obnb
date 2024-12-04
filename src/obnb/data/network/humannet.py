from obnb.data.network.base import BaseURLSparseGraphData
from obnb.alltypes import Converter, List
from obnb.util.registers import overload_class


class HumanNet(BaseURLSparseGraphData):
    """The HumanNetv3 gene interaction networks.

    https://staging2.inetbio.org/humannetv3/

    The HumanNetv3 gene interaction networks are constructed using various
    types gene association evidences:

    Integrated networks:

        - ``FN``: Functional gene interaction network (``CX`` + ``DB`` + ``DP``
          + ``GI`` + ``GN`` + ``PG`` + ``PI``).
        - ``XC``: Full gene interaction network extended by co-citation (``FN``
          + ``CC``).

    Individual networks:

        - ``CC``: Gene association inferred from co-citation.
        - ``CX``: Gene association inferred from co-expression.
        - ``DB``: Pathway database.
        - ``DP``: Protein domain profile associations.
        - ``GI``: Genetic interactions.
        - ``GN``: Gene neighborhood.
        - ``PG``: Phylogenetic profile associations.
        - ``PI``: Protein-protein interactions.

    """

    CONFIG_KEYS: List[str] = BaseURLSparseGraphData.CONFIG_KEYS + ["channel"]
    base_url: str = "https://staging2.inetbio.org/humannetv3/networks"
    download_zip_type: str = "none"
    individual_channels: List[str] = [
        "CC",  # co-citation
        "CX",  # co-expression
        "DB",  # pathway databases
        "DP",  # protein domain profile associations
        "GI",  # genetic interactions
        "GN",  # gene neighborhood
        "PG",  # phylogenetic profile associations
        "PI",  # protein-protein interactions
    ]
    integrated_channels: List[str] = [
        "FN",  # functional gene interaction network
        "XC",  # full gene interaction network extended by co-citation
    ]

    def __init__(
        self,
        root: str,
        *,
        channel: str = "XC",
        weighted: bool = True,
        directed: bool = False,
        largest_comp: bool = True,
        gene_id_converter: Converter = "HumanEntrez",
        **kwargs,
    ):
        """Initialize the HumanNet network data."""
        # Set up url based on the specified channel
        if not isinstance(channel, str):
            raise TypeError(f"channel must be str, got {type(channel)}: {channel!r}")
        elif channel in self.individual_channels:
            name = f"HS-{channel}"
        elif channel in self.integrated_channels:
            name = f"HumanNet-{channel}"
        else:
            all_channels = self.individual_channels + self.integrated_channels
            raise ValueError(
                f"Invalid channel: {channel!r}. Available options are: {all_channels}",
            )
        self.url = f"{self.base_url}/{name}.tsv"
        self.channel = channel

        super().__init__(
            root,
            weighted=weighted,
            directed=directed,
            largest_comp=largest_comp,
            gene_id_converter=gene_id_converter,
            **kwargs,
        )

    @property
    def raw_files(self) -> List[str]:
        return [f"data_{self.channel.lower()}.txt"]

    @property
    def processed_files(self) -> List[str]:
        return [f"data_{self.channel.lower()}.npz"]


HumanNet_CC = overload_class(HumanNet, "CC", channel="CC")
HumanNet_FN = overload_class(HumanNet, "FN", channel="FN")
