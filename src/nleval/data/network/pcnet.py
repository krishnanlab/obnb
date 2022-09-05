from nleval.data.network.base import BaseNdexData


class PCNet(BaseNdexData):
    """The PCNet (v1.3) Parsimonious Composit human gene interaction network."""

    cx_uuid = "7a686aa6-c494-11ec-b397-0ac135e8bacf"

    def __init__(self, root: str, **kwargs):
        """Initialize the PCNet network data."""
        super().__init__(
            root,
            weighted=False,
            directed=False,
            largest_comp=True,
            gene_id_converter="HumanEntrez",
            cx_kwargs={
                "interaction_types": ["neighbor-of"],
                "node_id_entry": "n",
                "node_id_prefix": None,
            },
            **kwargs,
        )
