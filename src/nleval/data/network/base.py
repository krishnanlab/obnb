import ndex2

from nleval.data.base import BaseData
from nleval.graph import SparseGraph
from nleval.typing import Any, Dict, List, Mapping, Optional, Union
from nleval.util.download import download_unzip
from nleval.util.logger import display_pbar


class BaseNDExData(BaseData, SparseGraph):
    """The BaseNdexData object for retrieving networks from NDEX.

    www.ndexbio.org

    """

    CONFIG_KEYS: List[str] = BaseData.CONFIG_KEYS + [
        "cx_uuid",
        "weighted",
        "directed",
        "largest_comp",
        "cx_kwargs",
    ]
    uuid: Optional[str] = None

    def __init__(
        self,
        root: str,
        weighted: bool,
        directed: bool,
        largest_comp: bool = False,
        gene_id_converter: Optional[Union[Mapping[str, str], str]] = "HumanEntrez",
        cx_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize the BaseNdexData object.

        Args:
            root (str): The root directory of the data.
            weighted (bool): Whether the network is weighted or not.
            directed (bool): Whether the network is directed or not.
            largest_comp (bool): If set to True, then only take the largest
                connected component of the graph.
            cx_kwargs: Keyword arguments used for reading the cx file.

        """
        self.largest_comp = largest_comp
        self.cx_kwargs: Dict[str, Any] = cx_kwargs or {}
        super().__init__(
            root,
            weighted=weighted,
            directed=directed,
            gene_id_converter=gene_id_converter,
            **kwargs,
        )

    @property
    def raw_files(self) -> List[str]:
        return ["data.cx"]

    @property
    def processed_files(self) -> List[str]:
        return ["data.npz"]

    def download(self):
        """Download data from NDEX via ndex2 client."""
        self.plogger.info(f"Retrieve NDEx network with uuid: {self.cx_uuid}")
        client = ndex2.client.Ndex2()
        client_resp = client.get_network_as_cx_stream(self.cx_uuid)
        with open(self.raw_file_path(0), "wb") as f:
            f.write(client_resp.content)

    def process(self):
        """Process data and save for later usage."""
        self.plogger.info(f"Process raw file {self.raw_file_path(0)}")
        cx_graph = SparseGraph(
            weighted=self.weighted,
            directed=self.directed,
            logger=self.plogger,
        )
        cx_graph.read_cx_stream_file(
            self.raw_file_path(0),
            node_id_converter=self.get_gene_id_converter(),
            **self.cx_kwargs,
        )
        if self.largest_comp:
            cx_graph = cx_graph.largest_connected_subgraph()
        cx_graph.save_npz(self.processed_file_path(0), self.weighted)
        self.plogger.info(f"Saved processed file {self.processed_file_path(0)}")

    def load_processed_data(self, path: Optional[str] = None):
        """Load processed network."""
        path = path or self.processed_file_path(0)
        self.plogger.info(f"Load processed file {path}")
        self.read_npz(path)  # FIX: make sure old data purged


class BaseURLSparseGraphData(BaseData, SparseGraph):
    """Base sparse graph object with data downloaded from URL.

    Notes:
        To set up a new instance, specify the following class attributes
        - :attr:`url`: URL from which the data will be downloaed.
        - :attr:`download_zip_type`: type of the zip file downloaded, `zip`
          or `gzip` (default is `gzip`)

    """

    CONFIG_KEYS: List[str] = BaseData.CONFIG_KEYS + [
        "url",
        "download_zip_type",
        "weighted",
        "directed",
        "largest_comp",
    ]
    url: Optional[str] = None
    download_zip_type: str = "gzip"

    def __init__(
        self,
        root: str,
        weighted: bool,
        directed: bool,
        largest_comp: bool = False,
        gene_id_converter: Optional[Union[Mapping[str, str], str]] = "HumanEntrez",
        **kwargs,
    ):
        """Initialize the BaseURLSparseGraphData object.

        Args:
            root: The root directory of the data.
            weighted: Whether the network is weighted or not.
            directed: Whether the network is directed or not.
            largest_comp: If set to True, then only take the largest connected
                component of the graph.

        """
        self.largest_comp = largest_comp
        super().__init__(
            root,
            weighted=weighted,
            directed=directed,
            gene_id_converter=gene_id_converter,
            **kwargs,
        )

    # TODO: add more flexibility to choice of raw_files (parse at init?)
    @property
    def raw_files(self) -> List[str]:
        return ["data.txt"]

    # TODO: add more flexibility to choice of processed_files (parse at init?)
    @property
    def processed_files(self) -> List[str]:
        return ["data.npz"]

    def download(self):
        """Download data from URL."""
        download_unzip(
            self.url,
            self.raw_dir,
            zip_type=self.download_zip_type,
            # TODO: what if multiple files? e.g., split by tissues
            rename=self.raw_files[0],
            logger=self.plogger,
        )

    # TODO: add more flexibility to the types of raw network file to handle
    def process(self):
        """Process data and save for later usage."""
        graph = SparseGraph.from_edglst(
            self.raw_file_path(0),
            weighted=self.weighted,
            directed=self.directed,
            show_pbar=display_pbar(self.log_level),
        )
        if self.largest_comp:
            graph = graph.largest_connected_subgraph()

        out_path = self.processed_file_path(0)
        graph.save_npz(out_path, self.weighted)
        self.plogger.info(f"Saved processed file {out_path}")

    def load_processed_data(self, path: Optional[str] = None):
        """Load processed network."""
        # TODO: what if multiple files? e.g., split by tissues
        path = path or self.processed_file_path(0)
        self.plogger.info(f"Load processed file {path}")
        self.read_npz(path)  # FIX: make sure old data purged
