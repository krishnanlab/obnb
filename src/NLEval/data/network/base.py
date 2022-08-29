import ndex2

from NLEval.data.base import BaseData
from NLEval.graph import SparseGraph
from NLEval.typing import Any, Dict, List, Mapping, Optional, Union
from NLEval.util.converter import MyGeneInfoConverter


class BaseNdexData(BaseData, SparseGraph):
    """The BaseNdexData object for retrieving networks from NDEX.

    www.ndexbio.org

    """

    CONFIG_KEYS: List[str] = BaseData.CONFIG_KEYS + [
        "uuid",
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
        node_id_converter: Optional[Union[Mapping[str, str], str]] = "HumanEntrez",
        cx_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize the BaseNdexData object.

        Args:
            root (str): The root directory of the data.
            weighted (bool): Whether the network is weighted or not.
            undirected (bool): Whether the network is undirected or not.
            largest_comp (bool): If set to True, then only take the largest
                connected component of the graph.
            node_id_converter (Union[Mapping[str, str], str], optional): A
                mapping object that maps a given node ID to a new node ID of
                interest. Or the name of a predefined MygeneInfoConverter
                object as a string.
            cx_kwargs: Keyword arguments used for reading the cx file.

        """
        self.largest_comp = largest_comp
        self.node_id_converter = node_id_converter  # type: ignore
        self.cx_kwargs: Dict[str, Any] = cx_kwargs or {}
        super().__init__(
            root,
            weighted=weighted,
            directed=directed,
            **kwargs,
        )

    @property
    def raw_files(self) -> List[str]:
        return ["data.cx"]

    @property
    def processed_files(self) -> List[str]:
        return ["data.npz"]

    @property
    def node_id_converter(self) -> Mapping[str, str]:
        if self._node_id_converter is None:
            return {}
        elif isinstance(self._node_id_converter, str):
            return MyGeneInfoConverter.construct(
                self._node_id_converter,
                root=self.root,
                log_level=self.log_level,
            )
        else:
            return self._node_id_converter

    @node_id_converter.setter
    def node_id_converter(self, node_id_converter):
        self._node_id_converter = node_id_converter

    def download(self):
        """Download data from NDEX via ndex2 client."""
        self.plogger.info(f"Retrieve NDEx network with uuid: {self.cx_uuid}")
        client = ndex2.client.Ndex2()
        client_resp = client.get_network_as_cx_stream(self.cx_uuid)
        with open(self.raw_file_path(0), "wb") as f:
            f.write(client_resp.content)

    def process(self):
        """Process data and save for later useage."""
        self.plogger.info(f"Process raw file {self.raw_file_path(0)}")
        cx_graph = SparseGraph(
            weighted=self.weighted,
            directed=self.directed,
            logger=self.plogger,
        )
        cx_graph.read_cx_stream_file(
            self.raw_file_path(0),
            node_id_converter=self.node_id_converter,
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
