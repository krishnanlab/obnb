import ndex2

from ... import logger
from ...graph import SparseGraph
from ...typing import Any
from ...typing import Dict
from ...typing import List
from ...typing import Optional
from ..base import BaseData


class BaseNdexData(BaseData, SparseGraph):
    """The BaseNdexData object for retrieving networks from NDEX.

    www.ndexbio.org

    """

    uuid: Optional[str] = None

    def __init__(
        self,
        root: str,
        weighted: bool,
        directed: bool,
        redownload: bool = False,
        reprocess: bool = False,
        cx_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize the BaseNdexData object.

        Args:
            root (str): The root directory of the data.
            weighted (bool): Whether the network is weighted or not.
            undirected (bool): Whether the network is undirected or not.
            redownload (bool): If set to True, always download the data
                even if the raw data file already exists in the corresponding
                data folder (default: :obj:`False`).
            reprocess (bool): If set to True, always process the data
                even if the processed data file already exists in the
                corresponding data folder (default: obj:`False`).
            cx_kwargs: Keyword arguments used for reading the cx file.

        """
        self.cx_kwargs: Dict[str, Any] = cx_kwargs or {}
        super().__init__(
            root,
            redownload=redownload,
            reprocess=reprocess,
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

    def download(self):
        """Download data from NDEX via ndex2 client."""
        logger.info(f"Retrieve NDEx network with uuid: {self.cx_uuid}")
        client = ndex2.client.Ndex2()
        client_resp = client.get_network_as_cx_stream(self.cx_uuid)
        with open(self.raw_file_path(0), "wb") as f:
            f.write(client_resp.content)

    def process(self):
        """Process data and save for later useage."""
        logger.info(f"Process raw file {self.raw_file_path(0)}")
        cx_graph = SparseGraph.from_cx_stream_file(
            self.raw_file_path(0),
            **self.cx_kwargs,
        )
        cx_graph.save_npz(self.processed_file_path(0), self.weighted)
        logger.info(f"Saved processed file {self.processed_file_path(0)}")

    def load_processed_data(self):
        """Load processed network."""
        logger.info(f"Load processed file {self.processed_file_path(0)}")
        self.read_npz(self.processed_file_path(0))
