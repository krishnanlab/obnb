import ndex2

from ...graph import SparseGraph
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
            **kwargs: Other keyword arguments used for reading the cx file.

        """
        super().__init__(
            root,
            redownload=redownload,
            reprocess=reprocess,
            weighted=weighted,
            directed=directed,
        )

    @property
    def raw_files(self) -> List[str]:
        return ["data.cx"]

    @property
    def processed_files(self) -> List[str]:
        return ["data.npz"]

    def download(self):
        """Download data from NDEX via ndex2 client."""
        client = ndex2.client.Ndex2()
        client_resp = client.get_network_as_cx_stream(self.cx_uuid)
        with open(self.raw_file_path(0), "wb") as f:
            f.write(client_resp.content)

    def process(self, **kwargs):
        """Process data and save for later useage."""
        self.read_cx_stream_file(self.raw_file_path(0), **kwargs)
        self.save_npz(self.processed_file_path(0), self.weighted)

    def load_processed_data(self):
        raise NotImplementedError
