import os
import os.path as osp

import ndex2

from ..graph.sparse import SparseGraph


class BaseNdexData(SparseGraph):
    """The BaseNdexData object for retrieving networks from NDEX.

    www.ndexbio.org

    """

    def __init__(self, root, weighted, directed, **kwargs):
        """Initialize the BaseNdexData object.

        Args:
            root (str): The root directory of the data.
            weighted (bool): Whether the network is weighted or not.
            undirected (bool): Whether the network is undirected or not.
            **kwargs: Other keyword arguments used for reading the cx file.

        """
        super().__init__(weighted=weighted, directed=directed)

        self.root = root
        if self._download():
            print("Processing...")
            self.read_cx_stream_file(self.raw_data_path, **kwargs)
            print("Done!")
            self.save_npz(self.processed_data_path, weighted)
        else:
            self.read_npz(self.processed_data_path, weighted, directed)

    @property
    def raw_data_path(self) -> str:
        return osp.join(self.raw_dir, "data.cx")

    @property
    def processed_data_path(self) -> str:
        return osp.join(self.processed_dir, "data.npz")

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.__class__.__name__, "processed")

    def download(self):
        """Download data from NDEX via ndex2 client."""
        client = ndex2.client.Ndex2()
        client_resp = client.get_network_as_cx_stream(self.cx_uuid)
        with open(self.raw_data_path, "wb") as f:
            f.write(client_resp.content)

    def _download(self) -> bool:
        """Check to see if files downloaded first before downloading.

        Returns:
            bool: False if already downloaded, otherwise True.

        """
        if not osp.isdir(self.raw_dir):
            os.makedirs(osp.expanduser(osp.normpath(self.raw_dir)))
            os.makedirs(osp.expanduser(osp.normpath(self.processed_dir)))

        if not osp.isfile(self.raw_data_path):
            print("Downloading...")
            self.download()
            return True

        return False
