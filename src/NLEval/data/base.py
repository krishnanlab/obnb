import os
import os.path as osp

import ndex2
import requests

from .. import logger
from ..graph import SparseGraph
from ..label import LabelsetCollection
from ..typing import Dict
from ..typing import Optional


class BaseNdexData(SparseGraph):
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
        super().__init__(weighted=weighted, directed=directed)

        self.root = root
        self.redownload = redownload
        self.reprocess = reprocess

        self._download()
        self._process(**kwargs)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def raw_data_path(self) -> str:
        return osp.join(self.raw_dir, "data.cx")

    @property
    def processed_data_path(self) -> str:
        return osp.join(self.processed_dir, "data.npz")

    @property
    def raw_dir(self) -> str:
        return cleandir(osp.join(self.root, self.name, "raw"))

    @property
    def processed_dir(self) -> str:
        return cleandir(osp.join(self.root, self.name, "processed"))

    def download(self):
        """Download data from NDEX via ndex2 client."""
        client = ndex2.client.Ndex2()
        client_resp = client.get_network_as_cx_stream(self.cx_uuid)
        with open(self.raw_data_path, "wb") as f:
            f.write(client_resp.content)

    def _download(self):
        """Check to see if files downloaded first before downloading."""
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        if self.redownload or not osp.isfile(self.raw_data_path):
            logger.info("Downloading...")
            self.download()

    def _process(self, **kwargs):
        """Check to see if processed file exist and process if not."""
        if (
            self.reprocess
            or self.redownload
            or not osp.isfile(self.processed_data_path)
        ):
            logger.info("Processing...")
            self.read_cx_stream_file(self.raw_data_path, **kwargs)
            self.save_npz(self.processed_data_path, self.weighted)
            logger.info("Done!")
        else:
            self.read_npz(self.processed_data_path)


class BaseAnnotatedOntologyData(LabelsetCollection):
    """General object for labelset collection from annotated ontology."""

    ontology_url: Optional[str] = None
    annotation_url: Optional[str] = None

    def __init__(
        self,
        root: str,
        redownload: bool = False,
        reprocess: bool = False,
        **kwargs,
    ):
        """Initialize the BaseAnnotatedOntologyData object."""
        super().__init__()

        self.root = root
        self.redownload = redownload
        self.reprocess = reprocess

        self._download()
        self._process()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def data_name_dict(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    def ontology_data_path(self) -> str:
        return osp.join(self.raw_dir, self.data_name_dict["ontology"])

    @property
    def annotation_data_path(self) -> str:
        return osp.join(self.raw_dir, self.data_name_dict["annotation"])

    @property
    def processed_data_path(self) -> str:
        return osp.join(self.processed_dir, "data.gmt")

    @property
    def raw_dir(self) -> str:
        return cleandir(osp.join(self.root, self.name, "raw"))

    @property
    def processed_dir(self) -> str:
        return cleandir(osp.join(self.root, self.name, "processed"))

    def download_ontology(self):
        """Download ontology from obo foundary."""
        resp = requests.get(self.ontology_url)
        ontology_file_name = self.data_name_dict["ontology"]
        with open(osp.join(self.raw_dir, ontology_file_name), "wb") as f:
            f.write(resp.content)

    def download_annotations(self):
        """Download annotations."""
        raise NotImplementedError

    def download(self):
        """Download the ontology and annotations."""
        self.download_ontology()
        self.download_annotations()

    def process(self):
        """Process raw data and save as gmt for future usage."""
        raise NotImplementedError

    def _download(self):
        """Download files if not all raw files are available."""
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        raw_file_names = list(self.data_name_dict.values())
        raw_files = [osp.join(self.raw_dir, i) for i in raw_file_names]
        if self.redownload or any(
            not osp.isfile(raw_file) for raw_file in raw_files
        ):
            logger.info("Downloading...")
            self.download()

    def _process(self):
        """Check to see if processed file exist and process if not."""
        if (
            self.redownload
            or self.reprocess
            or not osp.isfile(self.processed_data_path)
        ):
            logger.info("Processing...")
            self.process()
            logger.info("Done!")
        else:
            self.read_gmt(self.processed_data_path)


def cleandir(rawdir: str) -> str:
    """Expand user and truncate relative paths."""
    return osp.expanduser(osp.normpath(rawdir))
