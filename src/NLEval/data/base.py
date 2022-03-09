import os
import os.path as osp

import ndex2
import requests

from .. import logger
from ..graph import SparseGraph
from ..label import LabelsetCollection
from ..typing import List
from ..typing import Optional


class BaseData:
    """BaseData object.

    This is an abstract class for constructing data objects. The main methods
    are _download and _process, which are wrappers that download the raw files
    and process the files into the final processed file if they are not yet
    available. Otherwise, directly load the previously processed file.

    """

    def __init__(
        self,
        root: str,
        redownload: bool = False,
        reprocess: bool = False,
        **kwargs,
    ):
        """Initialize BaseData object.

        Args:
            root (str): Root directory of the data files.
            redownload (bool): If set to True, redownload the data even if all
                raw files are downloaded (default: False).
            reprocess (bool): If set to True, reprocess the data even if the
                processed data is available (default: False).

        """
        super().__init__(**kwargs)

        self.root = root
        self.redownload = redownload
        self.reprocess = reprocess

        self._download()
        self._process()

    @property
    def classname(self) -> str:
        """Return data object name."""
        return self.__class__.__name__

    @property
    def raw_dir(self) -> str:
        """Return raw file directory."""
        return cleandir(osp.join(self.root, self.classname, "raw"))

    @property
    def processed_dir(self) -> str:
        """Return raw file directory."""
        return cleandir(osp.join(self.root, self.classname, "processed"))

    @property
    def raw_files(self) -> List[str]:
        """Return a list of raw file names."""
        raise NotImplementedError

    @property
    def processed_files(self) -> List[str]:
        """Return a list of processed file names."""
        raise NotImplementedError

    def raw_file_path(self, idx: int) -> str:
        """Return path to a raw file given its index."""
        return osp.join(self.raw_dir, self.raw_files[idx])

    def processed_file_path(self, idx) -> str:
        """Return path to a processed file given its index."""
        return osp.join(self.processed_dir, self.processed_files[idx])

    def download_completed(self) -> bool:
        """Check if all raw files are downloaded."""
        return all(
            osp.isfile(osp.join(self.raw_dir, raw_file))
            for raw_file in self.raw_files
        )

    def process_completed(self) -> bool:
        """Check if all processed files are available.."""
        return all(
            osp.isfile(osp.join(self.processed_dir, processed_file))
            for processed_file in self.processed_files
        )

    def load_processed_data(self):
        """Load processed data into the data object."""
        raise NotImplementedError

    def download(self):
        """Download raw files."""
        raise NotImplementedError

    def _download(self):
        """Check to see if files downloaded first before downloading."""
        os.makedirs(self.raw_dir, exist_ok=True)
        if self.redownload or not self.download_completed():
            logger.info("Downloading...")
            self.download()

    def process(self):
        """Process raw files."""
        raise NotImplementedError

    def _process(self):
        """Check to see if processed file exist and process if not."""
        os.makedirs(self.processed_dir, exist_ok=True)
        if self.redownload or self.reprocess or not self.process_completed():
            logger.info("Processing...")
            self.process()
            logger.info("Done!")
        else:
            self.load_processed_data()


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


class BaseAnnotatedOntologyData(BaseData, LabelsetCollection):
    """General object for labelset collection from annotated ontology."""

    ontology_url: Optional[str] = None
    annotation_url: Optional[str] = None
    ontology_file_name: Optional[str] = None
    annotation_file_name: Optional[str] = None

    def __init__(
        self,
        root: str,
        redownload: bool = False,
        reprocess: bool = False,
        **kwargs,
    ):
        """Initialize the BaseAnnotatedOntologyData object."""
        super().__init__(root, redownload=redownload, reprocess=reprocess)

    @property
    def raw_files(self) -> List[str]:
        """List of available raw files."""
        files = [self.ontology_file_name, self.annotation_file_name]
        return list(filter(None, files))

    @property
    def processed_files(self) -> List[str]:
        return ["data.gmt"]

    @property
    def ontology_file_path(self) -> str:
        """Path to onlogy file."""
        if self.ontology_file_name is not None:
            return osp.join(self.raw_dir, self.ontology_file_name)
        else:
            raise ValueError(
                f"Ontology file name not available for {self.classname!r}",
            )

    @property
    def annotation_file_path(self) -> str:
        """Path to annotation fil."""
        if self.annotation_file_name is not None:
            return osp.join(self.raw_dir, self.annotation_file_name)
        else:
            raise ValueError(
                f"Annotation file name not available for {self.classname!r}",
            )

    def download_ontology(self):
        """Download ontology from obo foundary."""
        resp = requests.get(self.ontology_url)
        with open(osp.join(self.raw_dir, self.ontology_file_name), "wb") as f:
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

    def load_processed_data(self):
        raise NotImplementedError


def cleandir(rawdir: str) -> str:
    """Expand user and truncate relative paths."""
    return osp.expanduser(osp.normpath(rawdir))
