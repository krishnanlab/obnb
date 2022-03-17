import os
import os.path as osp

from .. import logger
from ..typing import List
from ..util.path import cleandir


class BaseData:
    """BaseData object.

    This is an abstract (mixin) class for constructing data objects. The main
    methods are _download and _process, which are wrappers that download the
    raw files and process the files into the final processed file if they are
    not yet available. Otherwise, directly load the previously processed file.

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
        self.load_processed_data()

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
            logger.info(f"Start downloading {self.classname}...")
            self.download()

    def process(self):
        """Process raw files."""
        raise NotImplementedError

    def _process(self):
        """Check to see if processed file exist and process if not."""
        os.makedirs(self.processed_dir, exist_ok=True)
        if self.redownload or self.reprocess or not self.process_completed():
            logger.info(f"Start processing {self.classname}...")
            self.process()
