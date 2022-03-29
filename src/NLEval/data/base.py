import logging
import os
import os.path as osp

from ..typing import Any
from ..typing import List
from ..typing import LogLevel
from ..typing import Optional
from ..util.logger import get_logger
from ..util.logger import log_file_context
from ..util.path import cleandir
from ..util.path import hash_to_hexdigest


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
        log_level: LogLevel = "INFO",
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
        self.log_level = log_level

        # Redownload > reprocess
        reprocess = reprocess or redownload

        self._setup_process_logger()
        log_path = osp.join(self.info_dir, "run.log")
        with log_file_context(self.plogger, log_path):
            self._download(redownload)
            self._process(reprocess)

        self.load_processed_data()

    def _setup_process_logger(self):
        """Set up process logger and file handler for data processing steps."""
        os.makedirs(self.info_dir, exist_ok=True)
        self.plogger = get_logger(
            None,
            base_logger="NLEval_precise",
            log_level=self.log_level,
        )

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
    def info_dir(self) -> str:
        """Return info file directory."""
        return cleandir(osp.join(self.root, self.classname, "info"))

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

    def _download(self, redownload: bool):
        """Check to see if files downloaded first before downloading."""
        os.makedirs(self.raw_dir, exist_ok=True)
        if redownload or not self.download_completed():
            self.plogger.info(f"Start downloading {self.classname}...")
            self.download()

    def process(self):
        """Process raw files."""
        raise NotImplementedError

    def _process(self, reprocess: bool):
        """Check to see if processed file exist and process if not."""
        os.makedirs(self.processed_dir, exist_ok=True)
        if reprocess or not self.process_completed():
            self.plogger.info(f"Start processing {self.classname}...")
            self.process()
