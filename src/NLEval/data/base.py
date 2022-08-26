import os
import os.path as osp
import shutil
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from pprint import pformat
from zipfile import ZipFile

import requests
import yaml

from NLEval._config import config
from NLEval.typing import Any, List, LogLevel, Optional
from NLEval.util.exceptions import DataNotFoundError
from NLEval.util.logger import get_logger, log_file_context
from NLEval.util.path import cleandir, hexdigest


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
        *,
        version: str = "latest",
        redownload: bool = False,
        reprocess: bool = False,
        retransform: bool = False,
        log_level: LogLevel = "INFO",
        transformation: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize BaseData object.

        Args:
            root (str): Root directory of the data files.
            redownload (bool): If set to True, redownload the data even if all
                raw files are downloaded (default: False).
            reprocess (bool): If set to True, reprocess the data even if the
                processed data is available (default: False).
            retransform (bool): If set to tTrue, retransform the data even if
                the cached transformation is available (default: False).
            transformation: Optional transformation to be applied to the data
                obect.

        """
        super().__init__(**kwargs)

        self.root = root
        self.version = version
        self.log_level = log_level
        self._setup_redos(redownload, reprocess, retransform)
        self._setup_process_logger()

        if version == "latest":
            with log_file_context(self.plogger, self.info_log_path):
                self._download()
                self._process()
        else:
            self._download_archive()

        self.load_processed_data()
        self._transform(transformation)

    def _setup_redos(self, redownload: bool, reprocess: bool, retransform: bool):
        # Redownload > reprocess > retransform
        self.redownload = redownload
        self.reprocess = reprocess or self.redownload
        self.retransform = retransform or self.reprocess

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

    @property
    def info_log_path(self) -> str:
        """Return path to the data processing information log file."""
        return osp.join(self.info_dir, "run.log")

    def raw_file_path(self, idx: int) -> str:
        """Return path to a raw file given its index."""
        return osp.join(self.raw_dir, self.raw_files[idx])

    def processed_file_path(self, idx) -> str:
        """Return path to a processed file given its index."""
        return osp.join(self.processed_dir, self.processed_files[idx])

    def download_completed(self) -> bool:
        """Check if all raw files are downloaded."""
        return all(
            osp.isfile(osp.join(self.raw_dir, raw_file)) for raw_file in self.raw_files
        )

    def process_completed(self) -> bool:
        """Check if all processed files are available.."""
        return all(
            osp.isfile(osp.join(self.processed_dir, processed_file))
            for processed_file in self.processed_files
        )

    def load_processed_data(self, path: Optional[str] = None):
        """Load processed data into the data object."""
        raise NotImplementedError

    def download(self):
        """Download raw files."""
        raise NotImplementedError

    def _download(self):
        """Check to see if files downloaded first before downloading."""
        os.makedirs(self.raw_dir, exist_ok=True)
        if self.redownload or not self.download_completed():
            self.plogger.info(f"Start downloading {self.classname}...")
            self.download()

    def process(self):
        """Process raw files."""
        raise NotImplementedError

    def _process(self):
        """Check to see if processed file exist and process if not."""
        os.makedirs(self.processed_dir, exist_ok=True)
        if self.reprocess or not self.process_completed():
            self.plogger.info(f"Start processing {self.classname}...")
            self.process()

    def transform(self, transformation: Any, cache_dir: str):
        """Apply a transformation to the loaded data."""
        raise NotImplementedError

    def _transform(self, transformation: Optional[Any]):
        """Check to see if cached transformed data exist and load if so."""
        # TODO: make this pretransform and add a transform version that do not save?
        if transformation is None:
            return

        config_dump = yaml.dump(transformation.to_config())
        hexhash = hexdigest(config_dump)
        self.plogger.debug(f"{hexhash=}")
        cache_dir = osp.join(self.processed_dir, hexhash)
        if osp.isdir(cache_dir):
            # TODO: option to furthercheck if info matches (config.yaml)
            if self.retransform:
                shutil.rmtree(cache_dir)
            else:
                cache_path = osp.join(cache_dir, "data.gmt")
                self.plogger.info(
                    f"Loading cached transformed data from {cache_path}",
                )
                self.load_processed_data(cache_path)
                return

        os.makedirs(cache_dir)
        with open(osp.join(cache_dir, "config.yaml"), "w") as f:
            f.write(config_dump)
        with log_file_context(self.plogger, osp.join(cache_dir, "run.log")):
            self.transform(transformation, cache_dir)

    def get_data_url(self, version: str) -> str:
        """Obtain archive data URL.

        The URL is constructed by joining the base archive data URL corresponds
        to the specified version with the data object name, ending with the
        '.zip' extension.

        Args:
            version: Archival version.

        Returns:
            str: URL to download the archive data.

        """
        if (base_url := config.NLEDATA_URL_DICT.get(version)) is None:
            versions = list(config.NLEDATA_URL_DICT_STABLE) + ["latest"]
            raise ValueError(
                f"Unrecognized version {version!r}, please choose from the "
                f"following versions:\n{pformat(versions)}",
            )

        data_url = urllib.parse.urljoin(base_url, f"{self.classname}.zip")
        try:
            with urllib.request.urlopen(data_url):
                self.plogger.debug("Connection successul")
        except urllib.error.HTTPError:
            reason = f"{self.classname} is unavailable in version: {version}"
            self.plogger.error(reason)
            raise DataNotFoundError(reason)

        return data_url

    def download_archive(self, version: str):
        """Load data from archived version that ensures reproducibility.

        Note:
            The downloaded data is assumed to be a zip file, which will be
            unzipped and saved to the :attr:`root` directory.

        Args:
            version: Archival verion.

        """
        data_url = self.get_data_url(version)
        self.plogger.info(f"Loading {self.classname} ({version=})...")
        self.plogger.info(f"Download URL: {data_url}")

        # TODO: progress bar
        # WARNING: assumes zip file
        r = requests.get(data_url)
        if not r.ok:
            self.plogger.error(f"Download filed: {r} {r.reason}")
            raise requests.exceptions.RequestException(r)

        self.plogger.info("Download completed, start unpacking...")
        zf = ZipFile(BytesIO(r.content))
        zf.extractall(self.root)

    def _download_archive(self):
        """Check if files data set up and download the archive if not."""
        # TODO: check version in the config file to see if matches
        if not self.download_completed() or not self.process_completed():
            self.download_archive(self.version)
