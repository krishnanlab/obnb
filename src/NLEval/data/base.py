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
        transform: Optional[Any] = None,
        pre_transform: Any = "default",
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
            transform: Optional transformation to be applied to the data
                object.
            pre_transform: Optional pre_transformation to be applied to the
                data object before saving as the final processed data object.
                If set to 'default', will use the default pre_transformation.

        Note:
            The `pre_transform` option is only valid when `version` is set to
            'latest'.

        """
        super().__init__(**kwargs)

        self.root = root
        self.version = version
        self.log_level = log_level
        self.pre_transform = pre_transform
        self._setup_redos(redownload, reprocess, retransform)
        self._setup_process_logger()

        if version == "latest":
            with log_file_context(self.plogger, self.info_log_path):
                self._download()
                self._process()
        else:
            self._download_archive()

        self.load_processed_data()
        self._apply_transform(transform)

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
    def _default_pre_transform(self) -> Any:
        return None

    @property
    def pre_transform(self) -> Any:
        return self._pre_transform

    @pre_transform.setter
    def pre_transform(self, pre_transform):
        if pre_transform == "default":
            self._pre_transform = self._default_pre_transform
        elif isinstance(pre_transform, str):
            raise ValueError(f"Unknown pre_transform option {pre_transform}")
        elif self.version != "latest":
            raise ValueError(
                "pre_transform option is only valid when version='latest', "
                f"got {self.version!r} instead",
            )
        else:
            self._pre_transform = pre_transform

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
    def cache_dir(self) -> str:
        """Return transformed data cache directory."""
        return cleandir(osp.join(self.processed_dir, ".cache"))

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
        """Load processed data into the data object.

        Note:
            Any existing data must be purged upon calling this function. That
            is, the data object (self) will contain exactly the data loaded,
            but not not anything else.

        """
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
        """Process raw files and save processed data."""
        raise NotImplementedError

    def _process(self):
        """Check to see if processed file exist and process if not."""
        os.makedirs(self.processed_dir, exist_ok=True)
        if not self.reprocess and self.process_completed():
            return

        # Process data
        self.plogger.info(f"Start processing {self.classname}...")
        self.process()

        if self.pre_transform is None:
            return

        # Pre-transform data
        self.load_processed_data()
        self.plogger.info(f"Applying pre-transformation {self.pre_transform}")
        self.apply_transform(self.pre_transform)

        outpath = self.processed_file_path(0)
        self.save(outpath)
        self.plogger.info(f"Saved pre-transformed file {outpath}")

    def save(self, path):
        """Save the data object to file.

        Args:
            path: Path to the data file to save.

        """
        raise NotImplementedError

    def apply_transform(self, transform: Any):
        """Apply a (pre-)transformation to the loaded data."""
        raise NotImplementedError

    def _apply_transform(self, transform: Optional[Any]):
        """Check to see if cached transformed data exist and load if so."""
        # TODO: make this pretransform and add a transform version that do not save?
        if transform is None:
            return

        # Check if transformed data cache is available and load directly if so
        config = transform.to_config()
        config_dump = yaml.dump(config)
        hexhash = hexdigest(config_dump)
        cache_dir = osp.join(self.cache_dir, hexhash)
        cache_config_path = osp.join(cache_dir, "config.yaml")
        if osp.isdir(cache_dir):
            # TODO: option to furthercheck if info matches (config.yaml)
            with open(cache_config_path, "r") as f:
                force_retransform = False
                if (cache_config := yaml.safe_load(f)) != config:
                    self.plogger.warning(
                        f"Found transformed cache in {cache_dir} but found in "
                        "compatible configs, over writting now. Please report "
                        "to the GitHub issue if you saw this message, along "
                        "with the specific transformation you used.",
                    )
                    force_retransform = True
                self.plogger.debug(f"config:\n{pformat(config)!s}")
                self.plogger.debug(f"cache_config:\n{pformat(cache_config)!s}")

            if not self.retransform and not force_retransform:
                cache_path = osp.join(cache_dir, self.processed_files[0])
                self.plogger.info(f"Loading cached transformed data from {cache_path}")
                self.load_processed_data(cache_path)
                return

            shutil.rmtree(cache_dir)

        os.makedirs(cache_dir)
        with open(cache_config_path, "w") as f:
            f.write(config_dump)

        # Transform and save data transformed data to cache
        # TODO: add option to disable saving option
        # Fix: imlement stats for graph/feature data types
        with log_file_context(self.plogger, osp.join(cache_dir, "run.log")):
            self.plogger.info(f"Before transformation:\n{self.stats()}")  # type: ignore
            self.plogger.info(f"Applying transformation:\n{transform}")
            self.apply_transform(transform)
            self.plogger.info(f"After transformation:\n{self.stats()}")  # type: ignore

            out_path = osp.join(cache_dir, self.processed_files[0])
            self.save(out_path)
            self.plogger.info(f"Saved cache transformation to {out_path}")

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
