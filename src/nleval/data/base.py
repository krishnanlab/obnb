import os
import os.path as osp
import shutil
from datetime import datetime
from pprint import pformat

import yaml

import nleval
from nleval.typing import Any, Converter, Dict, List, LogLevel, Mapping, Optional
from nleval.util.checkers import checkConfig
from nleval.util.converter import MyGeneInfoConverter
from nleval.util.download import download_unzip, get_data_url
from nleval.util.logger import get_logger, log_file_context
from nleval.util.path import cleandir, hexdigest


class BaseData:
    """BaseData object.

    This is an abstract (mixin) class for constructing data objects. The main
    methods are _download and _process, which are wrappers that download the raw
    files and process the files into the final processed file if they are not
    yet available. Otherwise, directly load the previously processed file.

    """

    CONFIG_KEYS: List[str] = ["version", "gene_id_converter"]

    # Set to new data release name when preparing data for new release
    _new_data_release: Optional[str] = None

    def __init__(
        self,
        root: str,
        *,
        version: str = "latest",
        redownload: bool = False,
        reprocess: bool = False,
        retransform: bool = False,
        log_level: LogLevel = "INFO",
        pre_transform: Any = "default",
        transform: Optional[Any] = None,
        cache_transform: bool = True,
        download_cache: bool = True,
        gene_id_converter: Converter = "HumanEntrez",
        **kwargs,
    ):
        """Initialize BaseData object.

        Args:
            root: Root directory of the data files.
            version: Name of the version of the data to use, default setting 'latest'
                will download and process the latest data from the source.
            redownload: If set to True, redownload the data even if all raw files are
                downloaded.
            reprocess: If set to True, reprocess the data even if the processed data
                is available.
            retransform: If set to tTrue, retransform the data even if the cached
                transformation is available.
            pre_transform: Optional pre_transformation to be applied to the data object
                before saving as the final processed data object. If set to 'default',
                will use the default pre_transformation.
            transform: Optional transformation to be applied to the data
                object.
            cache_transform: Whether or not to cache the transformed data. The cached
                transformed data will be saved under
                `<data_root_directory>/processed/.cache/`.
            download_cache: If set to True, then check to see if <root>/.cache exists,
                and if not, pull the cache from versioned archive.
            gene_id_converter: A mapping object that maps a given node ID to a new node
                ID of interest. Or the name of a predefined MygeneInfoConverter object
                as a string.

        Note:
            The `pre_transform` option is only valid when `version` is set to
            'latest'.

        """
        super().__init__(**kwargs)

        self.root = root
        self.version = version
        self.log_level = log_level
        self.cache_transform = cache_transform
        self.download_cache = download_cache
        self.gene_id_converter = gene_id_converter

        self.pre_transform = pre_transform
        self._setup_redos(redownload, reprocess, retransform)
        self._setup_process_logger()

        if version == "latest":
            with log_file_context(self.plogger, self.info_log_path):
                self._download()
                self._process()
        else:
            self._download_archive()
            self._process()  # FIX:

        self.load_processed_data()
        self._apply_transform(transform)

    def to_config(self) -> Dict[str, Any]:
        """Generate configuration dictionary from the data object.

        Note:
            If a parameter of the data object is a dictionary, it cannot
            contain value that is another dictionary. The only exception
            currently is `pre_transform`.

        """
        params = {key: getattr(self, key) for key in self.CONFIG_KEYS}
        # Set version to new data release version if it is set
        params["version"] = self._new_data_release or params["version"]
        if self.pre_transform is not None:
            params["pre_transform"] = self.pre_transform.to_config()

        config = {
            "package_version": nleval.__version__,
            "processed_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_module": self.classname,
            "data_module_params": params,
        }
        checkConfig("Data config", config, max_depth=3, white_list=["pre_transform"])
        return config

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
            base_logger="nleval_precise",
            log_level=self.log_level,
        )

    def get_gene_id_converter(self) -> Mapping[str, str]:
        if self.gene_id_converter is None:
            return {}
        elif isinstance(self.gene_id_converter, str):
            return MyGeneInfoConverter.construct(
                self.gene_id_converter,
                root=self.root,
                log_level=self.log_level,
            )
        else:
            return self.gene_id_converter

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
            self._pre_transform = pre_transform
            return  # FIX: allow custom pre_transform for archived version
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

        # Pre-transform data
        if self.pre_transform is not None:
            self.load_processed_data()
            self.plogger.info(f"Applying pre-transformation {self.pre_transform}")
            self.apply_transform(self.pre_transform)

            outpath = self.processed_file_path(0)
            self.save(outpath)
            self.plogger.info(f"Saved pre-transformed file {outpath}")

        # Save data config file
        config_path = osp.join(self.info_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(yaml.dump(self.to_config(), sort_keys=False))
            self.plogger.info(f"Config file saved to {config_path}")

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

        # Set up configs and cache related variables
        config = transform.to_config()
        config_dump = yaml.dump(config)
        hexhash = hexdigest(config_dump)
        cache_dir = osp.join(self.cache_dir, hexhash)
        cache_config_path = osp.join(cache_dir, "config.yaml")
        cache_file_path = osp.join(cache_dir, self.processed_files[0])

        # Check if transformed data cache is available and load directly if so
        if osp.isfile(cache_file_path):
            # TODO: option to furthercheck if info matches (config.yaml)
            with open(cache_config_path) as f:
                force_retransform = False
                if (cache_config := yaml.safe_load(f)) != config:
                    self.plogger.warning(
                        f"Found transformed cache in {cache_dir} but found in "
                        "compatible configs, over writing now. Please report "
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

        # Apply transformation to the data
        # TODO: add option to disable saving option
        # FIX: implement stats for graph/feature data types
        with log_file_context(self.plogger, osp.join(cache_dir, "run.log")):
            self.plogger.info(f"Before transformation:\n{self.stats()}")  # type: ignore
            self.plogger.info(f"Applying transformation:\n{transform}")
            self.apply_transform(transform)
            self.plogger.info(f"After transformation:\n{self.stats()}")  # type: ignore

        # Optionally, save transformed data to cache
        if self.cache_transform:
            self.save(cache_file_path)
            self.plogger.info(f"Saved cache transformation to {cache_file_path}")
        else:
            shutil.rmtree(cache_dir)

    def download_archive(self, version: str):
        """Load data from archived version that ensures reproducibility.

        Note:
            The downloaded data is assumed to be a zip file, which will be
            unzipped and saved to the :attr:`root` directory.

        Args:
            version: Archival version.

        """
        self.plogger.info(f"Loading {self.classname} ({version=})...")
        data_url = get_data_url(version, self.classname, logger=self.plogger)
        download_unzip(data_url, self.root, logger=self.plogger)

        if self.download_cache and not osp.isdir(osp.join(self.root, ".cache")):
            cache_url = get_data_url(version, ".cache", logger=self.plogger)
            download_unzip(cache_url, self.root, logger=self.plogger)

    def _download_archive(self):
        """Check if files data set up and download the archive if not."""
        # TODO: check version in the config file to see if matches
        if (
            self.redownload
            or not self.download_completed()
            or not self.process_completed()
        ):
            self.download_archive(self.version)
