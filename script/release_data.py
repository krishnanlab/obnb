import os.path as osp
from pathlib import Path
from pprint import pformat
from shutil import make_archive

import nleval
import nleval.data
from nleval.config import NLEDATA_URL_DICT
from nleval.data.base import BaseData

homedir = Path(".").resolve()
datadir = osp.join(homedir, "data_release")
archdir = osp.join(datadir, "archived")

all_data = sorted(nleval.data.__all__)
new_data_release = nleval.__data_version__

if (url := NLEDATA_URL_DICT.get(new_data_release)) is not None:
    raise ValueError(f"Data release version {new_data_release} exists ({url})")

# Set this to enable setting the correct version number instead of 'latest'
BaseData._new_data_release = new_data_release

logger = nleval.logger
logger.info(f"{homedir=!r}")
logger.info(
    f"Processing {len(all_data)} data objects for release "
    f"{new_data_release!r}:\n{pformat(all_data)}",
)

# TODO: clean up existing data directory

for name in all_data:
    getattr(nleval.data, name)(datadir)
    # TODO: validate data and print stats (# ndoes&edges for networks; stats() for lsc)
    make_archive(osp.join(archdir, name), "zip", datadir, name, logger=logger)

make_archive(osp.join(archdir, ".cache"), "zip", datadir, ".cache", logger=logger)

# TODO: validation summaries -> # of datasets, whih one of them failed/succeeded
# TODO: optionally, upload to zenodo and validate once done (check failed uploads)
