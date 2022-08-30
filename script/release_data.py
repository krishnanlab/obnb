import os.path as osp
import shutil
from pathlib import Path
from pprint import pformat

import NLEval.data
import NLEval.util.logger
from NLEval.data.base import BaseData

homedir = Path(".").resolve()
datadir = osp.join(homedir, "data_release")
archdir = osp.join(datadir, "archived")

all_data = sorted(NLEval.data.__all__)
# TODO: parse new release version?
new_data_release = "nledata-v0.1.0-dev"

BaseData._new_data_release = new_data_release

logger = NLEval.util.logger.get_logger(None, log_level="INFO")
logger.info(f"{homedir=!r}")
logger.info(
    f"Processing {len(all_data)} data objects for release "
    f"{new_data_release!r}:\n{pformat(all_data)}",
)

# TODO: clean up existing data directory

for name in all_data:
    getattr(NLEval.data, name)(datadir)
    # TODO: validate data and print stats (# ndoes&edges for networks; stats() for lsc)
    shutil.make_archive(osp.join(archdir, name), "zip", datadir, name, logger=logger)

# TODO: archive the cache dir
# TODO: validation summaries -> # of datasets, whih one of them failed/succeeded
# TODO: optionally, upload to zenodo and validate once done (check failed uploads)
