import os.path as osp
from pathlib import Path
from pprint import pformat
from shutil import make_archive, rmtree

import nleval
import nleval.data
import nleval.data.annotation
from nleval.config import NLEDATA_URL_DICT
from nleval.data.base import BaseData
from nleval.util.converter import GenePropertyConverter

homedir = Path(".").resolve()
datadir = osp.join(homedir, "data_release")
archdir = osp.join(datadir, "archived")

all_data = sorted(nleval.data.__all__)
annotation_data = sorted(nleval.data.annotation.__all__)
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

# Clean up old data
while osp.isdir(datadir):
    # TODO: make --allow-dirty option
    answer = input(f"Release data dir already exists ({datadir}), remove now? [yes/no]")
    if answer == "yes":
        logger.info(f"Removing old archives in {datadir}")
        rmtree(datadir)
        break
    elif answer == "no":
        exit()
    else:
        logger.error(f"Unknown option {answer!r}, please answer 'yes' or 'no'")

# Download, process, and archive all data
for name in all_data:
    getattr(nleval.data, name)(datadir)
    if name in annotation_data:
        # NOTE: annotation data objects could contain multiple raw files
        # preprared by different annotated ontology objects, so we need to wait
        # until all annotations are prepared before archiving them.
        continue
    # TODO: validate data and print stats (#nodes&#edges for networks; stats() for lsc)
    make_archive(osp.join(archdir, name), "zip", datadir, name, logger=logger)

# Archive annotation data once all raw files are prepared
for name in annotation_data:
    make_archive(osp.join(archdir, name), "zip", datadir, name, logger=logger)

# Download and process gene property data
GenePropertyConverter(datadir, name="PubMedCount")

# Archive cache
make_archive(osp.join(archdir, ".cache"), "zip", datadir, ".cache", logger=logger)

# TODO: validation summaries -> # of datasets, with one of them failed/succeeded
# TODO: optionally, upload to zenodo and validate once done (check failed uploads)
