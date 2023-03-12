import os.path as osp
from pathlib import Path
from pprint import pformat
from shutil import make_archive, rmtree

import nleval
import nleval.data
import nleval.data.annotation
from nleval import logger
from nleval.config import NLEDATA_URL_DICT
from nleval.data.base import BaseData
from nleval.util.converter import GenePropertyConverter

HOMEDIR = Path(__file__).resolve().parent
DATADIR = HOMEDIR / "data_release"
ARCHDIR = DATADIR / "archived"

ALL_DATA = sorted(nleval.data.__all__)
ANNOTATION_DATA = sorted(nleval.data.annotation.__all__)
DATA_RELEASE_VERSION = nleval.__data_version__


def setup_version():
    if (url := NLEDATA_URL_DICT.get(DATA_RELEASE_VERSION)) is not None:
        raise ValueError(f"Data release version {DATA_RELEASE_VERSION} exists ({url})")

    # Set this to enable setting the correct version number instead of 'latest'
    BaseData._new_data_release = DATA_RELEASE_VERSION

    logger.info(f"{HOMEDIR=!r}")
    logger.info(
        f"Processing {len(ALL_DATA)} data objects for release "
        f"{DATA_RELEASE_VERSION!r}:\n{pformat(ALL_DATA)}",
    )


def setup_dir():
    # Clean up old data
    while osp.isdir(DATADIR):
        # TODO: make --allow-dirty option
        answer = input(f"Release data dir exists ({DATADIR}), remove now? [yes/no]")
        if answer == "yes":
            logger.info(f"Removing old archives in {DATADIR}")
            rmtree(DATADIR)
            break
        elif answer == "no":
            exit()
        else:
            logger.error(f"Unknown option {answer!r}, please answer 'yes' or 'no'")


def download_process():
    # Download, process, and archive all data
    for name in ALL_DATA:
        getattr(nleval.data, name)(DATADIR)
        if name in ANNOTATION_DATA:
            # NOTE: annotation data objects could contain multiple raw files
            # prepared by different annotated ontology objects, so we need to
            # wait until all annotations are prepared before archiving them.
            continue
        # TODO: validate data and print stats (#nodes&#edges for nets; stats() for lsc)
        make_archive(osp.join(ARCHDIR, name), "zip", DATADIR, name, logger=logger)

    # Archive annotation data once all raw files are prepared
    for name in ANNOTATION_DATA:
        make_archive(osp.join(ARCHDIR, name), "zip", DATADIR, name, logger=logger)

    # Download and process gene property data
    GenePropertyConverter(DATADIR, name="PubMedCount")

    # Archive cache
    make_archive(osp.join(ARCHDIR, ".cache"), "zip", DATADIR, ".cache", logger=logger)


def main():
    setup_version()
    setup_dir()
    download_process()
    # TODO: validation summaries -> # of datasets, with one of them failed/succeeded
    # TODO: optionally, upload to zenodo and validate once done (check failed uploads)


if __name__ == "__main__":
    main()
