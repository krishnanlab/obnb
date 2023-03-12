import os.path as osp
from pathlib import Path
from pprint import pformat
from shutil import make_archive, rmtree

import numpy as np
import pandas as pd
from tqdm import tqdm

import nleval.data
from nleval import logger
from nleval.config import NLEDATA_URL_DICT
from nleval.data.base import BaseData
from nleval.typing import Dict, List, Tuple
from nleval.util.converter import GenePropertyConverter

HOMEDIR = Path(__file__).resolve().parent
DATADIR = HOMEDIR / "data_release"
ARCHDIR = DATADIR / "archived"

ALL_DATA = sorted(nleval.data.__all__)
ANNOTATION_DATA = sorted(nleval.data.annotation.__all__)
NETWORK_DATA = sorted(nleval.data.network.__all__)
LABEL_DATA = sorted(nleval.data.annotated_ontology.__all__)
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


def wrap_section(func):
    width = 100
    name = func.__name__

    def wrapped_func():
        logger.info(f"{f'[BEGIN] {name}':=^{width}}")
        func()
        logger.info(f"{f'[DONE] {name}':-^{width}}")

    return wrapped_func


@wrap_section
def report_network_stats():
    stats_list: List[Tuple[str, str, str]] = []  # (name, num_nodes, num_edges)
    stats_str_list: List[str] = []
    pbar = tqdm(sorted(set(ALL_DATA) & set(NETWORK_DATA)))
    for name in pbar:
        pbar.set_description(f"Loading stats for {name!r}")
        g = getattr(nleval.data, name)(DATADIR, log_level="WARNING")
        stats_list.append((name, f"{g.num_nodes:,}", f"{g.num_edges:,}"))
        stats_str_list.append(f'("{name}", {g.num_nodes:_}, {g.num_edges:_}),')

    stats_df = pd.DataFrame(stats_list, columns=["Network", "# Nodes", "# Edges"])
    logger.info(f"Number of networks: {stats_df.shape[0]}")
    logger.info(f"Network stats:\n{stats_df.to_markdown(index=False)}")

    paramatrize_str = "\n".join(stats_str_list)
    logger.info(f"Parametrize format:\n{paramatrize_str}")


@wrap_section
def report_annotation_stats():
    stats_dict_list: List[Dict[str, float]] = []
    pbar = tqdm(sorted(set(ALL_DATA) & set(LABEL_DATA)))
    for name in pbar:
        pbar.set_description(f"Loading stats for {name!r}")
        lsc = getattr(nleval.data, name)(DATADIR, log_level="WARNING")
        stats_dict_list.append(
            {
                "Name": name,
                "# Terms": len(lsc.sizes),
                "Num pos avg": np.mean(lsc.sizes),
                "Num pos std": np.std(lsc.sizes),
                "Num pos min": min(lsc.sizes),
                "Num pos max": max(lsc.sizes),
                "Num pos median": np.median(lsc.sizes),
                "Num pos upper quartile": np.quantile(lsc.sizes, 0.75),
                "Num pos lower quartile": np.quantile(lsc.sizes, 0.25),
            },
        )

    stats_df = pd.DataFrame(stats_dict_list)
    logger.info(f"Number of gene set collections: {stats_df.shape[0]}")
    logger.info(f"Label stats:\n{stats_df.to_markdown(index=False)}")


def report_stats():
    report_network_stats()
    report_annotation_stats()


def main():
    setup_version()
    setup_dir()
    download_process()
    report_stats()
    # TODO: validation summaries -> # of datasets, with one of them failed/succeeded
    # TODO: optionally, upload to zenodo and validate once done (check failed uploads)


if __name__ == "__main__":
    main()
