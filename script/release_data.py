import os.path as osp
from datetime import datetime
from pathlib import Path
from pprint import pformat
from shutil import make_archive, rmtree
from types import ModuleType

import click
import numpy as np
import pandas as pd
from jinja2 import Environment
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

REPORT_TEMPLATE = r"""
## Overview

- Release version: {{ version }}
- Release date: {{ time }}
- Number of networks: {{ num_networks }}
- Number of gene set collections (labels): {{ num_labels }}

## Network stats

{{ network_stats_table }}

## Label stats

{{ label_stats_table }}
"""


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


def setup_dir(allow_dirty: bool):
    if allow_dirty:
        logger.warning(
            "Skip cleaning old archives, be sure you know what you are doing!",
        )
        return

    # Clean up old data
    while osp.isdir(DATADIR):
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
        if name in ANNOTATION_DATA:
            # NOTE: annotation data objects could contain multiple raw files
            # prepared by different annotated ontology objects, so we need to
            # wait until all annotations are prepared before archiving them.
            continue

        if isinstance(obj := getattr(nleval.data, name), ModuleType):
            # Skip modules
            continue

        logger.info(f"Start downloading and processing {name!r}")
        obj(DATADIR)
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
        res = func()
        logger.info(f"{f'[DONE] {name}':-^{width}}")
        return res

    return wrapped_func


@wrap_section
def report_network_stats() -> Tuple[int, str]:
    stats_list: List[Tuple[str, str, str]] = []  # (name, num_nodes, num_edges)
    stats_str_list: List[str] = []
    pbar = tqdm(sorted(set(ALL_DATA) & set(NETWORK_DATA)))
    for name in pbar:
        pbar.set_description(f"Loading stats for {name!r}")
        g = getattr(nleval.data, name)(DATADIR, log_level="WARNING")
        stats_list.append((name, f"{g.num_nodes:,}", f"{g.num_edges:,}"))
        stats_str_list.append(f'("{name}", {g.num_nodes:_}, {g.num_edges:_}),')

    stats_df = pd.DataFrame(stats_list, columns=["Network", "# Nodes", "# Edges"])
    md_table_str = stats_df.to_markdown(index=False)
    logger.info(f"Number of networks: {stats_df.shape[0]}")
    logger.info(f"Network stats:\n{md_table_str}")

    paramatrize_str = "\n".join(stats_str_list)
    logger.info(f"Parametrize format:\n{paramatrize_str}")

    return stats_df.shape[0], md_table_str


@wrap_section
def report_label_stats() -> Tuple[int, str]:
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
    md_table_str = stats_df.to_markdown(index=False)
    logger.info(f"Number of gene set collections: {stats_df.shape[0]}")
    logger.info(f"Label stats:\n{md_table_str}")

    return stats_df.shape[0], md_table_str


def dump_report(network_stats: Tuple[int, str], label_stats: Tuple[int, str]):
    env = Environment()
    template = env.from_string(REPORT_TEMPLATE)
    rendered_str = template.render(
        {
            "version": DATA_RELEASE_VERSION,
            "time": datetime.now().strftime("%Y-%m-%d"),
            "num_networks": network_stats[0],
            "network_stats_table": network_stats[1],
            "num_labels": label_stats[0],
            "label_stats_table": label_stats[1],
        },
    )
    logger.info(f"Full report:\n{rendered_str}")

    outpath = ARCHDIR / "README.md"
    with open(outpath, "w") as f:
        f.write(rendered_str)
    logger.info(f"Report saved to {outpath}")


def report_stats():
    network_stats = report_network_stats()
    label_stats = report_label_stats()
    dump_report(network_stats, label_stats)


@click.command()
@click.option("--allow-dirty", is_flag=True, help="Do not clean data_release/")
def main(allow_dirty: bool):
    setup_version()
    setup_dir(allow_dirty)
    download_process()
    report_stats()
    # TODO: validation summaries -> # of datasets, with one of them failed/succeeded
    # TODO: optionally, upload to zenodo and validate once done (check failed uploads)


if __name__ == "__main__":
    main()
