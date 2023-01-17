import gzip
import os.path as osp
import time
import urllib.parse
from io import BytesIO
from logging import Logger
from pprint import pformat
from zipfile import ZipFile

import requests
from tqdm import tqdm

from nleval.config import (
    DEFAULT_RETRY_DELAY,
    MAX_DOWNLOAD_RETRIES,
    NLEDATA_URL_DICT,
    NLEDATA_URL_DICT_STABLE,
    STREAM_BLOCK_SIZE,
)
from nleval.exception import DataNotFoundError, ExceededMaxNumRetries
from nleval.typing import Literal, LogLevel, Optional, Tuple
from nleval.util.logger import display_pbar, get_logger

native_logger = get_logger(None, log_level="INFO")


def get_data_url(
    version: str,
    name: str,
    *,
    logger: Optional[Logger] = None,
) -> str:
    """Obtain archive data URL.

    The URL is constructed by joining the base archive data URL corresponds
    to the specified version with the data object name, ending with the
    `.zip` extension.

    Args:
        version: Archival version.
        name: Name of the zip file without the `.zip` extension.
        logger: Logger to use. Use default logger if not specified.

    Returns:
        str: URL to download the archive data.

    """
    logger = logger or native_logger

    if (base_url := NLEDATA_URL_DICT.get(version)) is None:
        versions = list(NLEDATA_URL_DICT_STABLE) + ["latest"]
        raise ValueError(
            f"Unrecognized version {version!r}, please choose from the "
            f"following versions:\n{pformat(versions)}",
        )

    data_url = urllib.parse.urljoin(base_url, f"{name}.zip")
    logger.info(f"Download URL: {data_url}")

    return data_url


def get_filename_from_url(url: str) -> str:
    """Extract filename from url."""
    path = urllib.parse.urlparse(url).path
    filename = path.split("/")[-1]
    return filename


def download_unzip(
    url: str,
    root: str,
    *,
    zip_type: Literal["zip", "gzip"] = "zip",
    logger: Optional[Logger] = None,
):
    """Download a zip archive and extract all contents.

    Args:
        url: The url to download the data from.
        root: Directory to put the extracted contents.
        zip_type: Type of zip files to extract, available options are ["zip",
            "gzip"].
        logger: Logger to use. Use default logger if not specified.

    """
    if zip_type not in ["zip", "gzip"]:  # check zip type first before downloading
        raise ValueError(
            f"Unknown zip type {zip_type!r}, available options are [zip|gzip]",
        )

    logger = logger or native_logger
    logger.info(f"Downloading zip archive from {url}")

    _, content = stream_download(url, logger=logger)
    logger.info("Download completed, start unpacking...")

    if zip_type == "zip":
        zf = ZipFile(BytesIO(content))
        zf.extractall(root)
    elif zip_type == "gzip":
        filename = get_filename_from_url(url).replace(".{zip_type}", "")
        with open(osp.join(root, filename), "w") as f:
            f.write(gzip.decompress(content).decode())
    else:
        raise ValueError(f"Fatal error! {zip_type=!r} should have been caught.")

    logger.info("Done extracting")


def stream_download(
    url: str,
    log_level: Optional[LogLevel] = None,
    logger: Optional[Logger] = None,
) -> Tuple[requests.Response, bytes]:
    """Download content from url with option to display progress bar."""
    logger = logger or native_logger
    log_level = log_level or logger.getEffectiveLevel()  # type: ignore

    for _ in range(MAX_DOWNLOAD_RETRIES):
        r, content = _stream_download(url, log_level)  # type: ignore

        if r.ok:
            return r, content
        elif r.status_code in [429, 503]:  # Retry later
            t = r.headers.get("Retry-after", DEFAULT_RETRY_DELAY)
            logger.warning(f"Server temporarily unavailable, waiting for {t} sec")
            time.sleep(int(t))
        elif r.status_code == 404:
            reason = f"{url} is unavailable, try using a more recent data version"
            logger.error(reason)
            raise DataNotFoundError(reason)
        else:
            logger.error(f"Failed to download {url}: {r} {r.reason}")
            raise requests.exceptions.RequestException(r)

    else:  # failed to download within the allowed number of retries
        logger.error(f"Failed to download {url}")
        reason = f"Max number of retries exceeded {MAX_DOWNLOAD_RETRIES=}"
        raise ExceededMaxNumRetries(reason)


def _stream_download(url: str, log_level: LogLevel) -> Tuple[requests.Response, bytes]:
    r = requests.get(url, stream=True)
    tot_bytes = int(r.headers.get("content-length", 0))
    pbar = tqdm(
        total=tot_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        disable=not display_pbar(log_level),
    )

    with BytesIO() as b:
        for data in r.iter_content(STREAM_BLOCK_SIZE):
            pbar.update(len(data))
            b.write(data)
        content = b.getvalue()

    return r, content
