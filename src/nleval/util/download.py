import time
import urllib.parse
from io import BytesIO
from logging import Logger
from pprint import pformat
from zipfile import ZipFile

import requests

from nleval.config import (
    DEFAULT_RETRY_DELAY,
    MAX_DOWNLOAD_RETRIES,
    NLEDATA_URL_DICT,
    NLEDATA_URL_DICT_STABLE,
)
from nleval.exception import DataNotFoundError, ExceededMaxNumRetries
from nleval.typing import Optional
from nleval.util.logger import get_logger

native_logger = get_logger(None, log_level="INFO")


def get_data_url(
    version: str,
    name: str = False,
    *,
    logger: Optional[Logger] = None,
) -> str:
    """Obtain archive data URL.

    The URL is constructed by joining the base archive data URL corresponds
    to the specified version with the data object name, ending with the
    `.zip` extension.

    Args:
        version: Archival version.
        name: Name of the zip file withou the `.zip` extension.
        logger: Logger to use. Use defaut logger if not specified.

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


def download_unzip(url: str, root: str, *, logger: Optional[Logger] = None):
    """Download a zip archive and extract all contents.

    Args:
        url: The url to download the data from.
        root: Directory to put the extracted contents.
        logger: Logger to use. Use defaut logger if not specified.

    """
    logger = logger or native_logger

    logger.info(f"Downloading zip archive from {url}")
    r = requests.get(url)
    if not r.ok:
        logger.error(f"Download filed: {r} {r.reason}")
        raise requests.exceptions.RequestException(r)

    logger.info("Download completed, start unpacking...")
    zf = ZipFile(BytesIO(r.content))
    zf.extractall(root)
    logger.info("Done extracting")
