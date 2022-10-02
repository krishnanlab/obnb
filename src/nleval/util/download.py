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

    num_tries = 0
    while num_tries < MAX_DOWNLOAD_RETRIES:
        num_tries += 1
        r = requests.get(url)

        if r.ok:
            logger.info("Download completed, start unpacking...")
            zf = ZipFile(BytesIO(r.content))
            zf.extractall(root)
            logger.info("Done extracting")
            break
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
