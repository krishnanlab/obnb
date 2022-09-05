from io import BytesIO
from logging import Logger
from zipfile import ZipFile

import requests

from nleval.typing import Optional
from nleval.util.logger import get_logger

native_logger = get_logger(None, log_level="INFO")


def download_unzip(url: str, root: str, *, logger: Optional[Logger] = None):
    """Download a zip archive and extract all contents.

    Args:
        url: The url to download the data from.
        root: Directory to put the extracted contents.
        logger: Logger to use, if not specified, then use the default logger.

    """
    logger = logger or native_logger

    logger.info(f"Donwloading zip archive from {url}")
    r = requests.get(url)
    if not r.ok:
        logger.error(f"Download filed: {r} {r.reason}")
        raise requests.exceptions.RequestException(r)

    logger.info("Download completed, start unpacking...")
    zf = ZipFile(BytesIO(r.content))
    zf.extractall(root)
    logger.info("Done extracting")
