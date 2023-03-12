"""Configurations used by nleval."""
from nleval.typing import Dict

__all__ = [
    "DEFAULT_RETRY_DELAY",
    "MAX_DOWNLOAD_RETRIES",
    "STREAM_BLOCK_SIZE",
    "NLEDATA_URL_DICT",
    "NLEDATA_URL_DICT_DEV",
    "NLEDATA_URL_DICT_STABLE",
]

DEFAULT_RETRY_DELAY = 5
MAX_DOWNLOAD_RETRIES = 10
STREAM_BLOCK_SIZE = 1024

NLEDATA_URL_DICT_STABLE: Dict[str, str] = {}
NLEDATA_URL_DICT_DEV: Dict[str, str] = {
    "nledata-v1.0-test": "https://sandbox.zenodo.org/record/1096827/files/",
    "nledata-v0.1.0-dev": "https://sandbox.zenodo.org/record/1097545/files/",
    "nledata-v0.1.0-dev1": "https://sandbox.zenodo.org/record/1099982/files/",
    "nledata-v0.1.0-dev2": "https://sandbox.zenodo.org/record/1103542/files/",
    "nledata-v0.1.0-dev3": "https://sandbox.zenodo.org/record/1127466/files/",
    "nledata-v0.1.0-dev4": "https://sandbox.zenodo.org/record/1163507/files/",
    "nledata-v0.1.0-dev5": "https://sandbox.zenodo.org/record/1164492/files/",
    "nledata-v0.1.0-dev6": "https://sandbox.zenodo.org/record/1172122/files/",
}
NLEDATA_URL_DICT: Dict[str, str] = {**NLEDATA_URL_DICT_STABLE, **NLEDATA_URL_DICT_DEV}
