"""Configurations used by obnb."""
from obnb.typing import Dict

__all__ = [
    "DEFAULT_RETRY_DELAY",
    "MAX_DOWNLOAD_RETRIES",
    "STREAM_BLOCK_SIZE",
    "OBNB_DATA_URL_DICT",
    "OBNB_DATA_URL_DICT_DEV",
    "OBNB_DATA_URL_DICT_STABLE",
]

DEFAULT_RETRY_DELAY = 5
MAX_DOWNLOAD_RETRIES = 10
STREAM_BLOCK_SIZE = 1024

OBNB_DATA_URL_DICT_STABLE: Dict[str, str] = {
    "obnbdata-0.1.0": "https://zenodo.org/record/8045270/files/",
}
OBNB_DATA_URL_DICT_DEV: Dict[str, str] = {
    "obnbdata-0.1.0-dev": "https://sandbox.zenodo.org/record/1212773/files/",
    "nledata-v0.1.0-dev6": "https://sandbox.zenodo.org/record/1172122/files/",
    "nledata-v0.1.0-dev5": "https://sandbox.zenodo.org/record/1164492/files/",
    "nledata-v0.1.0-dev4": "https://sandbox.zenodo.org/record/1163507/files/",
    "nledata-v0.1.0-dev3": "https://sandbox.zenodo.org/record/1127466/files/",
    "nledata-v0.1.0-dev2": "https://sandbox.zenodo.org/record/1103542/files/",
    "nledata-v0.1.0-dev1": "https://sandbox.zenodo.org/record/1099982/files/",
    "nledata-v0.1.0-dev": "https://sandbox.zenodo.org/record/1097545/files/",
    "nledata-v1.0-test": "https://sandbox.zenodo.org/record/1096827/files/",
}
OBNB_DATA_URL_DICT: Dict[str, str] = {
    **OBNB_DATA_URL_DICT_STABLE,
    **OBNB_DATA_URL_DICT_DEV,
}
