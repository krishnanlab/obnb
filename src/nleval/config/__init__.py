"""Configurations used by nleval."""
from nleval.typing import Dict

__all__ = [
    "NLEDATA_URL_DICT",
    "NLEDATA_URL_DICT_DEV",
    "NLEDATA_URL_DICT_STABLE",
]

NLEDATA_URL_DICT_STABLE: Dict[str, str] = {}
NLEDATA_URL_DICT_DEV: Dict[str, str] = {
    "nledata-v1.0-test": "https://sandbox.zenodo.org/record/1096827/files/",
    "nledata-v0.1.0-dev": "https://sandbox.zenodo.org/record/1097545/files/",
    "nledata-v0.1.0-dev1": "https://sandbox.zenodo.org/record/1099982/files/",
}
NLEDATA_URL_DICT: Dict[str, str] = {**NLEDATA_URL_DICT_STABLE, **NLEDATA_URL_DICT_DEV}
