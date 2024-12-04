"""Version utility module for obnb."""
from obnb import config
from obnb.alltypes import List


def parse_data_version(version: str = "current") -> str:
    """Parse data version string."""
    if version == "current":
        from obnb import __data_version__

        version = __data_version__
    elif (version != "latest") and (version not in config.OBNB_DATA_URL_DICT):
        raise ValueError(
            f"Unknown version {version!r}, please choose from "
            f"{get_available_data_versions(stable_only=False)}",
        )
    return version


def get_available_data_versions(stable_only: bool = True) -> List[str]:
    """Get available data versions."""
    if stable_only:
        return list(config.OBNB_DATA_URL_DICT_STABLE)
    else:
        return list(config.OBNB_DATA_URL_DICT)
