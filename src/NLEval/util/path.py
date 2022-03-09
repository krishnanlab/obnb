"""Path utilities."""
import os.path as osp


def cleandir(rawdir: str) -> str:
    """Expand user and truncate relative paths."""
    return osp.expanduser(osp.normpath(rawdir))
