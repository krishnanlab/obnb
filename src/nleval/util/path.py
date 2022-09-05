"""Path utilities."""
import hashlib
import os.path as osp


def cleandir(rawdir: str) -> str:
    """Expand user and truncate relative paths."""
    return osp.expanduser(osp.normpath(rawdir))


def hexdigest(x: str) -> str:
    """Turn string into fixed width hexcode using md5."""
    return hashlib.md5(x.encode()).hexdigest()
