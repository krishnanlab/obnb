import os
import random

import numpy as np

from obnb.alltypes import Any, Optional


class DotDict(dict):
    __getattr__ = dict.__getitem__  # type: ignore
    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore


def set_seed(seed: Optional[int] = None):
    os.environ["PYTHONHASHSEED"] = os.environ["OBNB_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def default(val: Any, default_val: Any):
    return default_val if val is None else val


def default_random_state() -> Optional[int]:
    seed = os.environ.get("OBNB_GLOBAL_SEED")
    int_seed = int(seed) if seed is not None else None
    return int_seed


def get_random_state(seed: Optional[int]) -> Optional[int]:
    return default(seed, default_random_state())


def get_num_workers(num_workers: int = 1) -> int:
    if not isinstance(num_workers, int):
        raise TypeError(f"num_workers must be an integer, got {type(num_workers)}")

    elif num_workers == -1:
        try:
            import numba

            num_workers = numba.config.NUMBA_DEFAULT_NUM_THREADS

        except ImportError:
            import multiprocessing

            num_workers = multiprocessing.cpu_count()

    elif num_workers < 1:
        raise ValueError(
            f"num_workers must be positive integer (or exactly -1 for using "
            f"all available threads), got {num_workers}",
        )

    return num_workers
