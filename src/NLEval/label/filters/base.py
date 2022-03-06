from tqdm import tqdm

from ...typing import LogLevel
from ...util.logger import get_logger


class BaseFilter:
    """Base Filter object containing basic filter operations.

    Notes:
        Loop through all instances (IDs) retrieved by `self.get_ids` and decide
        whether or not to apply modification using `self.criterion`, and finally
        apply modification if passes criterion using `mod_fun`.

    Basic components (methods) needed for children filter classes:
        criterion: retrun true if the corresponding value of an instance passes
            the criterion
        get_ids: return list of IDs to scan through
        get_val_getter: return a function that map ID of an instance to some
            corresponding values
        get_mod_fun: return a function that modifies an instance

    All three 'get' methods above take a `LabelsetCollection` object as input

    """

    def __init__(
        self,
        log_level: LogLevel = "WARNING",
        verbose: bool = False,
        **kwargs,
    ):
        """Initialize BaseFilter with logger.

        Args:
            log_level (LogLevel): Level of logging, see more in the Logging
                library documentation.
            verbose (bool): Shortcut for setting log_level to INFO. If the
                specified level is more specific to INFO, then do nothing,
                instead of rolling back to INFO level (default: :obj:`False`).

        """
        self.logger = get_logger(
            self.__class__.__name__,
            log_level=log_level,
            verbose=verbose,
        )

    def __repr__(self):
        """Return name of the filer."""
        return self.__class__.__name__

    def __call__(self, lsc, progress_bar):
        entity_ids = self.get_ids(lsc)
        val_getter = self.get_val_getter(lsc)
        mod_fun = self.get_mod_fun(lsc)

        pbar = tqdm(entity_ids, disable=not progress_bar)
        pbar.set_description(f"{self!r}")
        for entity_id in pbar:
            if self.criterion(val_getter(entity_id)):
                mod_fun(entity_id)
                self.logger.debug(
                    f"Modification ({self.mod_name}) criterion met for "
                    f"{entity_id!r}",
                )

    @property
    def mod_name(self):
        """Name of modification to entity."""
        return "UNKNOWN"
