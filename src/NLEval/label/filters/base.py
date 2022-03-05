import logging
from typing import Literal

from tqdm import tqdm

LogLevel = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]


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
        logger_name = f"defaultLogger.{self.__class__.__name__}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(getattr(logging, log_level))
        if verbose and self.logger.getEffectiveLevel() > 20:
            self.logger.setLevel(logging.INFO)

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
