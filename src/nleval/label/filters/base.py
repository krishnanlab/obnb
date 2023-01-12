from tqdm import tqdm

from nleval.typing import Any, Dict, List, LogLevel
from nleval.util.logger import get_logger


class BaseFilter:
    """Base Filter object containing basic filter operations.

    Loop through all instances (IDs) retrieved by `self.get_ids` and decide
    whether or not to apply modification using `self.criterion`. Finally, apply
    modification if passes criterion using `mod_fun`.

    Basic components (methods) needed for children filter classes:

    - criterion: return true if the corresponding value of an instance passesc
      the criterion.
    - get_ids: return list of IDs to scan through get_val_getter, which returns
      a function that map ID of an instance to some corresponding values.
    - get_mod_fun: return a function that modifies an instance.

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

    @property
    def params(self) -> List[str]:
        """Parameter list."""
        return []

    @property
    def all_params(self) -> List[str]:
        """All parameter list."""
        return self.params

    def __repr__(self):
        """Return name of the filer."""
        name = self.__class__.__name__
        params = ", ".join([f"{i}={getattr(self, i)!r}" for i in self.params])
        return f"{name}({params})"

    def to_config(self) -> Dict[str, Any]:
        """Turn into a config dict."""
        # XXX: has to use repr for now to make splitter display nicely, need to
        # come up with a better solution in the future.
        return {
            self.__class__.__name__: {
                param: repr(getattr(self, param)) for param in self.all_params
            },
        }

    def __call__(self, lsc, progress_bar):
        entity_ids = self.get_ids(lsc)
        val_getter = self.get_val_getter(lsc)
        mod_fun = self.get_mod_fun(lsc)

        pbar = tqdm(entity_ids, desc=f"{self!r}", disable=not progress_bar)
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


class Compose(BaseFilter):
    """Composition of filters."""

    def __init__(self, *filters, log_level: LogLevel = "WARNING"):
        """Initialize composition."""
        super().__init__(log_level=log_level)
        self.filters = filters

    def __repr__(self):
        """Return names of each filter."""
        reprs = "\n".join(f"\t- {filter_!r}" for filter_ in self.filters) or "None"
        return f"Composition of filters:\n{reprs}"

    def to_config(self):
        """Turn into a list of config dict."""
        return [filter_.to_config() for filter_ in self.filters]

    def __call__(self, lsc, progress_bar):
        for filter_ in self.filters:
            filter_.__call__(lsc, progress_bar)
            self.logger.info(lsc.stats())
