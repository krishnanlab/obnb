from abc import ABC, abstractmethod

# from obnb.dataset.base import Dataset
from obnb.typing import LogLevel, Optional
from obnb.util.logger import get_logger
from obnb.util.misc import get_random_state


class BaseTransform(ABC):
    NAME_PREFIX: Optional[str] = None

    def __init__(
        self,
        *,
        log_level: LogLevel = "INFO",
        random_state: Optional[int] = 42,
    ):
        self.log_level = log_level
        self.logger = get_logger(None, log_level=self.log_level)
        self.random_state = get_random_state(random_state)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def fullname(self) -> str:
        name = self.name
        if self.NAME_PREFIX is not None:
            name = "_".join([self.NAME_PREFIX, name])
        return name


class BaseDatasetTransform(BaseTransform, ABC):
    @abstractmethod
    def __call__(self, dataset):
        ...
