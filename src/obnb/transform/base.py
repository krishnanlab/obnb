from abc import ABC, abstractmethod

# from obnb.dataset.base import Dataset
from obnb.typing import LogLevel, Optional
from obnb.util.logger import get_logger
from obnb.util.misc import get_random_state


class BaseTransform(ABC):
    def __init__(
        self,
        *,
        log_level: LogLevel = "INFO",
        random_state: Optional[int] = 42,
    ):
        self.log_level = log_level
        self.logger = get_logger(None, log_level=self.log_level)
        self.random_state = get_random_state(random_state)


class BaseDatasetTransform(BaseTransform, ABC):
    @abstractmethod
    def __call__(self, dataset):
        ...
