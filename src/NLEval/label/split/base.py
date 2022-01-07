from typing import Iterator
from typing import Optional
from typing import Tuple

import numpy as np


class BaseSplit:
    """BaseSplit object for splitting dataset.

    This is an abstract class for objects used for splitting the dataset
    based on either the labels y and / or some properties of each entity,
    passed in as an 1-dimensional array x. This abstract class only defines a
    __repr__ magic method used for printing.

    """

    def __repr__(self) -> str:
        """Representation of the labelset split object."""
        name = self.__class__.__name__
        attrs = [f"{i.lstrip('_')}={j!r}" for i, j in self.__dict__.items()]
        attrstr = ", ".join(attrs)
        return f"{name}({attrstr})"


class BaseSortedSplit(BaseSplit):
    """BaseSortedSplit object for splitting dataset based on sorting."""

    def __init__(self, ascending: bool = True) -> None:
        """Initialize BaseSortedSplit object.

        Args:
            ascending: Sort the entities in the dataset ascendingly based on
                a property, parsed in a x. Consequently, entities with smaller
                valued properties are used for training and etc. If set to
                False, on the other hand, then sort descendingly.

        """
        self.ascending = ascending

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Iterator[Tuple[np.ndarray, ...]]:
        """Split the dataset based on split index.

        First sort the entity based on their 1-dimensional properties (x),
        then find the list of index used to split the dataset based on the
        sorted entities. Finally, yield the splits.

        Note:
            The use of yield instead of return is to make it compatible with
            the sklearn split methods.

        """
        x_sorted_idx, x_sorted_val = self.sort(x)
        idx = self.get_split_idx(x_sorted_val)
        yield self.split_by_idx(idx, x_sorted_idx)

    def sort(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return the sorted index and value of the entity properties.

        Note:
            Return the negative sorted values if ``self.ascending`` is set to
            False, which is effectively the same as sorting them descendingly.

        Args:
            x: properties of the entities as an 1-dimensional array.

        """
        x_val = x if self.ascending else -x
        x_sorted_idx = x_val.argsort()
        x_sorted_val = x_val[x_sorted_idx]
        return x_sorted_idx, x_sorted_val

    def get_split_idx(self, x_sorted_val):
        raise NotImplementedError

    @staticmethod
    def split_by_idx(idx, x_sorted_idx):
        raise NotImplementedError


class BaseRandomSplit(BaseSortedSplit):
    """BaseRandomSpilt object for randomly splitting dataset.

    Randomly generates the node properties, then use specific sorted split
    to split the dataset based on the random node properties.

    """

    def __init__(
        self,
        *args,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize BaseRandomSplit object.

        Args:
            shuffle (bool): Whether or not to shuffle the dataset ordering,
                if not, use the original ordering (default: :obj:`True`)
            random_state (int, optional): The random state used to shuffle
                the data points (default: :obj:`None`)

        """
        super().__init__(*args)
        self.shuffle = shuffle
        self.random_state = random_state

    def __call__(self, x, y):
        """Split dataset based on random node properties."""
        random_x = self.get_random_x(y)
        yield next(super().__call__(random_x, y))

    def get_random_x(self, y: np.ndarray) -> np.ndarray:
        """Generate random node properties."""
        n = y.shape[0]
        if not self.shuffle:
            x = np.arange(n)
        else:
            np.random.seed(self.random_state)
            x = np.random.choice(n, size=n, replace=False)
        return x
