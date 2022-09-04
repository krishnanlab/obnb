import typing

import numpy as np

from NLEval.typing import Any, Iterator, List, Mapping, Optional, Tuple
from NLEval.util.converter import BaseConverter


class BaseSplit:
    """BaseSplit object for splitting dataset.

    This is an abstract class for objects used for splitting the dataset
    based on either the labels y and / or some properties of each entity,
    passed in as an 1-dimensional array x. This abstract class only defines a
    __repr__ magic method used for printing.

    """

    @typing.no_type_check
    def __repr__(self) -> str:
        """Representation of the labelset split object."""
        attrs = [
            f"{i.lstrip('_')}={j!r}"
            for i, j in self.__dict__.items()
            if i != "property_converter"
        ]

        try:
            if isinstance(self.property_converter, BaseConverter):
                property_converter_repr = repr(self.property_converter)
            else:
                property_converter_repr = "CustomConverter"
            attrs = [f"property_converter={property_converter_repr}"] + attrs
        except AttributeError:
            pass

        return f"{self.__class__.__name__}({', '.join(attrs)})"


class BaseSortedSplit(BaseSplit):
    """BaseSortedSplit object for splitting dataset based on sorting."""

    def __init__(
        self,
        *,
        property_converter: Mapping[str, Any],
        ascending: bool = True,
    ) -> None:
        """Initialize BaseSortedSplit object.

        Args:
            property_converter: Mapping from entity ID to the corresponding
                property, used for sorting the entities. Note that the mapping
                must return values that are sortable, e.g., int, for all
                entities.
            ascending: Sort the entities in the dataset ascendingly based on
                a property, parsed in a x. Consequently, entities with smaller
                valued properties are used for training and etc. If set to
                False, on the other hand, then sort descendingly.

        """
        self.property_converter = property_converter
        self.ascending = ascending

    def __call__(
        self,
        ids: List[str],
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
        x_sorted_idx, x_sorted_val = self.sort(ids)
        idx = self.get_split_idx(x_sorted_val)
        yield self.split_by_idx(idx, x_sorted_idx)

    def sort(self, ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Return the sorted index and value of the entity properties.

        Note:
            Return the negative sorted values if ``self.ascending`` is set to
            False, which is effectively the same as sorting them descendingly.

        Args:
            x: properties of the entities as an 1-dimensional array.

        """
        try:
            x_val = list(map(self.property_converter.__getitem__, ids))  # type: ignore
        except AttributeError:
            x_val = list(map(self.property_converter, ids))  # type: ignore

        x_sorted_idx = sorted(
            range(len(ids)),
            key=x_val.__getitem__,
            reverse=not self.ascending,
        )
        x_sorted_val = list(map(x_val.__getitem__, x_sorted_idx))
        return np.array(x_sorted_idx), np.array(x_sorted_val)

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
        self.shuffle = shuffle
        self.random_state = random_state
        super().__init__(
            *args,
            property_converter=self._get_random_map(shuffle, random_state),
        )

    @staticmethod
    def _get_random_map(shuffle: bool, random_state: Optional[int]):
        if shuffle:
            rng = np.random.default_rng(random_state)
            return lambda id_: rng.random()
        else:
            return lambda id_: 1
