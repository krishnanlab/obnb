from obnb.alltypes import List
from obnb.label.filters.base import BaseFilter


class BaseExistenceFilter(BaseFilter):
    """Filter by existence in some given list of targets."""

    def __init__(
        self,
        target_lst: List[str],
        remove_specified: bool = False,
        **kwargs,
    ) -> None:
        """Initialize BaseExistenceFilter object.

        Args:
            target_lst: List of targets of interest to be preserved
            remove_specified: Remove specified tarets if True. Otherwise,
                preserve the specified targets and remove the unspecified ones.

        """
        super().__init__(**kwargs)
        self.target_lst = target_lst
        self.remove_specified = remove_specified

    @property
    def params(self) -> List[str]:
        """Parameter list."""
        return ["remove_specified"]

    @property
    def all_params(self) -> List[str]:
        """All parameter list."""
        return ["target_lst"] + self.params

    def criterion(self, val):
        if self.remove_specified:
            return val in self.target_lst
        else:
            return val not in self.target_lst


class EntityExistenceFilter(BaseExistenceFilter):
    """Filter entities by list of entiteis of interest.

    Example:
        The following example removes any entities in the labelset_collection
        that are not present in the specified entity_id_list.

        >>> existence_filter = EntityExistenceFilter(entity_id_list)
        >>> labelset_collection.apply(existence_filter, inplace=True)

        Alternatively, can preserve (instead of remove) only eneities not
        present in the entity_id_list by setting ``remove_specified=True``.

    """

    def __init__(
        self,
        target_lst: List[str],
        remove_specified: bool = False,
        **kwargs,
    ) -> None:
        """Initialize EntityExistenceFilter object."""
        super().__init__(target_lst, remove_specified, **kwargs)

    @property
    def mod_name(self):
        return "DROP ENTITY"

    @staticmethod
    def get_val_getter(lsc):
        return lambda x: x  # return entity ID itself

    @staticmethod
    def get_ids(lsc):
        return lsc.entity.lst

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.pop_entity


class LabelsetExistenceFilter(BaseExistenceFilter):
    """Filter labelset by list of labelsets of interest.

    Example:
        The following example removes any labelset in the labelset_collection
        that has a label name matching any of the element in label_name_list

        >>> labelset_existence_filter = LabelsetExistenceFilter(label_name_list)
        >>> labelset_collection.apply(labelset_existence_filter, inplace=True)

        Alternatively, can preserve (instead of remove) only labelsets not
        present in the label_name_list by setting ``remove_specified=True``.

    """

    def __init__(
        self,
        target_lst: List[str],
        remove_specified: bool = False,
        **kwargs,
    ):
        """Initialize LabelsetExistenceFilter object."""
        super().__init__(target_lst, remove_specified, **kwargs)

    @property
    def mod_name(self):
        return "DROP LABELSET"

    @staticmethod
    def get_val_getter(lsc):
        return lambda x: x  # return labelset ID itself

    @staticmethod
    def get_ids(lsc):
        return lsc.label_ids

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.pop_labelset
