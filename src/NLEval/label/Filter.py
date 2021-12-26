from typing import List

import numpy as np
from scipy.stats import hypergeom

__all__ = [
    "EntityRangeFilterNoccur",
    "LabelsetRangeFilterSize",
    "LabelsetRangeFilterTrainTestPos",
    "NegativeFilterHypergeom",
]


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

    def __call__(self, lsc):
        entity_ids = self.get_ids(lsc)
        val_getter = self.get_val_getter(lsc)
        mod_fun = self.get_mod_fun(lsc)

        for entity_id in entity_ids:
            if self.criterion(val_getter(entity_id)):
                mod_fun(entity_id)


class BaseExistanceFilter(BaseFilter):
    """Filter by existance in some given list of targets."""

    def __init__(
        self,
        target_lst: List[str],
        remove_specified: bool = False,
    ) -> None:
        """Initialize BaseExistanceFilter object.

        Args:
            target_lst: List of targets of interest to be preserved
            remove_specified: Remove specified tarets if True. Otherwise,
                preserve the specified targets and remove the unspecified ones.

        """
        super().__init__()
        self.target_lst = target_lst
        self.remove_specified = remove_specified

    def criterion(self, val):
        if self.remove_specified:
            return val in self.target_lst
        else:
            return val not in self.target_lst


class EntityExistanceFilter(BaseExistanceFilter):
    """Filter entities by list of entiteis of interest.

    Example:
        The following example removes any entities in the labelset_collection
        that are not present in the specified entity_id_list.

        >>> existance_filter = EntityExistanceFilter(entity_id_list)
        >>> labelset_collection.apply(existance_filter, inplace=True)

        Alternatively, can preserve (instead of remove) only eneities not
        present in the entity_id_list by setting ``remove_specified=True``.

    """

    def __init__(
        self,
        target_lst: List[str],
        remove_specified: bool = False,
    ) -> None:
        """Initialize EntityExistanceFilter object."""
        super().__init__(target_lst, remove_specified)

    @staticmethod
    def get_val_getter(lsc):
        return lambda x: x  # return entity ID itself

    @staticmethod
    def get_ids(lsc):
        return lsc.entity.lst

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.pop_entity


class LabelsetExistanceFilter(BaseExistanceFilter):
    """Filter labelset by list of labelsets of interest.

    Example:
        The following example removes any labelset in the labelset_collection
        that has a label name matching any of the element in label_name_list

        >>> labelset_existance_filter = LabelsetExistanceFilter(label_name_list)
        >>> labelset_collection.apply(labelset_existance_filter, inplace=True)

        Alternatively, can preserve (intead of remove) only labelsets not
        present in the label_name_list by setting ``remove_specified=True``.

    """

    def __init__(
        self,
        target_lst: List[str],
        remove_specified: bool = False,
    ):
        """Initialize LabelsetExistanceFilter object."""
        super().__init__(target_lst, remove_specified)

    @staticmethod
    def get_val_getter(lsc):
        return lambda x: x  # return labelset ID itself

    @staticmethod
    def get_ids(lsc):
        return lsc.label_ids

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.pop_labelset


class RangeFilter(BaseFilter):
    """Filter entities in labelset collection by range of values.

    Notes:
        If `None` specified for `min_val` or `max_val`, no filtering
        will be done on upper/lower bound.

    Attributes:
        min_val: minimum below which entities are removed
        max_val: maximum beyound which entiteis are removed

    """

    def __init__(self, min_val=None, max_val=None):
        """Initialize RangeFilter object."""
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def criterion(self, val):
        if self.min_val is not None:
            if val < self.min_val:
                return True
        if self.max_val is not None:
            if val > self.max_val:
                return True
        return False


class EntityRangeFilterNoccur(RangeFilter):
    """Pop entities based on number of occurance."""

    def __init__(self, min_val=None, max_val=None):
        """Initialize EntityRangeFilterNoccur object."""
        super().__init__(min_val, max_val)

    @staticmethod
    def get_val_getter(lsc):
        return lsc.get_noccur

    @staticmethod
    def get_ids(lsc):
        return lsc.entity.lst

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.pop_entity


class LabelsetRangeFilterSize(RangeFilter):
    """Pop labelsets based on size."""

    def __init__(self, min_val=None, max_val=None):
        """Initialize LabelsetRangeFilterSize object."""
        super().__init__(min_val, max_val)

    @staticmethod
    def get_val_getter(lsc):
        return lambda entity_id: len(lsc.get_labelset(entity_id))

    @staticmethod
    def get_ids(lsc):
        return lsc.label_ids

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.pop_labelset


class LabelsetRangeFilterJaccard(RangeFilter):
    """Pop labelsets based on Jaccard index."""

    def __init__(self, max_val=None):
        """Initialize LabelsetRangeFilterJaccard object."""
        super().__init__(None, max_val)

    @staticmethod
    def get_val_getter(lsc):
        # if jjaccard index greater than threshold, determin whether or not to
        # discard depending on the size, only discard the larger one
        def val_getter(label_id):
            val = 0
            labelset = lsc.get_labelset(label_id)
            for label_id2 in lsc.label_ids:
                if label_id2 == label_id:  # skip self
                    continue
                labelset2 = lsc.get_labelset(label_id2)
                jidx = len(labelset & labelset2) / len(labelset | labelset2)
                if (jidx > val) & (len(labelset) <= len(labelset2)):
                    val = jidx
            return val

        return val_getter

    @staticmethod
    def get_ids(lsc):
        return lsc.label_ids

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.pop_labelset


class LabelsetRangeFilterTrainTestPos(RangeFilter):
    """Pop labelsets based on number of positives in train/test sets.

    Note:
        Only intended to be work with Holdout split type for now. Would not
            raise error for other split types, but only will check the first
            split. If validation set is available, will also check the
            validation split.

    """

    def __init__(self, min_val):
        """Initialize LabelsetRangeFilterTrainTestPos object."""
        super().__init__(min_val=min_val)

    @staticmethod
    def get_val_getter(lsc):
        return lambda label_id: min(
            idx_ary.size
            for idx_ary in next(
                lsc.valsplit.get_split_idx_ary(
                    np.array(list(lsc.get_labelset(label_id))),
                    valid=lsc.valsplit.valid_index is not None,
                ),
            )
        )

    @staticmethod
    def get_ids(lsc):
        return lsc.label_ids

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.pop_labelset  # replace with soft filter


class ValueFilter(BaseFilter):
    """Filter based on certain values.

    Attributes:
        val: target value
        remove(bool): if true, remove any ID with matched value,
            else remove any ID with mismatched value

    """

    def __init__(self, val, remove=True):
        """Initialize ValueFilter object."""
        super().__init__()
        self.val = val
        self.remove = remove

    def criterion(self, val):
        return True if (val == self.val) is self.remove else False


class NegativeFilterHypergeom(BaseFilter):
    """Filter based on enrichment (hypergeometric test).

    Notes:
        Given a labelset, compare it with all other labelsets and if p-val
        from hypergemetric test less than some threshold, exclude genes
        from that labelset that are not possitive from training/testing sets,
        i.e. set to neutral.

    Attributes:
        p_thresh: p-val threshold used for filtering

    """

    def __init__(self, p_thresh):
        """Initialize NegativeFilterHypergeom object."""
        self.p_thresh = p_thresh

    def __call__(self, lsc):
        label_ids = lsc.label_ids
        num_labelsets = len(label_ids)
        # set of all entities in the labelset collection
        all_entities = set(lsc.entity_ids)

        def get_pval_mat():
            tot_num_entities = len(all_entities)
            pval_mat = np.zeros((num_labelsets, num_labelsets))

            for i in range(num_labelsets):
                label_id1 = label_ids[i]
                labelset1 = lsc.get_labelset(label_id1)
                num_entities = len(labelset1)  # size of first labelset

                for j in range(i + 1, num_labelsets):
                    label_id2 = label_ids[j]
                    labelset2 = lsc.get_labelset(label_id2)

                    k = len(labelset1 & labelset2)  # size of intersection
                    n = len(labelset2)  # size of second labelset

                    pval_mat[i, j] = pval_mat[j, i] = hypergeom.sf(
                        k - 1,
                        tot_num_entities,
                        n,
                        num_entities,
                    )

                    # if k >= 1: # for debugging
                    #     print(
                    #         f"{k=:>3d}, {tot_num_entities=:>5d}, {n=:>5d}, "
                    #         f"{num_entities=:>5d}, {pval=:>.4f}"
                    #     )

            return pval_mat

        pval_mat = get_pval_mat()

        for i, label_id1 in enumerate(label_ids):
            exclude_set = lsc.get_labelset(label_id1).copy()

            for j, label_id2 in enumerate(label_ids):
                if pval_mat[i, j] < self.p_thresh:
                    exclude_set.update(lsc.get_labelset(label_id2))

            negative = list(all_entities - exclude_set)
            lsc.set_negative(list(negative), label_id1)
