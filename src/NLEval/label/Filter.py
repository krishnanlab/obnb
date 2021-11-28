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

    def __init__(self):
        super(BaseFilter, self).__init__()

    def __call__(self, lsc):
        IDs = self.get_ids(lsc)
        val_getter = self.get_val_getter(lsc)
        mod_fun = self.get_mod_fun(lsc)

        for ID in IDs:
            if self.criterion(val_getter(ID)):
                mod_fun(ID)


class ExistanceFilter(BaseFilter):
    """Filter by existance in some given list of targets.

    Attributes:
        target_lst: list (or set) of targets of interest to be preserved
        remove_existance: boolean value indicating whether or not to remove
            tarets in `target_lst`. If True, remove any target present in the
            `target_lst` from the labelset collection; if False, preserve only
            those target present in the `target_lst`

    """

    def __init__(self, target_lst, remove_existance=False):
        super(ExistanceFilter, self).__init__()
        self.target_lst = target_lst
        self.remove_existance = remove_existance

    def criterion(self, val):
        if self.remove_existance:
            return val in self.target_lst
        else:
            return val not in self.target_lst


class EntityExistanceFilter(ExistanceFilter):
    """Filter entities by list of entiteis of interest."""

    def __init__(self, target_lst, remove_existance=False):
        super(EntityExistanceFilter, self).__init__(
            target_lst,
            remove_existance,
        )

    @staticmethod
    def get_val_getter(lsc):
        return lambda x: x  # return entity ID itself

    @staticmethod
    def get_ids(lsc):
        return lsc.entity.lst

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.popEntity


class LabelsetExistanceFilter(ExistanceFilter):
    """Filter labelset by list of labelsets of interest."""

    def __init__(self, target_lst, remove_existance=False):
        super(LabelsetExistanceFilter, self).__init__(
            target_lst,
            remove_existance,
        )

    @staticmethod
    def get_val_getter(lsc):
        return lambda x: x  # return labelset ID itself

    @staticmethod
    def get_ids(lsc):
        return lsc.labelIDlst

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.popLabelset


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
        super(RangeFilter, self).__init__()
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
        super(EntityRangeFilterNoccur, self).__init__(min_val, max_val)

    @staticmethod
    def get_val_getter(lsc):
        return lsc.getNoccur

    @staticmethod
    def get_ids(lsc):
        return lsc.entity.lst

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.popEntity


class LabelsetRangeFilterSize(RangeFilter):
    """Pop labelsets based on size."""

    def __init__(self, min_val=None, max_val=None):
        super(LabelsetRangeFilterSize, self).__init__(min_val, max_val)

    @staticmethod
    def get_val_getter(lsc):
        return lambda ID: len(lsc.getLabelset(ID))

    @staticmethod
    def get_ids(lsc):
        return lsc.labelIDlst

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.popLabelset


class LabelsetRangeFilterJaccard(RangeFilter):
    """Pop labelsets based on Jaccard index."""

    def __init__(self, max_val=None):
        super(LabelsetRangeFilterJaccard, self).__init__(None, max_val)

    @staticmethod
    def get_val_getter(lsc):
        # if jjaccard index greater than threshold, determin whether or not to
        # discard depending on the size, only discard the larger one
        def val_getter(labelID):
            val = 0
            labelset = lsc.getLabelset(labelID)
            for labelID2 in lsc.labelIDlst:
                if labelID2 == labelID:  # skip self
                    continue
                labelset2 = lsc.getLabelset(labelID2)
                jidx = len(labelset & labelset2) / len(labelset | labelset2)
                if (jidx > val) & (len(labelset) <= len(labelset2)):
                    val = jidx
            return val

        return val_getter

    @staticmethod
    def get_ids(lsc):
        return lsc.labelIDlst

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.popLabelset


class LabelsetRangeFilterTrainTestPos(RangeFilter):
    """Pop labelsets based on number of positives in train/test sets.

    Note:
        Only intended to be work with Holdout split type for now. Would not
            raise error for other split types, but only will check the first
            split. If validation set is available, will also check the
            validation split.

    """

    def __init__(self, min_val):
        super(LabelsetRangeFilterTrainTestPos, self).__init__(min_val=min_val)

    @staticmethod
    def get_val_getter(lsc):
        return lambda labelID: min(
            [
                idx_ary.size
                for idx_ary in next(
                    lsc.valsplit.get_split_idx_ary(
                        np.array(list(lsc.getLabelset(labelID))),
                        valid=lsc.valsplit.valid_ID_ary is not None,
                    ),
                )
            ],
        )

    @staticmethod
    def get_ids(lsc):
        return lsc.labelIDlst

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.popLabelset  # replace with soft filter


class ValueFilter(BaseFilter):
    """Filter based on certain values.

    Attributes:
        val: target value
        remove(bool): if true, remove any ID with matched value,
            else remove any ID with mismatched value

    """

    def __init__(self, val, remove=True):
        super(RangeFilter, self).__init__()
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
        self.p_thresh = p_thresh

    def __call__(self, lsc):
        IDs = lsc.labelIDlst
        num_labelsets = len(IDs)
        # set of all entities in the labelset collection
        all_entities = set(lsc.entityIDlst)

        def get_pval_mat():
            M = len(all_entities)
            pval_mat = np.zeros((num_labelsets, num_labelsets))

            for i in range(num_labelsets):
                ID1 = IDs[i]
                labelset1 = lsc.getLabelset(ID1)
                N = len(labelset1)  # size of first labelset

                for j in range(i + 1, num_labelsets):
                    ID2 = IDs[j]
                    labelset2 = lsc.getLabelset(ID2)

                    k = len(labelset1 & labelset2)  # size of intersection
                    n = len(labelset2)  # size of second labelset

                    pval_mat[i, j] = pval_mat[j, i] = hypergeom.sf(
                        k - 1,
                        M,
                        n,
                        N,
                    )

                    # if k >= 1: # for debugging
                    #     print(
                    #         f"{k=:>3d}, {M=:>5d}, {n=:>5d}, {N=:>5d}, "
                    #         f"{pval=:>.4f}"
                    #     )

            return pval_mat

        pval_mat = get_pval_mat()

        for i, ID1 in enumerate(IDs):
            exclude_set = lsc.getLabelset(ID1).copy()

            for j, ID2 in enumerate(IDs):
                if pval_mat[i, j] < self.p_thresh:
                    exclude_set.update(lsc.getLabelset(ID2))

            negative = list(all_entities - exclude_set)
            lsc.setNegative(list(negative), ID1)
