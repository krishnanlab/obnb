import numpy as np
from scipy.stats import hypergeom

from .base import BaseFilter


class NegativeGeneratorHypergeom(BaseFilter):
    """Filter based on enrichment (hypergeometric test).

    Given a labelset, it compares all pairs of labelsets via hypergometric
    test. If the p-val is less than ``p_thresh``, then exclude the entities
    from that labelset that are not possitive from training/testing sets,
    i.e. set to neutral.

    Example:
        The following example set up the negatives for each labelset using
        0.05 p-value threshold.

        >>> labelset_collection.apply(NegativeFilterHypergeom(0.05),
        >>>                           inplace=True)

    """

    def __init__(self, p_thresh: float) -> None:
        """Initialize NegativeFilterHypergeom object.

        Args:
            p_thresh: p-val threshold of the hypergeometric test.

        """
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
