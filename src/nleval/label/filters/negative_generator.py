from itertools import combinations

import numpy as np
from scipy.stats import hypergeom
from tqdm import tqdm

from nleval.label.filters.base import BaseFilter
from nleval.typing import List


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

    def __init__(self, p_thresh: float, **kwargs) -> None:
        """Initialize NegativeFilterHypergeom object.

        Args:
            p_thresh: p-val threshold of the hypergeometric test.

        """
        self.p_thresh = p_thresh
        super().__init__(**kwargs)

    @property
    def params(self) -> List[str]:
        """Parameter list."""
        return ["p_thresh"]

    def compute_pval_mat(self, lsc, progress_bar):
        """Compute labelset pairwise hyppergeometric p-val."""
        all_entities = set(lsc.entity_ids)
        tot_num_entities = len(all_entities)
        num_labelsets = len(lsc.label_ids)

        pval_mat = np.zeros((num_labelsets, num_labelsets))
        for i, j in tqdm(
            combinations(range(num_labelsets), 2),
            desc="Computing hypergeometric p-value matrix",
            total=(num_labelsets * (num_labelsets - 1) // 2),
            disable=not progress_bar,
        ):
            label_id1 = lsc.label_ids[i]
            label_id2 = lsc.label_ids[j]
            labelset1 = lsc.get_labelset(label_id1)
            labelset2 = lsc.get_labelset(label_id2)

            num_entities1 = len(labelset1)
            num_entities2 = len(labelset2)
            num_intersect = len(labelset1 & labelset2)

            pval = pval_mat[i, j] = pval_mat[j, i] = hypergeom.sf(
                num_intersect - 1,
                tot_num_entities,
                num_entities2,
                num_entities1,
            )

            if num_intersect > 0:
                self.logger.debug(
                    f"{label_id1}({num_entities1}) vs {label_id2}"
                    f"({num_entities2}) -> {num_intersect=}, "
                    f"{tot_num_entities=}, {pval=:>.4f}",
                )

        return pval_mat, all_entities

    def __call__(self, lsc, progress_bar):
        pval_mat, all_entities = self.compute_pval_mat(lsc, progress_bar)

        pbar = tqdm(lsc.label_ids, disable=not progress_bar)
        pbar.set_description(f"{self!r}")
        for i, label_id1 in enumerate(pbar):
            exclude_set = lsc.get_labelset(label_id1).copy()

            for j, label_id2 in enumerate(lsc.label_ids):
                if pval_mat[i, j] < self.p_thresh:
                    exclude_set.update(lsc.get_labelset(label_id2))

            negative = list(all_entities - exclude_set)
            lsc.set_negative(list(negative), label_id1)
            self.logger.info(
                f"Setting negatives for {label_id1} (num negatives = "
                f"{len(negative)} out of {len(all_entities)})",
            )
