from functools import partial
from itertools import combinations

from NLEval.graph import SparseGraph
from NLEval.label.collection import LabelsetCollection
from NLEval.label.filters.base import BaseFilter
from NLEval.typing import List, Set


class BaseLabelsetNonRedFilter(BaseFilter):
    """Filter out redundant labelsets in a labelset collection.

    The detailed procedure can be found in the supplementary data of
    https://doi.org/10.1093/bioinformatics/btaa150
    In brief, given a labelset collection, a graph of labelsets if first
    constructed based on the redundancy score function of interest (e.g.,
    Jaccard index, overlap coefficient). Then, for each connected component
    in this graph, retreieve representative labelsets according to the
    proportion of the elements in a labelset that is covered in this component.
    Note that in the case of a tie, it will try to prioritize the one that
    have more number of elements.

    """

    def __init__(self, threshold: float, **kwargs):
        """Initialize BaseLabelsetNonRedFilter object.

        Args:
            threshold: Value of the redundancy score between a pair of labelset
                above which the labelset pair is considered connected
                in a labelset graph. Accept values with in [0, 1].
            inclusive: Whether or not to include value exactly at the threshold
                when constructing the labelset graph.

        """
        super().__init__(**kwargs)
        self.threshold = threshold

    @property
    def mode_name(self):
        return "DROP LABELSET"

    @staticmethod
    def get_ids(lsc):
        return lsc.label_ids

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.pop_labelset

    @property
    def mod_name(self):
        return "DROP LABELSET"

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        if not isinstance(threshold, float):
            raise TypeError(f"'threshold' must be float type, got {type(threshold)}")
        elif 0 <= threshold <= 1:
            self._threshold = threshold
        else:
            raise ValueError(f"'threshold' must be within [0, 1], got {threshold}")

    @staticmethod
    def compute_redundancy(labelset1: Set[str], labelset2: Set[str]) -> float:
        raise NotImplementedError

    def construct_labelset_graph(self, lsc):
        t = self.threshold
        g = SparseGraph(weighted=False, directed=False, logger=self.logger)
        # TODO: change to g.add_nodes
        _ = list(map(g.add_id, self.get_ids(lsc)))
        for label_id_pair in combinations(self.get_ids(lsc), 2):
            red = self.compute_redundancy(*map(lsc.get_labelset, (label_id_pair)))
            if red > t:
                self.logger.debug(f"Add edge: {label_id_pair=}, {red=:.3f}")
                g.add_edge(*label_id_pair)
        return g

    @staticmethod
    def _get_redundant_ratio(labelsets: List[Set[str]], idx: int) -> float:
        """Compute the ratio of elements in a set that is in some other sets."""
        current_labelset = labelsets[idx]
        all_execpt_idx = [i for i in range(len(labelsets)) if i != idx]
        others = set.union(*(labelsets[i] for i in all_execpt_idx))
        common = current_labelset.intersection(others)
        return len(common) / len(current_labelset)

    def get_nonred_label_ids(
        self,
        g: SparseGraph,
        lsc: LabelsetCollection,
    ) -> Set[str]:
        """Extract non-redundant labelsets.

        Args:
            g: The labelset graph connecting different labelsets according to
                the extend they are redundant.
                See :meth:`construct_labelset_graph`.
            lsc: The labelset collection object.

        Returns:
            set[str]: The set of non-redundant labelset IDs.

        """
        nonred_label_ids = set()
        for idx, component in enumerate(g.connected_components()):
            # TODO: add stack info to logger
            self.logger.debug(f"iter {idx}, {component=!r}")

            if (comp_size := len(component)) == 1:
                # Singleton node indicates that it is a representative labelset
                nonred_label_ids.update(component)
            else:
                # Determine the representative labelset to use
                labelsets = list(map(lsc.get_labelset, component))
                get_redundant_ratio = partial(self._get_redundant_ratio, labelsets)
                r = list(map(get_redundant_ratio, range(comp_size)))
                s = list(map(len, labelsets))
                self.logger.debug(f"Redundant ratios {r=!r}")
                self.logger.debug(f"Labelset sizes {s=!r}")

                # Sort by redundant ratios first, then labelset sizes
                sorted_idx = sorted(range(comp_size), key=list(zip(r, s)).__getitem__)
                self.logger.debug(f"Sorted index = {sorted_idx}")

                nonred_label_id = component.pop(sorted_idx[-1])
                self.logger.debug(f"{nonred_label_id=}, {component=!r}")

                # TODO: change to g.get_nbrs
                nbrs = {g.idmap.lst[i] for i in g.edge_data[g.idmap[nonred_label_id]]}
                component = list(set(component).difference(nbrs))

                # Extract representative labelsets from the rest of the labelsets
                nonred_label_ids.update(
                    self.get_nonred_label_ids(g.induced_subgraph(component), lsc)
                    | {nonred_label_id},
                )

        return nonred_label_ids

    def get_val_getter(self, lsc):
        # Extract non-redundant labelset ids and then remove anything outside
        # of this sed
        g = self.construct_labelset_graph(lsc)

        nonred_label_ids = self.get_nonred_label_ids(g, lsc)
        self.logger.debug(f"{nonred_label_ids=}")

        def val_getter(label_id):
            return label_id not in nonred_label_ids

        return val_getter

    def criterion(self, val: bool) -> bool:
        return val


class LabelsetNonRedFilterJaccard(BaseLabelsetNonRedFilter):
    """Filter redundant labelsets based on Jaccard index.

    The Jaccard index is computed as the size of the intersection divided by
    the size of the union of two sets.

    Example:
        >>> labelset_collection.iapply(LabelsetNonRedFilterJaccard(0.7))

    """

    @staticmethod
    def compute_redundancy(labelset1, labelset2):
        return len(labelset1 & labelset2) / len(labelset1 | labelset2)


class LabelsetNonRedFilterOverlap(BaseLabelsetNonRedFilter):
    """Filter redundant labelsets based on Overlap coefficient.

    The Overlap coefficient is computed as the size of the intersection divided
    by the minimum size of the two sets.

    Example:
        >>> labelset_collection.iapply(LabelsetNonRedFilterOverlap(0.8))

    """

    @staticmethod
    def compute_redundancy(labelset1, labelset2):
        return len(labelset1 & labelset2) / min(map(len, (labelset1, labelset2)))
