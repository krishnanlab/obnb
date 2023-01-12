from functools import partial
from itertools import combinations

from nleval.graph import SparseGraph
from nleval.label.collection import LabelsetCollection
from nleval.label.filters.base import BaseFilter
from nleval.typing import List, Set, Tuple


class LabelsetNonRedFilter(BaseFilter):
    """Filter out redundant labelsets in a labelset collection.

    The detailed procedure can be found in the supplementary data of
    https://doi.org/10.1093/bioinformatics/btaa150 In brief, given a labelset
    collection, a graph of labelsets if first constructed based on the
    redundancy score function of interest. Here, we use the combination of
    Jaccard index and overlap coefficient. Then, for each connected component in
    this graph, retreieve representative labelsets according to the sum of the
    proportions of genes in a geneset that is contained in any other gene sets
    within that component.

    """

    def __init__(self, *thresholds: Tuple[float, float], **kwargs):
        """Initialize BaseLabelsetNonRedFilter object.

        Args:
            thresholds: Thresholds for Jaccard index and overlap coefficient,
                respectively. If a pair of genesets have Jaccard index and
                overlap coefficient above the specified threshold
                simultaneously, then an edge is added connecting the two gene
                sets. Accept values within [0, 1].
            inclusive: Whether or not to include value exactly at the threshold
                when constructing the labelset graph.

        """
        super().__init__(**kwargs)
        self.thresholds = thresholds

    @property
    def params(self) -> List[str]:
        """Parameter list."""
        return ["thresholds"]

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
    def thresholds(self):
        return self._thresholds

    @thresholds.setter
    def thresholds(self, thresholds):
        if len(thresholds) != 2:
            raise ValueError(
                "Expect two thresholds, one for Jaccard index and the other "
                f"one for overlap coefficient, got {len(thresholds)} instead",
            )
        for threshold in thresholds:
            if not isinstance(threshold, float):
                raise TypeError(
                    f"threshold must be a pair of float type, got {type(threshold)}",
                )
            elif not 0 <= threshold <= 1:
                raise ValueError(f"threshold must be within [0, 1], got {threshold}")
        self._thresholds = thresholds

    @staticmethod
    def compute_redundancy(labelset1: Set[str], labelset2: Set[str]) -> float:
        raise NotImplementedError

    def construct_labelset_graph(self, lsc):
        g = SparseGraph(weighted=False, directed=False, logger=self.logger)
        g.add_nodes(self.get_ids(lsc))

        for label_id_pair in combinations(self.get_ids(lsc), 2):
            labelset_pair = list(map(lsc.get_labelset, label_id_pair))
            jaccard, overlap = _compute_jaccard_overlap(*labelset_pair)
            if (jaccard > self.thresholds[0]) & (overlap > self.thresholds[1]):
                g.add_edge(*label_id_pair)

        return g

    @staticmethod
    def _get_repr_score(labelsets: List[Set[str]], idx: int) -> float:
        """Compute the representative score of a gene set in the component.

        For a given gene set in the component of gene sets, the representative
        score is the sum of the ratios of genes in any other gene sets that are
        contained in this gene set.

        """
        current_labelset = labelsets[idx]
        all_execpt_current = [labelsets[i] for i in range(len(labelsets)) if i != idx]
        return sum(len(current_labelset & i) / len(i) for i in all_execpt_current)

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
        comps = g.connected_components()
        self.logger.debug(f"{len(comps):<4}{list(map(len, comps))[:20]}")
        for idx, component in enumerate(g.connected_components()):
            # TODO: add stack info to logger
            self.logger.debug(f"iter {idx}, {component=!r}")

            if (comp_size := len(component)) == 1:
                # Singleton node indicates that it is a representative labelset
                nonred_label_ids.update(component)
            else:
                # Determine the representative labelset to use
                labelsets = list(map(lsc.get_labelset, component))
                get_repr_score = partial(self._get_repr_score, labelsets)
                r = list(map(get_repr_score, range(comp_size)))
                self.logger.debug(f"Redundant ratios {r=!r}")

                # Sort by redundant ratios first, then labelset sizes
                sorted_idx = sorted(range(comp_size), key=r.__getitem__, reverse=True)
                self.logger.debug(f"Sorted index = {sorted_idx}")

                nonred_label_id = component.pop(sorted_idx[0])
                nonred_label_ids.add(nonred_label_id)
                self.logger.debug(f"{nonred_label_id=}, {component=!r}")

                nbrs = g.get_neighbors(nonred_label_id)
                remaining = list(set(component).difference(nbrs))
                # Only call itself if the remaining component is not empty
                if len(remaining) > 0:
                    subgraph = g.induced_subgraph(remaining)
                    nonred_label_ids.update(self.get_nonred_label_ids(subgraph, lsc))

        return nonred_label_ids

    def get_val_getter(self, lsc):
        # TODO: add progress bar to the preprocessing step
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


def _compute_jaccard_overlap(set1, set2):
    num_intersect = len(set1 & set2)
    jaccard = num_intersect / len(set1 | set2)
    overlap = num_intersect / min(map(len, (set1, set2)))
    return jaccard, overlap
