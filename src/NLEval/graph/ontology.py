import functools
import itertools
import logging
from collections import defaultdict
from contextlib import contextmanager

from tqdm import trange

from NLEval.graph.sparse import DirectedSparseGraph
from NLEval.typing import (
    DefaultDict,
    Iterable,
    Iterator,
    List,
    LogLevel,
    Optional,
    Set,
    Term,
    TextIO,
    Union,
)
from NLEval.util import idhandler
from NLEval.util.exceptions import OboTermIncompleteError


class OntologyGraph(DirectedSparseGraph):
    """Ontology graph.

    An ontology graph is a directed acyclic graph (DAG). Here, we represent
    this data type using DirectedSparseGraph, which keeps track of both the
    forward direction of edges (``_edge_data``) and the reversed direction of
    edges (``_rev_edge_data``). This bidirectional awareness is useful in the
    context of propagating information "upwards" or "downloads".

    The ``idmap`` attribute is swapped with a more functional ``IDProp``
    object that allows the storing of node informations such as name and
    node attributes.

    """

    def __init__(
        self,
        log_level: LogLevel = "WARNING",
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the ontology graph."""
        super().__init__(log_level=log_level, verbose=verbose, logger=logger)
        self.idmap = idhandler.IDprop()
        self.idmap.new_property("node_attr", default_val=None)
        self.idmap.new_property("node_name", default_val=None)
        self._edge_stats: List[int] = []
        self._use_cache: bool = False

    def __hash__(self):
        """Hash the ontology graph based on edge statistics."""
        return 0 if self._use_cache else hash(tuple(self._edge_stats))

    def release_cache(self):
        """Release cache."""
        self._aggregate_node_attrs.cache_clear()
        self._ancestors.cache_clear()

    @contextmanager
    def cache_on_static(self):
        """Use cached values to speed up computation on static ontology.

        Note:
            This should only be used when the ontology graph is stable, meaning
            that no further changes including edge and node addition/removal
            will be introduced. However, node attribute manipulation is ok.

        """
        self._use_cache = True
        try:
            yield
        finally:
            self._use_cache = False
            self.release_cache()

    def ancestors(self, node: Union[str, int]) -> Set[str]:
        """Return the ancestor nodes of a given node.

        Note:
            To enable cache utilization to optimize dynamic programing, execute
            this with the cach_on_static context. Note that this would only be
            done when not more structural changes (node and edge modifications)
            will be introduced throughout the span of this context.

        """
        if self._use_cache:
            return self._ancestors(node)
        else:
            return self._ancestors.__wrapped__(self, node)

    @functools.lru_cache(maxsize=None)  # noqa: B019
    def _ancestors(self, node: Union[str, int]) -> Set[str]:
        node_idx = self.get_node_idx(node)
        if len(self._edge_data[node_idx]) == 0:  # root node
            ancestors_set = set()
        else:
            parents_idx = self._edge_data[node_idx]
            ancestors_set = set.union(
                {self.get_node_id(i) for i in parents_idx},
                *(self.ancestors(i) for i in parents_idx),
            )
        return ancestors_set

    def _new_node_data(self):
        super()._new_node_data()
        self._edge_stats.append(0)

    def add_edge(
        self,
        node_id1: str,
        node_id2: str,
        weight: float = 1.0,
        reduction: Optional[str] = None,
    ):
        super().add_edge(node_id1, node_id2, weight, reduction)
        self._edge_stats[self.idmap[node_id2]] += 1

    def set_node_attr(self, node: Union[str, int], node_attr: List[str]):
        """Set node attribute of a given node.

        Args:
            node (Union[str, int]): Node index (int) or node ID (str).
            node_attr (:obj:`list` of :obj:`str`): Node attributes to set.

        """
        self.idmap.set_property(self.get_node_id(node), "node_attr", node_attr)

    def get_node_attr(self, node: Union[str, int]) -> Optional[List[str]]:
        """Get node attribute of a given node.

        Args:
            node (Union[str, int]): Node index (int) or node ID (str).

        """
        return self.idmap.get_property(self.get_node_id(node), "node_attr")

    def _update_node_attr_partial(
        self,
        node: Union[str, int],
        new_node_attr: Union[List[str], str],
    ):
        """Update the ndoe attributes of a node without reduction and sort."""
        if not isinstance(new_node_attr, list):
            new_node_attr = [new_node_attr]
        if self.get_node_attr(node) is None:
            self.set_node_attr(node, [])
        self.get_node_attr(node).extend(new_node_attr)

    def _update_node_attr_finalize(
        self,
        node: Optional[Union[str, int]] = None,
    ):
        """Finalize the node attributes update by reduction and sort.

        If ``node`` is not set, finalize attributes for all nodes.

        """
        if node is not None:
            node_attr = self.get_node_attr(node)
            if node_attr is not None:
                self.set_node_attr(node, sorted(set(node_attr)))
        else:
            for node_id in self.node_ids:
                self._update_node_attr_finalize(node_id)

    def update_node_attr(
        self,
        node: Union[str, int],
        new_node_attr: Union[List[str], str],
    ):
        """Update node attributes of a given node.

        Can update using a single instance or a lsit of instances.

        Args:
            node (Union[str, int]): Node index (int) or node ID (str).
            new_node_attr (Union[List[str], str]): Node attribute(s) to update.

        """
        self._update_node_attr_partial(node, new_node_attr)
        self._update_node_attr_finalize(node)

    def set_node_name(self, node: Union[str, int], node_name: str):
        """Set the name of a given node.

        Args:
            node (Union[str, int]): Node index (int) or node ID (str).
            node_attr (:obj:`list` of :obj:`str`): Node attributes to set.

        """
        self.idmap.set_property(self.get_node_id(node), "node_name", node_name)

    def get_node_name(self, node: Union[str, int]) -> str:
        """Get the name of a given node.

        Args:
            node (Union[str, int]): Node index (int) or node ID (str).

        """
        return self.idmap.get_property(self.get_node_id(node), "node_name")

    @functools.lru_cache(maxsize=None)  # noqa: B019
    def _aggregate_node_attrs(self, node_idx: int) -> List[str]:
        node_attr: Iterable[str]
        if len(self._rev_edge_data[node_idx]) == 0:  # is leaf node
            node_attr = self.get_node_attr(node_idx) or []
        else:
            children_attrs = [
                self._aggregate_node_attrs(nbr_idx)
                for nbr_idx in self._rev_edge_data[node_idx]
            ]
            self_attrs = self.get_node_attr(node_idx) or []
            node_attr = itertools.chain(*children_attrs, self_attrs)
        return sorted(set(node_attr))

    def complete_node_attrs(self, pbar: bool = False):
        """Node attribute completion by propagation upwards.

        Starting from the leaf node, propagate the node attributes to its
        parent node so that the parent node contains all the node attributes
        from its children, plus its original node attributes. This is done via
        recursion _aggregate_node_attrs.

        Note:
            To enable effective dynamic programing of propagating attributes,
            lru_cache is used to decorate _aggregate_node_attrs. By the end of
            this function run, the cache is cleared to prevent overhead of
            calling __eq__ in the next execution.

        Args:
            pbar (bool): If set to True, display a progress bar showing the
                progress of annotation propagation (default: :obj:`False`).

        """
        pbar = trange(self.size, disable=not pbar)
        pbar.set_description("Propagating annotations")
        with self.cache_on_static():
            for node_idx in pbar:
                self.set_node_attr(
                    node_idx,
                    self._aggregate_node_attrs(node_idx),
                )

    @staticmethod
    def iter_terms(fp: TextIO) -> Iterator[Term]:
        """Iterate over terms from a file pointer and yield OBO terms.

        Args:
            fp (TextIO): File pointer, can be iterated over the lines.

        """
        groups = itertools.groupby(fp, lambda line: line.strip() == "")
        for _, stanza_lines in groups:
            if next(stanza_lines).startswith("[Term]"):
                yield OntologyGraph.parse_stanza_simplified(stanza_lines)

    @staticmethod
    def parse_stanza_simplified(stanza_lines: Iterable[str]) -> Term:
        """Return an OBO term (id, name, xref, is_a) from the stanza.

        Note:
            term_xrefs and term_parents can be None if such information is not
            available. Meanwhile, term_id and term_name will always be
            available; otherwise an exception will be raised.

        Args:
            stanza_lines (Iterable[str]): Iterable of strings (lines), and each
                line contains certain type of information inferred by the line
                prefix. Here, we are only interested in four such items, namely
                "id: " (identifier of the term), "name: " (name of the term),
                "xref: " (cross reference of the term) and "is_a: " (parent(s)
                of the term).

        Raises:
            OboTermIncompleteError: If either term_id or term_name is not
                available.

        """
        term_id = term_name = None
        term_xrefs, term_parents = [], []

        for line in stanza_lines:
            if line.startswith("id: "):
                term_id = line.strip()[4:]
            elif line.startswith("name: "):
                term_name = line.strip()[6:]
            elif line.startswith("xref: "):
                term_xrefs.append(line.strip()[6:])
            elif line.startswith("is_a: "):
                term_parents.append(line.strip()[6:].split(" ! ")[0])

        if term_id is None or term_name is None:
            raise OboTermIncompleteError

        return term_id, term_name, term_xrefs, term_parents

    def read_obo(
        self,
        path: str,
        xref_prefix: Optional[str] = None,
    ) -> Optional[DefaultDict[str, Set[str]]]:
        """Read OBO-formatted ontology.

        Args:
            path (str): Path to the OBO file.
            xref_prefix (str, optional): Prefix of xref to be captured and
                return a dictionary of xref to term_id. If not set, then do
                not capture any xref (default: :obj:`None`).

        """
        xref_to_term_id = None if xref_prefix is None else defaultdict(set)
        with open(path, "r") as f:
            for term in self.iter_terms(f):
                term_id, term_name, term_xrefs, term_parents = term

                self._default_add_node(term_id)

                if self.get_node_name(term_id) is None:
                    self.set_node_name(term_id, term_name)

                if term_parents is not None:
                    for parent_id in term_parents:
                        self.add_edge(term_id, parent_id)

                if xref_prefix is not None and term_xrefs is not None:
                    for xref in term_xrefs:
                        xref_terms = xref.split(":")
                        prefix = xref_terms[0]
                        xref_id = ":".join(xref_terms[1:])
                        if prefix == xref_prefix:
                            xref_to_term_id[xref_id].add(term_id)

        return xref_to_term_id

    @classmethod
    def from_obo(cls, path: str):
        """Construct the ontology graph from an obo file."""
        graph = cls()
        graph.read_obo(path)
        return graph
