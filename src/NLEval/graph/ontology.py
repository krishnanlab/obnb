import functools
import itertools
from collections import defaultdict
from typing import DefaultDict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import TextIO
from typing import Tuple
from typing import Union

from ..util import idhandler
from ..util.exceptions import OboTermIncompleteError
from .sparse import DirectedSparseGraph

Term = Tuple[str, str, Optional[List[str]], Optional[List[str]]]


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

    def __init__(self):
        """Initialize the ontology graph."""
        super().__init__()
        self.idmap = idhandler.IDprop()
        self.idmap.new_property("node_attr", default_val=None)
        self.idmap.new_property("node_name", default_val=None)

    def __hash__(self):
        """Trivial hash.

        This hash is solely for the sake of enabling LRU cache when calling
        _aggregate_node_attrs recursion.

        """
        return 0

    def get_node_id(self, node: Union[str, int]) -> str:
        """Return the node ID given the node index.

        Args:
            node (Union[str, int]): Node index (int) or node ID (str). If input
                is already node ID, return directly. If input is node index,
                then return the node ID of the corresponding node index.

        Return:
            str: Node ID.

        """
        return node if isinstance(node, str) else self.idmap.lst[node]

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

    @functools.lru_cache(maxsize=None)
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

    def complete_node_attrs(self):
        """Node attribute completion by propagation upwards.

        Starting from the leaf node, propagate the node attributes to its
        parent node so that the parent node contains all the node attributes
        from its children, plus its original node attributes. This is done via
        recursion _aggregate_node_attrs.

        """
        for node_idx in range(self.size):
            self.set_node_attr(node_idx, self._aggregate_node_attrs(node_idx))

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

                self._default_add_id(term_id)

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
