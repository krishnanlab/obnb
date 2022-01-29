from typing import List
from typing import Optional
from typing import Union

import numpy as np

from ..util import checkers
from ..util.idhandler import IDmap
from .base import BaseGraph


class SparseGraph(BaseGraph):
    """SparseGraph object sotring data as adjacency list."""

    def __init__(self, weighted=True, directed=False):
        """Initialize SparseGraph object."""
        super().__init__()
        self._edge_data = []
        self.weighted = weighted
        self.directed = directed

    @property
    def edge_data(self):
        """:obj:`list` of :obj:`dict`: adjacency list data."""
        return self._edge_data

    @property
    def weighted(self):
        """bool: Whether weights (3rd column in edgelist) are available."""
        return self._weighted

    @property
    def directed(self):
        """bool: Indicate whether edges are directed or not."""
        return self._directed

    @weighted.setter
    def weighted(self, val):
        checkers.checkType("weighted", bool, val)
        self._weighted = val

    @directed.setter
    def directed(self, val):
        checkers.checkType("directed", bool, val)
        self._directed = val

    @property
    def num_edges(self) -> int:
        """int: Number of edges."""
        return sum(len(nbrs) for nbrs in self.edge_data)

    def __getitem__(self, key):
        """Return slices of constructed adjacency matrix.

        Args:
            key(str): key of ID
            key(:obj:`list` of :obj:`str`): list of keys of IDs

        """
        idx = self.idmap[key]
        if isinstance(idx, int):
            fvec = self.construct_adj_vec(idx)
        else:
            fvec_lst = []
            fvec_lst = [self.construct_adj_vec(int(i)) for i in idx]
            fvec = np.asarray(fvec_lst)
        return fvec

    def construct_adj_vec(self, src_idx):
        """Construct and return a specific row vector of the adjacency matrix.

        Args:
            src_idx(int): index of row

        """
        checkers.checkType("src_idx", int, src_idx)
        fvec = np.zeros(self.size)
        for nbr_idx, weight in self.edge_data[src_idx].items():
            fvec[nbr_idx] = weight
        return fvec

    def add_id(self, node_id):
        # TODO: add_ids
        self.idmap.add_id(node_id)
        self._edge_data.append({})

    def add_edge(
        self,
        node_id1: str,
        node_id2: str,
        weight: float = 1.0,
        reduction: Optional[str] = None,
    ):
        # Check reduction type
        if reduction not in [None, "max", "min"]:
            raise ValueError(f"Unknown reduction type {reduction!r}")

        # Check if node_id exists, add new if not
        for node_id in [node_id1, node_id2]:
            if node_id not in self.idmap:
                self.add_id(node_id)

        node_idx1 = self.idmap[node_id1]
        node_idx2 = self.idmap[node_id2]

        # Check duplicated edge weights and apply reduction
        if node_idx2 in self._edge_data[node_idx1]:
            old_weight = self._edge_data[node_idx1][node_idx2]
            if old_weight != weight:  # check if edge weight is modified
                if reduction is None:
                    print(
                        f"WARNING: edge between {self.idmap[node_id1]} and "
                        f"{self.idmap[node_id2]} exists with weight "
                        f"{old_weight:.2f}, overwriting with {weight:.2f}",
                    )
                elif reduction == "max":
                    weight = max(old_weight, weight)
                elif reduction == "min":
                    weight = min(old_weight, weight)

        self._edge_data[node_idx1][node_idx2] = weight
        if not self.directed:
            self._edge_data[node_idx2][node_idx1] = weight

    def get_edge(self, node_id1, node_id2):
        try:
            return self.edge_data[self.idmap[node_id1]][self.idmap[node_id2]]
        except KeyError:
            return 0

    @staticmethod
    def edglst_reader(edg_fp, weighted, directed, cut_threshold):
        """Edge list file reader.

        Read line by line from a edge list file and yield (node_id1, node_id2,
        weight)

        """
        with open(edg_fp, "r") as f:
            for line in f:
                try:
                    node_id1, node_id2, weight = line.split("\t")
                    weight = float(weight)
                    if weight <= cut_threshold:
                        continue
                    if not weighted:
                        weight = float(1)
                except ValueError:
                    node_id1, node_id2 = line.split("\t")
                    weight = float(1)
                node_id1 = node_id1.strip()
                node_id2 = node_id2.strip()
                yield node_id1, node_id2, weight

    @staticmethod
    def npy_reader(mat, weighted, directed, cut_threshold):
        """Numpy reader.

        Load an numpy matrix (either from file path or numpy matrix directly)
        and yield node_id1, node_id2, weight.

        Note:
            The matrix should be in shape (N, N+1), where N is number of nodes.
            The first column of the matrix encodes the node IDs

        """
        if isinstance(mat, str):
            # load numpy matrix from file if provided path instead of numpy matrix
            mat = np.load(mat)
        num_nodes = mat.shape[0]

        for i in range(num_nodes):
            node_id1 = mat[i, 0]

            for j in range(num_nodes):
                node_id2 = mat[j, 0]
                weight = mat[i, j + 1]
                if weight > cut_threshold:
                    try:
                        yield str(int(node_id1)), str(int(node_id2)), weight
                    except TypeError:
                        yield str(node_id1), str(node_id2), weight

    def read(self, file, reader="edglst", cut_threshold=0):
        """Read data and construct sparse graph.

        Args:
            file(str): path to input file
            weighted(bool): if not weighted, all weights are set to 1
            directed(bool): if not directed, automatically add 2 edges
            reader: generator function that yield edges from file
            cut_threshold(float): threshold below which edges are not considered

        TODO: reader part looks sus, check unit test

        """
        for node_id1, node_id2, weight in reader(
            file,
            self.weighted,
            self.directed,
            cut_threshold,
        ):
            self.add_edge(node_id1, node_id2, weight)

    @classmethod
    def from_edglst(cls, path_to_edglst, weighted, directed, cut_threshold=0):
        graph = cls(weighted=weighted, directed=directed)
        reader = cls.edglst_reader
        graph.read(path_to_edglst, reader=reader, cut_threshold=cut_threshold)
        return graph

    @classmethod
    def from_npy(cls, npy, weighted, directed, cut_threshold=0):
        graph = cls(weighted=weighted, directed=directed)
        reader = cls.npy_reader
        graph.read(npy, reader=reader, cut_threshold=cut_threshold)
        return graph

    @classmethod
    def from_mat(
        cls,
        mat,
        ids: Optional[Union[List[str], IDmap]] = None,
    ):  # noqa
        """Construct SparseGraph using ids and adjacency matrix.

        Args:
            mat(:obj:`numpy.ndarray`): 2D numpy array of adjacency matrix
            ids(list or :obj:`IDmap`): list of IDs or idmap of the
                adjacency matrix, if None, use input ordering of nodes as IDs.
                (default: :obj:`None`).

        """
        if ids is None:
            ids = list(map(str, range(mat.shape[0])))
        graph = cls(weighted=True, directed=True)
        for i in ids:
            graph.add_id(i)
        for i, j in zip(*np.where(mat != 0)):
            graph.add_edge(graph.idmap.lst[i], graph.idmap.lst[j], mat[i, j])
        return graph

    @classmethod
    def from_cx_stream_file(cls, path: str, undirected: bool = True, **kwargs):
        """Read from a CX stream file."""
        graph = cls(weighted=True, directed=not undirected)
        graph.read_cx_stream_file(path, **kwargs)
        return graph

    def read_cx_stream_file(
        self,
        path: str,
        interaction_types: Optional[List[str]] = None,
        node_id_prefix: Optional[str] = "ncbigene",
        node_id_entry: str = "r",
        default_edge_weight: float = 1.0,
        edge_weight_attr_name: Optional[str] = None,
        reduction: Optional[str] = "max",
        use_node_alias: bool = False,
    ):
        """Construct SparseGraph from a CX stream file.

        Args:
            path (str): Path to the cx file.
            interaction_types (list of str, optional): Types of interactions to
                be considered if not set, consider all (default: :obj:`None`).
            node_id_prefix (str, optional): Prefix of the ID to be considered,
                if not set, consider all IDs (default: "ncbigene").
            node_id_entry (str): use "n" for name of the entity, or "r" for
                other reprentations (default: "r").
            default_edge_weight (float): edge weight to use if no edge weights
                specified by edge attributes (default: 1.0).
            edge_weight_attr_name (str, optional): name of the edge attribute
                to use for edge weights, must be numeric type, i.e. "d" must
                be "double" or "integer" or "long". If not set, do to use any
                edge attributes (default: :obj:`None`)
            reduction (str, optional): How to due with duplicated edge weights,
                current options are "max" and "min", which uses the maximum or
                minimum value among the duplicated edge weights; alternatively,
                if set to :obj:`None`, then it will use the last edge weight
                seen (default: "max").
            use_node_alias (bool): If set, use node alias as node ID, otherwise
                use the default node aspect for reading node ID. Similar to the
                default node ID option, this requires that the prefix matches
                node_id_prefix if set. Note that when use_node_alias is set,
                the node_id_prefix becomes mandatory.If multiple node ID
                aliases with matching prefix are available, use the first one.
                (defaut: :obj:`False`)

        """
        import json  # noreorder

        if node_id_entry not in ["r", "n"]:
            raise ValueError(f"Unkown node ID entry {node_id_entry!r}")

        with open(path, "r") as f:
            cx_stream = json.load(f)

        entry_map = {list(j)[0]: i for i, j in enumerate(cx_stream)}
        raw_edges = cx_stream[entry_map["edges"]]["edges"]

        # Load node IDs
        node_id_to_idx = {}
        if not use_node_alias:
            raw_nodes = cx_stream[entry_map["nodes"]]["nodes"]
            for node in raw_nodes:
                node_name = node[node_id_entry]
                if node_id_prefix is not None:
                    if not node_name.startswith(node_id_prefix):
                        print(
                            f"Skipping node: {node_name!r} due to mismatch "
                            f"node_id_prefix. {node}",
                        )
                        continue
                    node_name = node_name.split(":")[1]
                node_id_to_idx[node["@id"]] = node_name
        else:
            if node_id_prefix is None:
                raise ValueError(
                    "Must specify node_id_prefix when use_node_alias is set.",
                )
            for na in cx_stream[entry_map["nodeAttributes"]]["nodeAttributes"]:
                if na["n"] == "alias":
                    idx, values = na["po"], na["v"]
                    values = values if isinstance(values, list) else [values]
                    for value in values:
                        if value.startswith(node_id_prefix):
                            node_id_to_idx[idx] = value.split(":")[1]
                            break

        # Load edge weights using the specified edge attribute name
        edge_weight_dict = {}
        if edge_weight_attr_name is not None:
            for ea in cx_stream[entry_map["edgeAttributes"]]["edgeAttributes"]:
                if ea["n"] == edge_weight_attr_name:
                    if not ea["d"] in ["double", "integer", "long"]:
                        raise TypeError(
                            "Only allow numeric type edge attribute to be used"
                            f" as edge weights, got {ea['d']!r}: {ea}",
                        )
                    edge_weight_dict[ea["po"]] = float(ea["v"])

        # Write edges
        for edge in raw_edges:
            try:
                node_id1 = node_id_to_idx[edge["s"]]
                node_id2 = node_id_to_idx[edge["t"]]
                if (
                    interaction_types is not None
                    and edge["i"] not in interaction_types
                ):
                    print(
                        f"Skipping edge: {edge} due to mismatched interaction "
                        f"type with the specified {interaction_types}",
                    )
                    continue

                eid = edge["@id"]
                weight = (
                    edge_weight_dict[eid]
                    if eid in edge_weight_dict
                    else default_edge_weight
                )
                self.add_edge(node_id1, node_id2, weight, reduction=reduction)

            except KeyError:
                print(f"Skipping edge: {edge} due to unkown nodes")

    @staticmethod
    def edglst_writer(outpth, edge_gen, weighted, directed, cut_threshold):
        """Edge list file writer.

        Write line by line to edge list.

        """
        with open(outpth, "w") as f:
            for src_node_id, dst_node_id, weight in edge_gen():
                if weighted:
                    if weight > cut_threshold:
                        f.write(
                            f"{src_node_id}\t{dst_node_id}\t{weight:.12f}\n",
                        )
                else:
                    f.write(f"{src_node_id}\t{dst_node_id}\n")

    @staticmethod
    def npy_writer():
        raise NotImplementedError

    def edge_gen(self):
        edge_data_copy = self._edge_data[:]
        for src_idx in range(len(edge_data_copy)):
            src_nbrs = edge_data_copy[src_idx]
            src_node_id = self.idmap.getID(src_idx)
            for dst_idx in src_nbrs:
                dst_node_id = self.idmap.getID(dst_idx)
                if not self.directed:
                    edge_data_copy[dst_idx].pop(src_idx)
                weight = edge_data_copy[src_idx][dst_idx]
                yield src_node_id, dst_node_id, weight

    def save(self, outpth, writer="edglst", cut_threshold=0):
        """Save graph to file.

        Args:
            outpth (str): path to output file
            writer (str): writer function (or name of default writer) to
                generate file ('edglst', 'npy').
            cut_threshold (float): threshold of edge weights below which the
                edges are not considered.

        """
        if isinstance(writer, str):
            if writer == "edglst":
                writer = self.edglst_writer
            elif writer == "npy":
                writer = self.npy_writer
            else:
                raise ValueError(f"Unknown writer function name {writer!r}")
        writer(
            outpth,
            self.edge_gen,
            self.weighted,
            self.directed,
            cut_threshold,
        )

    def to_adjmat(self, default_val=0):
        """Construct adjacency matrix from edgelist data.

        Args:
            default_val(float): default value for missing edges

        """
        num_nodes = self.idmap.size
        mat = np.ones((num_nodes, num_nodes)) * default_val
        for src_node, src_nbrs in enumerate(self._edge_data):
            for dst_node in src_nbrs:
                mat[src_node, dst_node] = src_nbrs[dst_node]
        return mat

    def to_pyg_edges(self):
        """Convert to Pytorch Geometric edge_index and edge_weight."""
        num_edges = self.num_edges
        edge_index = np.zeros((2, num_edges), dtype=int)
        edge_weight = np.zeros(num_edges, dtype=np.float32)

        start_pos = 0
        for node1, nbrs in enumerate(self.edge_data):
            end_pos = start_pos + len(nbrs)
            slice_ = slice(start_pos, end_pos)

            edge_index[0, slice_] = node1
            nbr_idx = sorted(nbrs)
            edge_index[1, slice_] = nbr_idx
            edge_weight[slice_] = list(map(nbrs.get, nbr_idx))

            start_pos = end_pos

        if not self.weighted:
            edge_weight = None

        return edge_index, edge_weight
