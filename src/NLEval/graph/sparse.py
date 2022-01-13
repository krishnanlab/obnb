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

    def add_edge(self, node_id1, node_id2, weight):
        # TODO: default weight = 1
        for node_id in [node_id1, node_id2]:
            # check if node_id exists, add new if not
            if node_id not in self.idmap:
                self.add_id(node_id)
        try:
            old_weight = self._edge_data[self.idmap[node_id1]][
                self.idmap[node_id2]
            ]
            if old_weight != weight:  # check if edge exists
                print(
                    f"WARNING: edge between {self.idmap[node_id1]} and "
                    f"{self.idmap[node_id2]} exists with weight {old_weight:.2f}"
                    f", overwriting with {weight:.2f}",
                )
        except KeyError:
            self._edge_data[self.idmap[node_id1]][self.idmap[node_id2]] = weight
            if not self.directed:
                self._edge_data[self.idmap[node_id2]][
                    self.idmap[node_id1]
                ] = weight

    def get_edge(self, node_id1, node_id2):
        try:
            return self.edge_data[self.idmap[node_id1]][self.idmap[node_id2]]
        except KeyError:
            return 0

    @staticmethod
    def edglst_reader(edg_fp, weighted, directed, cut_threshold):
        """Edge list file reader.

        Read line by line from a edge list file and yield node_id1, node_id2, weight

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

        """
        graph = cls(weighted=True, directed=True)
        for i in ids:
            graph.add_id(i)
        for i, j in zip(*np.where(mat != 0)):
            graph.add_edge(graph.idmap.lst[i], graph.idmap.lst[j], mat[i, j])
        return graph

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
