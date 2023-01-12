import itertools
import logging

import numpy as np

from nleval.exception import EdgeNotExistError, IDNotExistError
from nleval.graph.base import BaseGraph
from nleval.typing import EdgeData, EdgeDir, List, LogLevel, Mapping, Optional, Union
from nleval.util import checkers
from nleval.util.cx_explorer import CXExplorer
from nleval.util.idhandler import IDmap


class SparseGraph(BaseGraph):
    """SparseGraph object storing data as adjacency list."""

    def __init__(
        self,
        weighted: bool = True,
        directed: bool = False,
        self_loops: bool = False,
        log_level: LogLevel = "WARNING",
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize SparseGraph object."""
        super().__init__(log_level=log_level, verbose=verbose, logger=logger)
        self._edge_data: EdgeData = []
        self.weighted = weighted
        self.directed = directed
        self.self_loops = self_loops

    @property
    def edge_data(self) -> EdgeData:
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

    def __getitem__(self, key: Union[str, List[str]]):
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

    def _get_nbr_idxs(self, node_idx: int, direction: EdgeDir) -> List[int]:
        if self.directed and direction != "out":
            raise NotImplementedError("Use DirectedSparseGraph instead.")
        return sorted(self.edge_data[node_idx])

    def induced_subgraph(self, node_ids: List[str]):
        """Return a subgraph induced by a subset of nodes.

        Args:
            node_ids (List[str]): List of nodes of interest.

        """
        graph = SparseGraph(
            weighted=self.weighted,
            directed=self.directed,
            self_loops=self.self_loops,
            logger=self.logger,
        )

        # Add nodes to new graph and make sure all nodes are present
        old_idx_to_new_idx = {}
        for new_idx, node in enumerate(node_ids):
            if node not in self.idmap:
                raise IDNotExistError(f"{node!r} is not in the graph")
            graph.add_node(node)
            old_idx_to_new_idx[self.idmap[node]] = new_idx

        # Map edge data to the new graph
        for node1 in node_ids:
            node1_idx = self.idmap[node1]
            graph._edge_data[old_idx_to_new_idx[node1_idx]] = {
                old_idx_to_new_idx[node2_idx]: weight
                for node2_idx, weight in self.edge_data[node1_idx].items()
                if node2_idx in old_idx_to_new_idx
            }

        return graph

    def connected_components(self) -> List[List[str]]:
        """Find connected components via Breadth First Search.

        Returns a list of connected components sorted by the number of nodes, each of
        which is a list of node ids within a connected component.

        """
        unvisited = set(range(self.num_nodes))
        connected_components = []

        while unvisited:
            visited = set()
            tovisit = {unvisited.pop()}

            while tovisit:
                visited.update(tovisit)
                tovisit_next = itertools.chain.from_iterable(
                    [self._edge_data[i] for i in tovisit],
                )
                tovisit = set(tovisit_next).difference(visited)

            unvisited.difference_update(visited)
            connected_components.append([self.idmap.lst[i] for i in visited])

        return sorted(connected_components, key=len, reverse=True)

    def construct_adj_vec(self, src_idx: int):
        """Construct and return a specific row vector of the adjacency matrix.

        Args:
            src_idx(int): index of row

        """
        checkers.checkType("src_idx", int, src_idx)
        fvec = np.zeros(self.size)
        for nbr_idx, weight in self.edge_data[src_idx].items():
            fvec[nbr_idx] = weight
        return fvec

    def _new_node_data(self):
        self._edge_data.append({})

    def _add_edge(
        self,
        node_id1: str,
        node_id2: str,
        weight: float,
        reduction: Optional[str],
        edge_data: EdgeData,
    ):
        """Update edge data.

        Note:
            ``edge_data`` is being passed in for more flexibility in choosing
            which edge_data to be modieifed. For example, in the directed
            graph case, where reversed edge data is present for the sake of
            reversed propagation, one can specify to use _add_edge to update
            the reversed edge data.

        Args:
            node_id1 (str): ID of node 1.
            node_id2 (str): ID of node 2.
            weight (float): Edge weight to use.
            reduction (str): Type of edge reduction to use if edge already
                existsed, if not set, warn if old edge exists with different
                values and overwrite it with the new avlue.
            edge_data: The edge data of the sparse graph, in the form of an
                adjacency list with edge weights.

        """
        node_idx1, node_idx2 = self.idmap[node_id1], self.idmap[node_id2]

        # Check self loops
        if not self.self_loops and node_idx1 == node_idx2:
            return

        # Check duplicated edge weights and apply reduction
        if node_idx2 in edge_data[node_idx1]:
            old_weight = edge_data[node_idx1][node_idx2]
            if old_weight != weight:  # check if edge weight is modified
                if reduction is None:
                    self.logger.warning(
                        f"edge between {node_id1} and {node_id2} exists with "
                        f"weight {old_weight:.2f}, overwriting with it with "
                        f"{weight:.2f}",
                    )
                elif reduction == "max":
                    weight = max(old_weight, weight)
                elif reduction == "min":
                    weight = min(old_weight, weight)

        edge_data[node_idx1][node_idx2] = weight
        if not self.directed:
            edge_data[node_idx2][node_idx1] = weight

    def add_edge(
        self,
        node_id1: str,
        node_id2: str,
        weight: float = 1.0,
        reduction: Optional[str] = None,
    ):
        """Add or update an edge in the sparse graph.

        Args:
            node_id1 (str): ID of node 1.
            node_id2 (str): ID of node 2.
            weight (float): Edge weight to use.
            reduction (str, optional): Type of edge reduction to use if edge
                already existsed. If it is not set, warn if old edge exists
                with different values and overwrite it with the new avlue.
                Possible options are 'min', 'max', and :obj:`None` (default:
                :obj:`None`).

        """
        # Check reduction type
        if reduction not in [None, "max", "min"]:
            raise ValueError(f"Unknown reduction type {reduction!r}")

        self.add_nodes([node_id1, node_id2], exist_ok=True)
        self._add_edge(
            node_id1,
            node_id2,
            weight,
            reduction,
            self._edge_data,
        )

    def get_edge(self, node_id1, node_id2):
        try:
            return self.edge_data[self.idmap[node_id1]][self.idmap[node_id2]]
        except KeyError:
            return 0

    def remove_edge(self, node_id1: str, node_id2: str):
        """Remove an edge in the graph.

        Args:
            node_id1: ID of node 1.
            node_id2: ID of node 2.

        """
        self._remove_edge(node_id1, node_id2, self.edge_data)
        if not self.directed:
            self._remove_edge(node_id2, node_id1, self.edge_data)

    def _remove_edge(self, node_id1: str, node_id2: str, edge_data: EdgeData):
        node_idx1 = self.get_node_idx(node_id1)
        node_idx2 = self.get_node_idx(node_id2)

        try:
            edge_data[node_idx1].pop(node_idx2)
        except KeyError:
            raise EdgeNotExistError(f"The edge {node_id1}-{node_id2} does not exist.")

    @staticmethod
    def edglst_reader(edg_path, weighted, directed, cut_threshold):
        """Edge list file reader.

        Read line by line from a edge list file and yield (node_id1, node_id2, weight)

        """
        with open(edg_path) as f:
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
        **kwargs,
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
            graph.add_node(i)
        for i, j in zip(*np.where(mat != 0)):
            graph.add_edge(graph.idmap.lst[i], graph.idmap.lst[j], mat[i, j])
        return graph

    @classmethod
    def from_cx_stream_file(
        cls,
        path: str,
        directed: bool = False,
        self_loops: bool = False,
        **kwargs,
    ):
        """Construct SparseGraph from a CX stream file."""
        graph = cls(
            weighted=True,
            directed=directed,
            self_loops=self_loops,
        )
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
        node_id_converter: Optional[Mapping[str, str]] = None,
    ):
        """Read from a CX stream file.

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
                (default: :obj:`False`)
            node_id_converter (Mapping[str, str], optional): A mapping object
                that maps a given node ID to a new node ID of interest.

        """
        import json  # noreorder

        if node_id_entry not in ["r", "n"]:
            raise ValueError(f"Unknown node ID entry {node_id_entry!r}")

        self.logger.info("Loading raw cx file")
        with open(path) as f:
            cx_data = CXExplorer.from_cx_stream(json.load(f))

        # Load node IDs
        self.logger.info("Loading nodes")
        node_idx_to_id = {}
        if not use_node_alias:
            for node in cx_data["nodes"]:
                node_name = node[node_id_entry]
                if node_id_prefix is not None:
                    if not node_name.startswith(node_id_prefix):
                        self.logger.debug(
                            f"Skipping node {node_name!r} due to mismatch "
                            f"node_id_prefix {node}",
                        )
                        continue
                    node_name = node_name.split(":")[1]
                node_idx_to_id[node["@id"]] = node_name
        else:
            if node_id_prefix is None:
                raise ValueError(
                    "Must specify node_id_prefix when use_node_alias is set.",
                )
            for na in cx_data["nodeAttributes"]:
                if na["n"] == "alias":
                    idx, values = na["po"], na["v"]
                    values = values if isinstance(values, list) else [values]
                    for value in values:
                        if value.startswith(node_id_prefix):
                            node_idx_to_id[idx] = value.split(":")[1]
                            break

        # Convert node IDs
        if node_id_converter is not None:
            self.logger.info("Start converting gene IDs.")
            log_level_id = logging.getLevelName(self.log_level)

            try:
                node_id_converter.logger.setLevel(log_level_id)
                node_id_converter.query_bulk(list(node_idx_to_id.values()))
            except AttributeError:
                pass

            node_idx_to_id_converted = {}
            for i, j in node_idx_to_id.items():
                node_id_converted = node_id_converter[j]
                if node_id_converted is not None:
                    node_idx_to_id_converted[i] = node_id_converted

            self.logger.info("Done converting gene IDs.")
        else:
            node_idx_to_id_converted = node_idx_to_id

        # Load edge weights using the specified edge attribute name
        self.logger.info("Reading edges")
        edge_weight_dict = {}
        if edge_weight_attr_name is not None:
            for ea in cx_data["edgeAttributes"]:
                if ea["n"] == edge_weight_attr_name:
                    try:
                        edge_weight_dict[ea["po"]] = float(ea["v"])
                    except ValueError:
                        self.logger.debug(
                            f"Skipping edge attr: {ea} due to value error",
                        )

        # Write edges
        self.logger.info("Loading edges to graph")
        for edge in cx_data["edges"]:
            try:
                node_id1 = node_idx_to_id_converted[edge["s"]]
                node_id2 = node_idx_to_id_converted[edge["t"]]
                if interaction_types is not None and edge["i"] not in interaction_types:
                    self.logger.debug(
                        f"Skipping edge {edge} due to mismatched interaction "
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
                self.logger.debug(
                    f"Skipping edge: {edge} due to unknown nodes",
                )

    @classmethod
    def from_npz(cls, path, weighted, directed=False, **kwargs):
        """Construct SparseGraph from a npz file."""
        graph = cls(weighted=weighted, directed=directed)
        graph.read_npz(path, **kwargs)
        return graph

    def read_npz(
        self,
        path: str,
        cut_threshold: Optional[float] = None,
    ):
        """Read from npz file.

        The file contains two fields: "edge_index" and "node_ids", where the
        first is a 2 x m numpy array encoding the m edges, and the second
        is a one dimensional numpy array of str type encoding the node IDs.
        Additionally, "edge_weight" is available if the graph is weighted,
        which is a one dimensional numpy array (of size m) of edge weights.

        Note:
            The ``weighted`` and ``directed`` options are for compatibility
                to the SparseGraph object.

        Args:
            path (str): path to the .npz file
            cut_threshold (float, optional): threshold of edge weights below
                which the edges are ignored, if not set, consider all edges
                (default: :obj:`None`).

        """
        files = np.load(path)
        node_ids = files["node_ids"].tolist()
        edge_index = files["edge_index"]

        self.idmap = self.idmap.from_list(node_ids)
        self._edge_data = [{} for _ in range(len(node_ids))]

        if self.weighted:
            edge_weight = files["edge_weight"]
            for (i, j), w in zip(edge_index.T, edge_weight):
                self._edge_data[i][j] = w
        else:
            for i, j in edge_index.T:
                self._edge_data[i][j] = 1.0

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

    def save_npz(self, out_path: str, weighted: bool = True):
        """Save the graph as npz.

        The npz file contains three fields, including "edge_index",
            "edge_weight", and "node_ids". If the the ``weighted`` option is
            set to :obj:`False`, then the "edge_weight" would not be saved.

        Args:
            out_path (str): path to the output file.
            weighted (bool): whether should save the edge weights
                (default: :obj:`True`).

        """
        edge_index = np.zeros((2, self.num_edges), dtype=np.uint32)
        edge_weight = np.zeros(self.num_edges, dtype=np.float32)
        node_ids = np.array(self.idmap.lst)

        idx = 0
        for i, nbrs in enumerate(self._edge_data):
            for nbr, weight in nbrs.items():
                edge_index[0, idx] = i
                edge_index[1, idx] = nbr
                edge_weight[idx] = weight
                idx += 1

        if weighted:
            np.savez(
                out_path,
                edge_index=edge_index,
                edge_weight=edge_weight,
                node_ids=node_ids,
            )
        else:
            np.savez(out_path, edge_index=edge_index, node_ids=node_ids)

    def edge_gen(self):
        edge_data_copy = self._edge_data[:]
        for src_idx in range(len(edge_data_copy)):
            src_nbrs = edge_data_copy[src_idx]
            src_node_id = self.idmap.get_id(src_idx)
            for dst_idx in src_nbrs:
                dst_node_id = self.idmap.get_id(dst_idx)
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
        mat = np.ones((self.num_nodes, self.num_nodes)) * default_val
        for src_node, src_nbrs in enumerate(self._edge_data):
            for dst_node in src_nbrs:
                mat[src_node, dst_node] = src_nbrs[dst_node]
        return mat

    def to_coo(self):
        """Convert to edge_index and edge_weight."""
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

    def to_dense_graph(self):
        """Convert SparseGraph to a DenseGraph."""
        from nleval.graph.dense import DenseGraph  # noreorder

        return DenseGraph.from_mat(self.to_adjmat(), self.idmap)


class DirectedSparseGraph(SparseGraph):
    """Directed sparse graph that also keeps track of reversed edge data.

    The reversed edge data is captured for more efficient "propagation upwards" in
    addition to the more natural "propagation downwards" operation.

    """

    def __init__(
        self,
        weighted: bool = True,
        log_level: LogLevel = "WARNING",
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the directed sparse graoh."""
        super().__init__(
            weighted=weighted,
            directed=True,
            log_level=log_level,
            verbose=verbose,
            logger=logger,
        )
        self._rev_edge_data: EdgeData = []

    @property
    def directed(self) -> bool:
        return True

    @directed.setter
    def directed(self, directed: bool):
        if not directed:
            raise ValueError("{self.__class__.__name__} only allow directed=True")

    @property
    def rev_edge_data(self) -> EdgeData:
        """Adjacency list of reversed edge direction."""
        return self._rev_edge_data

    def _get_nbr_idxs(self, node_idx: int, direction: EdgeDir) -> List[int]:
        out_nbrs_idxs = set(self.edge_data[node_idx])
        in_nbrs_idxs = set(self.rev_edge_data[node_idx])

        if direction == "in":
            return sorted(in_nbrs_idxs)
        elif direction == "out":
            return sorted(out_nbrs_idxs)
        else:
            return sorted(in_nbrs_idxs | out_nbrs_idxs)

    def _new_node_data(self):
        self._edge_data.append({})
        self._rev_edge_data.append({})

    def add_edge(
        self,
        node_id1: str,
        node_id2: str,
        weight: float = 1.0,
        reduction: Optional[str] = None,
    ):
        """Add a directed edge and record in the reversed adjacency list."""
        super().add_edge(node_id1, node_id2, weight, reduction)

        self._add_edge(
            node_id2,
            node_id1,
            weight,
            reduction,
            self._rev_edge_data,
        )

    def remove_edge(self, node_id1: str, node_id2: str):
        """Remove an edge in the graph.

        Args:
            node_id1: ID of node 1.
            node_id2: ID of node 2.

        """
        self._remove_edge(node_id1, node_id2, self.edge_data)
        self._remove_edge(node_id2, node_id1, self.rev_edge_data)

    def connected_components(self):
        """Find connected components."""
        raise NotImplementedError
