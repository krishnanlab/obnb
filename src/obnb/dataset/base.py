"""Dataset object."""
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

from obnb.feature import FeatureVec, MultiFeatureVec
from obnb.feature.base import BaseFeature
from obnb.graph import DenseGraph, SparseGraph
from obnb.graph.base import BaseGraph
from obnb.label.collection import LabelsetCollection
from obnb.label.split.base import BaseSplit
from obnb.typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Literal,
    Optional,
    PyG_Data,
    Tuple,
    Union,
)
from obnb.util.checkers import checkLiteral, checkNumpyArrayShape, checkType
from obnb.util.idhandler import IDmap
from obnb.util.resolver import resolve_transform


class Dataset:
    """Dataset object.

    Args:
        graph: Graph object.
        feature: Feature object.
        label: Label set collection object.
        auto_generate_feature: Automatically generate features from the input
            graph if it is graph is available. If specified as None, then do
            not generate features from the graph automatically.
        dual: If set to True, consider each feature dimension as a sample.
        transform: Transform function or name of the transform class.
        transform_kwargs: Keyword arguments for initializing the transform
            function. Only effective when transform is passed as a string.
        splitter: Splitter object that determins train/val/test split.
        split_kwargs: Keyword arguments for calling the split function of the
            splitter.

    """

    def __init__(
        self,
        *,
        graph: Optional[BaseGraph] = None,
        feature: Optional[BaseFeature] = None,
        label: Optional[LabelsetCollection] = None,
        auto_generate_feature: Optional[str] = "OneHotLogDeg",  # TODO: deprecate
        dual: bool = False,
        transform: Optional[Callable] = None,
        transform_kwargs: Optional[Dict[str, Any]] = None,
        splitter: Optional[BaseSplit] = None,
        **split_kwargs,
    ):
        """Initialize Dataset."""
        self.set_idmap(graph, feature)
        self.graph = graph
        self.feature = feature
        self.extras: Dict[str, np.ndarray] = {}

        # TODO: replace by transform
        if self.feature is None and auto_generate_feature:
            self.generate_features(auto_generate_feature)

        self.label = label
        self.splitter = splitter
        if label is None:
            raise ValueError("Missing required kwarg 'label'")
        elif splitter is None:
            self.y = label.get_y(target_ids=tuple(self.idmap.lst))
            self.masks = None
        else:
            self.y, self.masks = label.split(
                splitter,
                target_ids=tuple(self.idmap.lst),
                **split_kwargs,
            )

        # TODO: replace consider_negative option in label.split with this
        _, self.y_mask = label.get_y(
            target_ids=tuple(self.idmap.lst),
            return_y_mask=True,
        )

        transform = resolve_transform(transform, transform_kwargs)
        if transform is not None:
            transform(self)

    def get_adj(self) -> np.ndarray:
        """Get dense adjacency matrix."""
        assert self.graph is not None
        return self.get_dense_graph().mat

    def get_sparse_graph(self) -> SparseGraph:
        """Get sparse graph object."""
        assert self.graph is not None
        is_sparse = isinstance(self.graph, SparseGraph)
        return self.graph if is_sparse else self.graph.to_sparse_graph()

    def get_dense_graph(self) -> DenseGraph:
        """Get dense graph object."""
        assert self.graph is not None
        is_dense = isinstance(self.graph, DenseGraph)
        return self.graph if is_dense else self.graph.to_dense_graph()

    @property
    def idmap(self) -> IDmap:
        """Map instance IDs to indexes."""
        return self._idmap

    @property
    def size(self) -> int:
        """Instances number in the dataset."""
        return self.idmap.size

    @property
    def fset_idmap(self) -> Optional[IDmap]:
        """Map multifeature IDs to indexes."""
        return self._fset_idmap

    def set_idmap(
        self,
        graph: Optional[BaseGraph],
        feature: Optional[BaseFeature],
    ) -> None:
        """Set mapping of node IDs to node index.

        Use the IDmap of either graph or feature if only one is specified.
        If both are specified, it checks whether the two IDmaps are aligned.

        Raises:
            ValueError: If neither the graph or the feature are specified,
                or both are specified but the IDmaps do not align.

        """
        if graph is not None and feature is not None:
            # TODO: fix IDmap.__eq__ to compare list instead of set
            if not feature.idmap.lst == graph.idmap.lst:
                raise ValueError("Misaligned IDs between graph and feature")
            self._idmap = graph.idmap.copy()
        elif graph is not None:
            self._idmap = graph.idmap.copy()
        elif feature is not None:
            self._idmap = feature.idmap.copy()
        else:
            raise ValueError("Must specify either graph or feature.")

    @property
    def y(self) -> Optional[np.ndarray]:
        return getattr(self, "_y", None)

    @y.setter
    def y(self, y: Optional[np.ndarray]):
        if y is not None and y.shape[0] != self.size:
            raise ValueError(f"Incorrect shape {y.shape=}")
        self._y = y

    @property
    def y_mask(self) -> Optional[np.ndarray]:
        return getattr(self, "_y_mask", None)

    @y_mask.setter
    def y_mask(self, y_mask: Optional[np.ndarray]):
        if y_mask is not None and y_mask.shape[0] != self.size:
            raise ValueError(f"Incorrect shape {y_mask.shape=}")
        self._y_mask = y_mask

    @property
    def dual(self):
        return self._dual

    @dual.setter
    def dual(self, dual):
        if dual:
            if not isinstance(self.features, MultiFeatureVec):
                raise TypeError(
                    "'dual' mode only works when the features is of type Multi"
                    f"FeatureVec, but received type {type(self.features)!r}",
                )
            target_indptr = np.arange(self.features.indptr.size)
            if not np.all(self.features.indptr == target_indptr):
                raise ValueError(
                    "'dual' mode only works when the MultiFeatureVec only "
                    "contains one-dimensional feature sets.",
                )
            self._dual = True
            self._fset_idmap = self.features.fset_idmap.copy()
        else:
            self._dual = False
            self._fset_idmap = None

    @property
    def feature(self):
        return self._feature

    @feature.setter
    def feature(self, feature: Optional[BaseFeature]):
        if feature is not None:
            checkType("feature", BaseFeature, feature)
        self._feature = feature

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph: Optional[BaseGraph]):
        if graph is not None:
            checkType("graph", BaseGraph, graph)
        self._graph = graph

    def get_feat(
        self,
        ind: Union[Iterable[int], Iterable[str], np.ndarray],
        /,
        *,
        mode: Literal["ids", "idxs", "mask", "auto"] = "auto",
    ) -> np.ndarray:
        """Obtain feature given indicators.

        Args:
            ind: Indicator about which instances to retrieve the featuers from.
            mode: Type of indicator, accepted options are
                ['ids', 'idxs', 'mask', 'auto']. `'ids'` means the indicator is
                a sequence of instances IDs; `'idxs'` means the indicator is a
                sequence of instances indexes; `'mask'` means the indicator is a
                numpy array whose entries corresponding to the instances of
                interest are marked; finally, `'auto'` tires to determine the
                type automatically. It is recommended to set the mode
                explicitly rather than `'auto'` when possible.

        """
        if self.feature is None:
            raise ValueError("feature not set")

        # Determine the indexing mode
        checkLiteral("mode", Literal["ids", "idxs", "mask", "auto"], mode)
        if mode != "auto":
            ind_mode = mode
        elif isinstance(ind, str) or (
            isinstance(ind, Iterable) and all(isinstance(i, str) for i in ind)
        ):
            ind_mode = "ids"
        elif isinstance(ind, int) or (
            isinstance(ind, Iterable) and all(isinstance(i, int) for i in ind)
        ):
            ind_mode = "idxs"
        elif (
            isinstance(ind, np.ndarray)
            and (ind.shape[0] == self.idmap.size)
            and (np.unique(ind).size == 2)
        ):
            ind_mode = "mask"
        else:
            raise ValueError("Unable to automatcially determine indexing mode.")

        # Obtain features using the corresponding indexing mode
        if ind_mode == "ids":
            return self._get_feat_from_ids(ind)  # type: ignore
        elif ind_mode == "idxs":
            return self._get_feat_from_idxs(ind)  # type: ignore
        elif ind_mode == "mask":
            return self._get_feat_from_mask(ind)  # type: ignore
        else:
            raise ValueError("This should not happen")

    def _get_feat_from_idxs(self, idxs: Iterable[int]) -> np.ndarray:
        return self.feature.mat[idxs]

    def _get_feat_from_ids(self, ids: Iterable[str]) -> np.ndarray:
        idxs = self.idmap[ids]
        return self._get_feat_from_idxs(idxs)

    def _get_feat_from_mask(self, mask: np.ndarray) -> np.ndarray:
        checkNumpyArrayShape("mask", self.size, mask)
        idxs = np.where(mask)[0]
        return self._get_feat_from_idxs(idxs)

    def get_mask(self, name: str, split_idx: int) -> np.ndarray:
        """Return the mask given name and split index."""
        if self.masks is None:
            raise ValueError("Masks not set.")
        return self.masks[name][:, split_idx]

    def get_split(self, name: str, split_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return feature and label pair given the split name and index."""
        if self.feature is None or self.y is None:
            raise ValueError("Both feature and y must be set.")
        mask = self.get_mask(name, split_idx)
        x = self.get_feat(mask, mode="mask")
        y = self.y[mask]
        return x, y

    def splits(
        self,
        split_idx: int,
    ) -> Iterator[Tuple[str, Tuple[np.ndarray, np.ndarray]]]:
        """Iterate over all masks and return the mask name along with split."""
        if self.masks is None:
            raise ValueError("Masks not set.")
        for mask_name in self.masks:
            yield mask_name, self.get_split(mask_name, split_idx)

    def generate_features(self, name: str = "OneHotLogDeg", overwrite: bool = False):
        if self.graph is None:
            raise ValueError("Missing graph in the dataset object")

        if self.feature is not None and not overwrite:
            raise ValueError(
                "Feature already exists. Set overwrite to True to force "
                "overwrite the original feature in the dataset object.",
            )

        if name == "OneHotLogDeg":
            deg = self.graph.degree(weighted=False)[:, None]
            feat = KBinsDiscretizer(
                n_bins=32,
                encode="onehot-dense",
                strategy="uniform",
            ).fit_transform(np.log(deg))
            self._feature = FeatureVec.from_mat(feat, list(self.graph.node_ids))
        else:
            raise NotImplementedError(f"{name} feature is not implemented yet.")

    def update_extras(self, key, val):
        if key in self.extras:
            raise KeyError(f"{key} extras already exist")
        self.extras[key] = val

    def to_pyg_data(
        self,
        *,
        device: str = "cpu",
        mask_suffix: str = "_mask",
    ) -> PyG_Data:
        """Convert dataset into PyG data."""
        # TODO: dense option
        import torch
        from torch_geometric.data import Data

        device = torch.device(device)
        num_nodes = self.size

        x = self.feature.mat
        edge_index, edge_weight = self.graph.to_coo()  # TODO: empty graph?

        x = torch.FloatTensor(self.feature.mat)
        edge_index = torch.LongTensor(edge_index)
        edge_weight = None if edge_weight is None else torch.FloatTensor(edge_weight)

        data = Data(
            num_nodes=num_nodes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            x=x,
        )

        if self.graph is not None:
            data.node_ids = list(self.graph.node_ids)

        if self.label is not None:
            data.task_ids = list(self.label.label_ids)

        # Label (true) matrix
        if self.y is not None:
            data.y = torch.FloatTensor(self.y)

        # Label mask (negative selection) matrix
        if self.y_mask is not None:
            data.y_mask = torch.BoolTensor(self.y_mask)

        # Split mask matrix
        if self.masks is not None:
            data.masks = []
            for mask_name, mask in self.masks.items():
                data.masks.append(attrname := mask_name + mask_suffix)
                setattr(data, attrname, torch.BoolTensor(mask))

        # Extra data
        for key, val in self.extras.items():
            setattr(data, key, torch.tensor(val))

        data.to(device)

        return data

    def to_dgl_data(
        self,
        *,
        device: str = "cpu",
        mask_suffix: str = "_mask",
    ):
        """Convert dataset into a DGL graph."""
        # TODO: dense option
        import dgl
        import torch

        device = torch.device(device)
        num_nodes = self.size

        # Use trivial feature if feature not available
        x = self.feature.mat
        (edges_src, edges_dst), edge_weight = self.graph.to_coo()  # TODO: empty graph?

        dglgraph = dgl.graph(
            (torch.LongTensor(edges_src), torch.LongTensor(edges_dst)),
            num_nodes=num_nodes,
        )
        dglgraph.ndata["feat"] = torch.FloatTensor(x)

        if edge_weight is not None:
            dglgraph.edata["weight"] = torch.FloatTensor(edge_weight)

        if self.graph is not None:
            dglgraph.node_ids = list(self.graph.node_ids)

        if self.label is not None:
            dglgraph.task_ids = list(self.label.label_ids)

        # Label (true) matrix
        if self.y is not None:
            dglgraph.ndata["label"] = torch.FloatTensor(self.y)

        # Label mask (negative selection) matrix
        if self.y_mask is not None:
            dglgraph.ndata["label_mask"] = torch.BoolTensor(self.y_mask)

        # Split mask matrix
        if self.masks is not None:
            for mask_name, mask in self.masks.items():
                dglgraph.ndata[mask_name + mask_suffix] = torch.BoolTensor(mask)

        # Extra data
        for key, val in self.extras.items():
            dglgraph.ndata[key] = torch.tensor(val)

        dglgraph = dglgraph.to(device)

        return dglgraph
