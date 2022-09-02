import numpy as np

from NLEval.feature import MultiFeatureVec
from NLEval.feature.base import BaseFeature
from NLEval.graph.base import BaseGraph
from NLEval.typing import Iterable, Literal, Optional, PyG_Data, Union
from NLEval.util.checkers import checkLiteral, checkNumpyArrayShape, checkType
from NLEval.util.idhandler import IDmap


class Dataset:
    def __init__(
        self,
        *,
        graph: Optional[BaseGraph] = None,
        feature: Optional[BaseFeature] = None,
        dual: bool = False,
    ):
        self.set_idmap(graph, feature)
        self.graph = graph
        self.feature = feature

    @property
    def idmap(self) -> IDmap:
        """Map instance IDs to indexes."""
        return self._idmap

    @property
    def size(self) -> int:
        """Number of instances in the dataset."""
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

    @property
    def label(self):
        # XXX:
        return None

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
                type automatically. It is recommented to set the mode
                explicitely rather than `'auto'` when possible.

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
            return self._get_feat_from_ids(ind)
        elif ind_mode == "idxs":
            return self._get_feat_from_idxs(ind)
        elif ind_mode == "mask":
            return self._get_feat_from_mask(ind)
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

    def get_mask(self):
        # TODO: get from lsc split?
        raise NotImplementedError

    def to_pyg_data(self, device: str = "cpu") -> PyG_Data:
        # TODO: dense option
        import torch
        from torch_geometric.data import Data

        device = torch.device(device)

        num_nodes = self.size
        # Use trivial feature if feature not available
        x = np.ones((num_nodes, 1)) if self.feature is None else self.feature.mat
        edge_index, edge_weight = self.graph.to_coo()  # TODO: empty graph?

        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_index)
        edge_weight = torch.FloatTensor(edge_weight)

        data = Data(
            num_nodes=num_nodes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            x=x,
            device=device,
        )
        data.to(device)
        return data

    # XXX: combine the following with Dataset.to_pyg_data
    # def export_pyg_data(
    #     self,
    #     y: np.ndarray,
    #     masks: Dict[str, np.ndarray],
    #     mask_suffix: str = "_mask",
    # ) -> Data:
    #     """Export PyTorch Geometric Data object.

    #     Args:
    #         y: Label array.
    #         masks: Dictionary of masks.
    #         mask_suffix (str): Mask name suffix.

    #     """
    #     data = self.data.clone().detach().cpu()
    #     data.y = torch.Tensor(y).float()
    #     for mask_name, mask in masks.items():
    #         setattr(data, mask_name + mask_suffix, torch.from_numpy(mask))
    #     return data

    # XXX: implement
    # def get_x_from_mask(self, mask):
    #     """Obtain features of specific nodes from a specific feature set.

    #     In each iteraction, use one single feature set, indicated by
    #     ``self._curr_fset_name``, which updated within the for loop in the
    #     ``train`` method below.

    #     """
    #     checkNumpyArrayShape("mask", len(self.idmap), mask)
    #     idx = np.where(mask)[0]
    #     fset_idx = self.features.fset_idmap[self._curr_fset_name]
    #     return self.features.get_features_from_idx(idx, fset_idx)
