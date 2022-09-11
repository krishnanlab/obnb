"""Dataset object."""
import numpy as np

from nleval.feature import MultiFeatureVec
from nleval.feature.base import BaseFeature
from nleval.graph.base import BaseGraph
from nleval.label.collection import LabelsetCollection
from nleval.label.split.base import BaseSplit
from nleval.typing import Iterable, Iterator, Literal, Optional, PyG_Data, Tuple, Union
from nleval.util.checkers import checkLiteral, checkNumpyArrayShape, checkType
from nleval.util.idhandler import IDmap


class Dataset:
    """Dataset object."""

    def __init__(
        self,
        *,
        graph: Optional[BaseGraph] = None,
        feature: Optional[BaseFeature] = None,
        label: Optional[LabelsetCollection] = None,
        dual: bool = False,
        splitter: Optional[BaseSplit] = None,
        **split_kwargs,
    ):
        """Initialize Dataset."""
        self.set_idmap(graph, feature)
        self.graph = graph
        self.feature = feature

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
        return self._y

    @y.setter
    def y(self, y: Optional[np.ndarray]):
        if y is not None and y.shape[0] != self.size:
            raise ValueError(f"Incorrect shape {y.shape=}")
        self._y = y

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

        # Use trivial feature if feature not available
        x = np.ones((num_nodes, 1)) if self.feature is None else self.feature.mat
        edge_index, edge_weight = self.graph.to_coo()  # TODO: empty graph?

        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_index)
        edge_weight = None if edge_weight is None else torch.FloatTensor(edge_weight)

        data = Data(
            num_nodes=num_nodes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            x=x,
        )

        if self.y is not None:
            data.y = torch.FloatTensor(self.y)

        if self.masks is not None:
            data.masks = []
            for mask_name, mask in self.masks.items():
                data.masks.append(attrname := mask_name + mask_suffix)
                setattr(data, attrname, torch.BoolTensor(mask))

        data.to(device)

        return data
