import numpy as np

from NLEval.feature import MultiFeatureVec
from NLEval.graph.base import BaseGraph
from NLEval.typing import Any, Callable, Dict, LogLevel, Optional, Sequence
from NLEval.util.checkers import checkNumpyArrayShape
from NLEval.util.logger import get_logger


class BaseTrainer:
    """The BaseTrainer object.

    Abstract class for trainer objects, which serve as interfaces or shortcuts
    for training specific types of models.

    """

    def __init__(
        self,
        metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
        graph: Optional[BaseGraph] = None,
        features: Optional[BaseGraph] = None,
        train_on: str = "train",
        dual: bool = False,
        log_level: LogLevel = "INFO",
    ):
        """Initialize BaseTraining.

        Note: "dual" mode only works if the input features is MultiFeatureVec.

        Args:
            metrics: Dictionary of metrics used to train/evaluate the model.
            graph: Optional graph object.
            features: Optional node feature vectors.
            train_on: Which mask to use for training.
            dual (bool): If set to true, predict the label of individual
                feature, i.e.  individual columns (default: :obj:`False`)

        """
        self.metrics = metrics
        self.set_idmap(graph, features)
        self.graph = graph
        self.features = features
        self.train_on = train_on
        self.dual = dual
        self.logger = get_logger(
            self.__class__.__name__,
            log_level=log_level,
            base_logger="NLEval_brief",
        )

    @property
    def idmap(self):
        """Map node IDs to node index."""
        return self._idmap

    def set_idmap(
        self,
        graph: Optional[BaseGraph],
        features: Optional[BaseGraph],
    ) -> None:
        """Set mapping of node IDs to node index.

        Use the IDmap of either graph or features if only one is specified.
        If both are specified, it checks whether the two IDmaps are aligned.

        Raises:
            ValueError: If neither the graph or the features are specified,
                or both are specified but the IDmaps do not align.

        """
        if graph is not None and features is not None:
            # TODO: fix IDmap.__eq__ to compare list instead of set
            if not features.idmap.lst == graph.idmap.lst:
                raise ValueError("Misaligned IDs between graph and features")
            self._idmap = graph.idmap.copy()
        elif graph is not None:
            self._idmap = graph.idmap.copy()
        elif features is not None:
            self._idmap = features.idmap.copy()
        else:
            raise ValueError("Must specify either graph or features.")

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
            self.fset_idmap = self.features.fset_idmap.copy()
        else:
            self._dual = False
            self.fset_idmap = None

    def get_x(self, idx: Sequence[int]) -> np.ndarray:
        """Return features given list of node or feature index."""
        # TODO: make this more generic, e.g. what if we want to use SparseGraph
        if self.features is not None:
            mat = self.features.mat
            return mat[:, idx].T if self.dual else mat[idx]
        else:
            raise ValueError("Features not set")

    def get_x_from_ids(self, ids: Sequence[str]):
        """Return features given list of node or feature IDs."""
        idx = self.fset_idmap[ids] if self.dual else self.idmap[ids]
        return self.get_x(idx)

    def get_x_from_mask(self, mask: np.ndarray):
        """Return features given an 1-dimensional node mask."""
        shape = len(self.fset_idmap) if self.dual else len(self.idmap)
        checkNumpyArrayShape("mask", shape, mask)
        idx = np.where(mask)[0]
        return self.get_x(idx)

    @staticmethod
    def get_mask(
        masks: Dict[str, np.ndarray],
        mask_name: str,
        split_idx: int,
    ) -> np.ndarray:
        """Return a specific mask."""
        return masks[mask_name][:, split_idx]

    def train(
        self,
        model: Any,
        y: np.ndarray,
        masks: Dict[str, np.ndarray],
        split_idx: int = 0,
    ):
        """Train model and return metrics.

        Args:
            model: Model to be trained.
            y: Label array with the shape of (n_tot_samples, n_classes) or
                (n_tot_samples,) if n_classes = 1.
            masks: Masks for splitting data, see the ``split`` method in
                ``label.collection.LabelsetCollection`` for moer info.
            split_idx: Which split to use for training and evaluation.

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not have functional ``train`` "
            f"method, use a derived class instead.",
        )
