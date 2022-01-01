from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
from NLEval.graph.base import BaseGraph
from NLEval.util.checkers import checkNumpyArrayShape


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
    ):
        """Initialize BaseTraining.

        Args:
            metrics: Dictionary of metrics used to train/evaluate the model.
            graph: Optional graph object.
            features: Optional node feature vectors.

        """
        self.metrics = metrics
        self.set_idmap(graph, features)
        self.graph = graph
        self.features = features

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

    def get_x(self, idx: Sequence[int]) -> np.ndarray:
        """Return features given list of node index."""
        # TODO: make this more generic, e.g. what if we want to use SparseGraph
        if self.features is not None:
            return self.features.mat[idx]
        else:
            raise ValueError("Features not set")

    def get_x_from_ids(self, ids: Sequence[str]):
        """Return features given list of node IDs."""
        idx = self.idmap[ids]
        return self.get_x(idx)

    def get_x_from_mask(self, mask: np.ndarray):
        """Return features given an 1-dimensional node mask."""
        checkNumpyArrayShape("mask", len(self.idmap), mask)
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
        train_on: str = "train",
    ):
        """Train model and return metrics.

        Args:
            model: Model to be trained.
            y: Label array with the shape of (n_tot_samples, n_classes) or
                (n_tot_samples,) if n_classes = 1.
            masks: Masks for splitting data, see the ``split`` method in
                ``label.labelset_collection.LSC`` for moer info.
            split_idx: Which split to use for training and evaluation.
            train_on: Which mask to use for training.

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not have functional ``train`` "
            f"method, use a derived class instead.",
        )
