from copy import deepcopy

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Constant

from ..typing import Any
from ..typing import Dict
from ..typing import List
from ..typing import LogLevel
from ..typing import Optional
from ..typing import Tuple
from .base import BaseTrainer


class GNNTrainer(BaseTrainer):
    """Trainner for GNN models."""

    def __init__(
        self,
        metrics,
        graph,
        features=None,
        train_on="train",
        val_on: str = "val",
        device: str = "cpu",
        metric_best: Optional[str] = None,
        lr: float = 0.01,
        epochs: int = 100,
        eval_steps: int = 10,
        log_level: LogLevel = "INFO",
    ):
        """Initialize GNNTrainer.

        Args:
            val_on (str): Validation mask name (default: :obj:`"train"`).
            device (str): Training device (default: :obj:`"cpu"`).
            metric_best (str): Metric used for determining the best model
                (default: :obj:`None`).
                if set to True (default: :obj:`False`)
            lr (float): Learning rate (default: :obj:`0.01`)
            epochs (int): Total epochs (default: :obj:`100`)
            eval_steps (int): Interval for evaluation (default: :obj:`10`)

        """
        super().__init__(
            metrics,
            graph=graph,
            features=features,
            train_on=train_on,
            log_level=log_level,
        )

        self.val_on = val_on
        self.metric_best = metric_best
        self.lr = lr
        self.epochs = epochs
        self.eval_steps = eval_steps

        edge_index, edge_weight = graph.to_pyg_edges()
        self.data = Data(
            num_nodes=graph.size,
            edge_index=torch.from_numpy(edge_index),
            edge_weight=torch.from_numpy(edge_weight),
        )

        # Use trivial feature if not available
        if features is not None:
            self.data.x = torch.from_numpy(features.to_pyg_x())
        else:
            Constant(cat=False)(self.data)

        self.data.to(device)
        self.device = device

    @property
    def metric_best(self):
        """str: Metric used for determining the best model."""
        return self._metric_best

    @metric_best.setter
    def metric_best(self, metric_best):
        """Setter for :attr:`metric_best`.

        Raises:
            ValueError: More than one metrics is available but did not specify
                metric_best.
            KeyError: metric_best did not match any of the spcified metrics.

        """
        if metric_best is None or metric_best == "auto":
            if len(self.metrics) != 1:
                raise ValueError(
                    "Multiple metrics found but did not specify metric_best",
                )
            else:
                self._metric_best = list(self.metrics)[0]
        elif metric_best not in self.metrics:
            raise KeyError(f"No metrics named {metric_best!r}")
        else:
            self._metric_best = metric_best

    def new_stats(
        self,
        masks: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, List], Dict[str, float], Dict[str, torch.Tensor]]:
        """Create new stats for tracking model performance."""
        stats: Dict[str, List] = {"epoch": [], "loss": []}
        best_stats: Dict[str, float] = {"epoch": 0, "loss": 1e12}
        best_model_state: Dict[str, torch.Tensor] = {}

        for mask_name in masks:
            for metric_name in self.metrics:
                name = f"{mask_name}_{metric_name}"
                stats[name] = []
                best_stats[name] = 0.0

        return stats, best_stats, best_model_state

    def update_stats(
        self,
        model: Any,
        stats: Dict[str, List],
        best_stats: Dict[str, float],
        best_model_state: Dict[str, torch.Tensor],
        new_results: Dict[str, float],
        epoch: int,
        loss: float,
    ) -> None:
        """Update model performance stats using the new evaluation results.

        Args:
            model: GNN model.
            stats: Full performance history to be updated.
            best_stats: Current performance.
            best_model_state: State information of the current best model.
            new_results: New evaluation results.
            epoch: Current epoch.
            loss: Current loss.

        """
        new_results["epoch"] = epoch
        new_results["loss"] = loss
        name = f"{self.val_on}_{self.metric_best}"
        if new_results[name] > best_stats[name]:
            best_stats.update(new_results)
            best_model_state.update(deepcopy(model.state_dict()))
        for i, j in new_results.items():
            stats[i].append(j)

    def export_pyg_data(
        self,
        y: np.ndarray,
        masks: Dict[str, np.ndarray],
        mask_suffix: str = "_mask",
    ) -> Data:
        """Export PyTorch Geometric Data object.

        Args:
            y: Label array.
            masks: Dictionary of masks.
            mask_suffix (str): Mask name suffix.

        """
        data = self.data.clone().detach().cpu()
        data.y = torch.Tensor(y).float()
        for mask_name, mask in masks.items():
            setattr(data, mask_name + mask_suffix, torch.from_numpy(mask))
        return data

    def is_eval_epoch(self, cur_epoch: int) -> bool:
        """Return true if current epoch is eval epoch."""
        return cur_epoch % self.eval_steps == 0


class SimpleGNNTrainer(GNNTrainer):
    """Simple GNN trainer using Adam with fixed learning rate."""

    @staticmethod
    def train_epoch(model, data, y, train_mask, optimizer):
        """Train a single epoch."""
        model.train()
        criterion = torch.nn.BCEWithLogitsLoss()

        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_weight)[train_mask]
        loss = criterion(out, y[train_mask])
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, model, y, masks, split_idx):
        """Evaluate current model."""
        model.eval()
        args = (self.data.x, self.data.edge_index, self.data.edge_weight)
        y_pred = model(*args).detach().cpu().numpy()

        results = {}
        for metric_name, metric_func in self.metrics.items():
            for mask_name in masks:
                mask = self.get_mask(masks, mask_name, split_idx)
                score = metric_func(y[mask], y_pred[mask])
                results[f"{mask_name}_{metric_name}"] = score

        return results

    def train(self, model, y, masks, split_idx=0):
        """Train the GNN model."""
        model.to(self.device)
        data = self.data
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        train_mask = self.get_mask(masks, self.train_on, split_idx)
        y_torch = torch.from_numpy(y.astype(float)).to(self.device)

        stats, best_stats, best_model_state = self.new_stats(masks)
        for cur_epoch in range(self.epochs):
            loss = self.train_epoch(model, data, y_torch, train_mask, optimizer)

            if self.is_eval_epoch(cur_epoch):
                new_results = self.evaluate(model, y, masks, split_idx)
                self.update_stats(
                    model,
                    stats,
                    best_stats,
                    best_model_state,
                    new_results,
                    cur_epoch,
                    loss,
                )
                self.logger.info(new_results)

        # Rewind back to best model
        model.load_state_dict(best_model_state)

        return best_stats
