from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Constant

from .base import BaseTrainer


class GNNTrainer(BaseTrainer):
    """Trainner for GNN models."""

    def __init__(
        self,
        metrics,
        graph,
        features=None,
        device: str = "cpu",
        metric_best: Optional[str] = None,
    ):
        """Initialize GNNTrainer.

        Args:
            device (str): Training device.
            metric_best (str): Metric used for determining the best model.

        """
        super().__init__(metrics, graph=graph, features=features)

        self.metric_best = metric_best

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
        if metric_best is None:
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
    ) -> Tuple[Dict[str, List], Dict[str, float], Dict[str, Any]]:
        """Create new stats for tracking model performance."""
        stats: Dict[str, List] = {"epoch": [], "loss": []}
        best_stats: Dict[str, float] = {"epoch": 0, "loss": 1e12}
        best_model_state: Dict[str, Any] = {}

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
        val_on: str,
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
            val_on: Validation mask name.

        """
        new_results["epoch"] = epoch
        new_results["loss"] = loss
        name = f"{val_on}_{self.metric_best}"
        if new_results[name] > best_stats[name]:
            best_stats.update(new_results)
            best_model_state.update(deepcopy(model.state_dict()))
        for i, j in new_results.items():
            stats[i].append(j)


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

    def train(
        self,
        model,
        y,
        masks,
        split_idx=0,
        train_on="train",
        val_on: str = "val",
        lr: float = 0.01,
        epochs: int = 100,
        eval_steps: int = 10,
        log: bool = False,
    ):
        """Train the GNN model.

        Args:
            val_on (str): Validation mask name (default: :obj:`"train"`)
            lr (float): Learning rate (default: :obj:`0.01`)
            epochs (int): Total epochs (default: :obj:`100`)
            eval_steps (int): Interval for evaluation (default: :obj:`10`)
            log (bool): Print evaluation results at each evaluation epoch
                if set to True (default: :obj:`False`)

        """
        model.to(self.device)
        data = self.data
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_mask = self.get_mask(masks, train_on, split_idx)
        y_torch = torch.from_numpy(y.astype(float)).to(self.device)

        stats, best_stats, best_model_state = self.new_stats(masks)

        for epoch in range(epochs):
            loss = self.train_epoch(model, data, y_torch, train_mask, optimizer)

            if epoch % eval_steps == 0:
                new_results = self.evaluate(model, y, masks, split_idx)
                self.update_stats(
                    model,
                    stats,
                    best_stats,
                    best_model_state,
                    new_results,
                    epoch,
                    loss,
                    val_on,
                )

                if log:
                    print(new_results)

        model.load_state_dict(best_model_state)

        return best_stats
