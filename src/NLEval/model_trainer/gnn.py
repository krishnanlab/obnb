from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Constant

from .base import BaseTrainer


class GNNTrainer(BaseTrainer):
    def __init__(
        self,
        metrics,
        graph,
        features=None,
        device="cpu",
        metric_best=None,  # if none, use the only one, err if more than one
    ):
        super().__init__(metrics, graph=graph, features=features)

        edge_index, edge_weight = graph.to_pyg_edges()
        self.data = Data(
            num_nodes=graph.size,
            edge_index=torch.from_numpy(edge_index),
            edge_weight=torch.from_numpy(edge_weight),
        )

        if features is not None:
            self.data.x = torch.from_numpy(features.to_pyg_x())
        else:
            Constant(cat=False)(self.data)

        self.metric_best = metric_best  # TODO: error if not in metircs
        self.device = device
        self.data.to(device)

    def new_stats(
        self,
        masks: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, List], Dict[str, float]]:
        stat_name_list = ["epoch", "loss"] + list(self.metrics)

        stats = {"epoch": [], "loss": []}
        best_stats = {"epoch": 0, "loss": 1e12}

        for mask_name in masks:
            for metric_name in self.metrics:
                name = f"{mask_name}_{metric_name}"
                stats[name] = []
                best_stats[name] = 0.0

        return stats, best_stats

    def update_stats(
        self,
        stats: Dict[str, List],
        best_stats: Dict[str, float],
        new_results: Dict[str, float],
        epoch: int,
        loss: float,
        val_on: str,
    ) -> None:
        new_results["epoch"] = epoch
        new_results["loss"] = loss
        name = f"{val_on}_{self.metric_best}"
        if new_results[name] > best_stats[name]:
            best_stats.update(new_results)
        for i, j in new_results.items():
            stats[i].append(j)


class SimpleGNNTrainer(GNNTrainer):
    @staticmethod
    def train_epoch(model, data, y, train_mask, optimizer):
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
        val_on="val",
        lr=0.01,
        epochs=100,
        eval_steps=10,
        log=False,
    ):
        model.to(self.device)
        data = self.data
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_mask = self.get_mask(masks, train_on, split_idx)
        train_mask_torch = torch.from_numpy(train_mask).to(self.device)
        y_torch = torch.from_numpy(y.astype(float)).to(self.device)

        stats, best_stats = self.new_stats(masks)

        for epoch in range(epochs):
            loss = self.train_epoch(model, data, y_torch, train_mask, optimizer)

            if epoch % eval_steps == 0:
                new_results = self.evaluate(model, y, masks, split_idx)
                self.update_stats(
                    stats,
                    best_stats,
                    new_results,
                    epoch,
                    loss,
                    val_on,
                )

                if log:
                    print(f"{epoch} {best_stats}")

        # TODO: rewind to the best model based on metric_best

        return best_stats
