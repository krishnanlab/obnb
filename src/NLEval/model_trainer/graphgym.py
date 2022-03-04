import logging
from typing import Any
from typing import List
from typing import Optional

import torch
from torch_geometric import graphgym as pyg_gg
from torch_geometric import seed_everything
from torch_geometric.data import DataLoader
from torch_geometric.graphgym import cfg as cfg_gg
from torch_geometric.graphgym import logger as logger_gg
from torch_geometric.graphgym.config import assert_cfg
from torch_geometric.graphgym.config import dump_cfg
from torch_geometric.graphgym.loader import get_loader
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.register import register_loss
from torch_geometric.graphgym.register import register_metric
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device

from .gnn import GNNTrainer


class GraphGymTrainer(GNNTrainer):
    """Trainer built upon GraphGym.

    Specify configurations either as file, or as kwargs. Then GrphGymTrainer
    will use those configurations to set up GraphGym. One can then create the
    model as specified in the configurations using the ``create_model`` method.

    """

    def __init__(
        self,
        metrics,
        graph,
        features=None,
        device: str = "auto",
        metric_best: str = "auto",
        cfg_file: Optional[str] = None,
        cfg_opts: List[Any] = None,
    ):
        """Initialize GraphGymTrainer.

        Args:
            cfg_file (str): Configuration file to use for setting up GraphGym.
            kwargs: Remaining configuration options for graphgym.

        """
        if cfg_file is None:
            cfg_file = "configs/graphgym/default_config.yaml"
            logging.warn(
                "No configuration file specified, using the default "
                f"configuration from {cfg_file}",
            )
        cfg_gg.merge_from_file(cfg_file)

        args = ["device", device, "metric_best", metric_best]
        if cfg_opts is not None:
            args += cfg_opts
        cfg_gg.merge_from_list(args)

        assert_cfg(cfg_gg)
        # Only support multilabel classification
        cfg_gg.dataset.task_type = "classification_multilabel"
        dump_cfg(cfg_gg)

        pyg_gg.set_run_dir(cfg_gg.out_dir, cfg_file)
        pyg_gg.set_printing()  # TODO: remove log file? Use only for training..

        seed_everything(cfg_gg.seed)
        auto_select_device()
        selected_device = cfg_gg.device

        logging.info(cfg_gg)

        self.register_metrics(metrics, metric_best)

        super().__init__(metrics, graph, features, device=selected_device)

    @staticmethod
    def register_metrics(metrics, metric_best):
        """Register custom metrics to be used by GraphGym."""
        for metric_name, metric in metrics.items():
            register_metric(metric_name, metric)
            logging.info(f"Registered metric {metric_name!r}")
        cfg_gg.custom_metrics = list(metrics)
        cfg_gg.metric_best = metric_best

    def create_model(self, dim_in: int, dim_out: int, to_device: bool = True):
        """Create model based on the GraphGym configuration."""
        mdl = create_model(dim_in=dim_in, dim_out=dim_out, to_device=to_device)
        logging.info(mdl)
        cfg_gg.params = params_count(mdl)
        return mdl

    def get_loggers(self, masks) -> List[logger_gg.Logger]:
        """Obtain GraphGym loggers."""
        return [logger_gg.Logger(name=name) for name in masks]

    def get_loaders(self, y, masks, split_idx) -> List[DataLoader]:
        """Obtain GraphGym data loader."""
        # Create a copy of the data used for evaluation
        data = self.data.clone().detach().cpu()
        data.y = torch.Tensor(y).float()

        for mask_name in masks:
            mask = self.get_mask(masks, mask_name, split_idx)
            torch_mask = torch.Tensor(mask).bool()
            setattr(data, f"{mask_name}_mask", torch_mask)
        logging.info(data)

        loaders = [get_loader([data], "full_batch", 1, True)]
        for _ in range(len(masks) - 1):
            loaders.append(get_loader([data], "full_batch", 1, False))

        return loaders

    def train(self, model, y, masks, split_idx=0):
        """Train model using GraphGym."""
        loggers = self.get_loggers(masks)
        loaders = self.get_loaders(y, masks, split_idx)

        optimizer = pyg_gg.create_optimizer(model.parameters(), cfg_gg.optim)
        scheduler = pyg_gg.create_scheduler(optimizer, cfg_gg.optim)

        # TODO: find out a way to rewind the model back to optimal state.
        pyg_gg.train(loggers, loaders, model, optimizer, scheduler)


@register_loss("multilabel_cross_entropy")
def multilabel_cross_entropy(pred, true):
    """Binary Cross Entropy as loss for multilabel classification tasks."""
    if cfg_gg.dataset.task_type == "classification_multilabel":
        bce_loss = torch.nn.BCEWithLogitsLoss()
        return bce_loss(pred, true), torch.sigmoid(pred)
