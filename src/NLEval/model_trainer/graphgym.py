import logging
from itertools import chain
from typing import Any
from typing import Dict
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
from torch_geometric.graphgym.train import eval_epoch
from torch_geometric.graphgym.train import train_epoch
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.utils.epoch import is_eval_epoch
from torch_geometric.graphgym.utils.epoch import is_train_eval_epoch

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
        cfg_opts: Dict[str, Any] = None,
    ):
        """Initialize GraphGymTrainer.

        Args:
            cfg_file (str): Configuration file to use for setting up GraphGym.
            cfg_opts: Remaining configuration options for graphgym.

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
            args += list(chain.from_iterable(cfg_opts.items()))
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

    @torch.no_grad()
    def evaluate(self, loaders, model, masks):
        model.eval()

        results = {}
        for split, loader in zip(masks, loaders):
            batch = list(loader)[0]
            batch.split = split  # TODO: make 'all'
            batch.to(torch.device(cfg_gg.device))
            pred, true = model(batch)

            for metric_name, metric_func in self.metrics.items():
                results[f"{split}_{metric_name}"] = metric_func(
                    true.detach().cpu().numpy(),
                    pred.detach().cpu().numpy(),
                )

        return results

    def train(self, model, y, masks, split_idx=0):
        """Train model using GraphGym."""
        loggers = self.get_loggers(masks)  # TODO: remove 'val' and 'test' lgrs
        loaders = self.get_loaders(y, masks, split_idx)

        optimizer = pyg_gg.create_optimizer(model.parameters(), cfg_gg.optim)
        scheduler = pyg_gg.create_scheduler(optimizer, cfg_gg.optim)

        # TODO: find out a way to rewind the model back to optimal state.
        split_names = [i for i in masks if i != "train"]
        stats, best_stats, best_model_state = self.new_stats(masks)
        for cur_epoch in range(cfg_gg.optim.max_epoch):
            train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)

            if is_eval_epoch(cur_epoch):
                new_results = self.evaluate(loaders, model, masks)
                self.update_stats(
                    model,
                    stats,
                    best_stats,
                    best_model_state,
                    new_results,
                    cur_epoch,
                    loggers[0].basic()["loss"],
                )
                logging.info(new_results)

        for logger in loggers:
            logger.close()

        logging.info("Task done, results saved in {}".format(cfg_gg.run_dir))


@register_loss("multilabel_cross_entropy")
def multilabel_cross_entropy(pred, true):
    """Binary Cross Entropy as loss for multilabel classification tasks."""
    if cfg_gg.dataset.task_type == "classification_multilabel":
        bce_loss = torch.nn.BCEWithLogitsLoss()
        return bce_loss(pred, true), torch.sigmoid(pred)
