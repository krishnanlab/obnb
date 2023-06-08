import logging
import os
import shutil
from functools import wraps
from itertools import chain

import torch
from torch_geometric import graphgym as pyg_gg
from torch_geometric import seed_everything
from torch_geometric.data import Batch, DataLoader
from torch_geometric.graphgym import cfg as cfg_gg
from torch_geometric.graphgym.config import assert_cfg, dump_cfg
from torch_geometric.graphgym.loader import get_loader
from torch_geometric.graphgym.logger import Logger as Logger_gg
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.register import register_metric
from torch_geometric.graphgym.train import train_epoch
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device

from obnb.model_trainer.gnn import GNNTrainer
from obnb.typing import Any, Dict, List, Optional


def _patch_gg_makedirs_rm_exist(dir_):  # patch for PyG<2.1.0
    if os.path.isdir(dir_):
        shutil.rmtree(dir_)
    os.makedirs(dir_, exist_ok=True)


def _patch_gg_set_out_dir(out_dir, fname):  # patch for PyG<2.1.0
    fname = fname.split("/")[-1]
    if fname.endswith(".yaml"):
        fname = fname[:-5]
    elif fname.endswith(".yml"):
        fname = fname[:-4]

    cfg_gg.out_dir = os.path.join(out_dir, fname)
    # Make output directory
    if cfg_gg.train.auto_resume:
        os.makedirs(cfg_gg.out_dir, exist_ok=True)
    else:
        _patch_gg_makedirs_rm_exist(cfg_gg.out_dir)


def _patch_gg_set_run_dir(out_dir):  # patch for PyG<2.1.0
    cfg_gg.run_dir = os.path.join(out_dir, str(cfg_gg.seed))
    # Make output directory
    if cfg_gg.train.auto_resume:
        os.makedirs(cfg_gg.run_dir, exist_ok=True)
    else:
        _patch_gg_makedirs_rm_exist(cfg_gg.run_dir)


class GraphGymTrainer(GNNTrainer):
    """Trainer built upon GraphGym.

    Specify configurations either as file, or as kwargs. Then GrphGymTrainer
    will use those configurations to set up GraphGym. One can then create the
    model as specified in the configurations using the ``create_model`` method.

    """

    def __init__(
        self,
        metrics,
        device: str = "auto",
        metric_best: str = "auto",
        cfg_file: Optional[str] = None,
        cfg_opts: Optional[Dict[str, Any]] = None,
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
        cfg_gg.dataset.task_type = "classification"

        try:
            pyg_gg.set_out_dir(cfg_gg.out_dir, cfg_file)
            pyg_gg.set_run_dir(cfg_gg.out_dir)
        except AttributeError:
            _patch_gg_set_out_dir(cfg_gg.out_dir, cfg_file)
            _patch_gg_set_run_dir(cfg_gg.out_dir)
        dump_cfg(cfg_gg)
        pyg_gg.set_printing()  # TODO: remove log file? Use only for training..

        seed_everything(cfg_gg.seed)
        auto_select_device()
        selected_device = cfg_gg.device

        logging.info(cfg_gg)

        self.register_metrics(metrics, metric_best)

        self.epochs = cfg_gg.optim.max_epoch
        self.eval_steps = cfg_gg.train.eval_period

        super().__init__(metrics, device=selected_device)

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

    def get_loaders(self, dataset, split_idx: int) -> List[DataLoader]:
        """Obtain GraphGym data loaders.

        Two loaders are set, one is for training and the other is for 'all'. The
        reason for using the 'all' loader is that in the transductive node
        classification setting, the predictions between training stage and
        inference stage are exactly the same. So there is no need to recompute
        the predictions just for the sake the obtaining different masked values.
        Instead, we can directly mask on the full predction values.

        """
        # Create a copy of the data used for evaluation
        # Store at CPU by default, device management done by graphgym
        data = dataset.to_pyg_data(device="cpu", mask_suffix=self.mask_suffix)

        # Setting masks for GraphGym, which has to be one dimensional
        for mask_name in data.masks:
            data[mask_name] = data[mask_name][:, split_idx]
        data.train_mask = data[self.train_on + self.mask_suffix]

        # Add 'all_mask' to eliminate redundant model executions during the
        # evaluation step in the transductive node classification setting.
        data.all_mask = data.train_mask.new_tensor([1] * data.train_mask.shape[0])
        logging.info(data)

        # Two loaders, one for train and one for all. Note that the shuffle
        # option does nothing in the full batch setting.
        loaders = [
            get_loader([data], "full_batch", 1, shuffle=True),
            get_loader([data], "full_batch", 1, shuffle=False),
        ]

        return loaders

    @torch.no_grad()
    def evaluate(self, loaders, model, masks: List[str]):
        """Evaluate the model performance at a specific epoch.

        First obtain the prediction values using the 'all_mask'. Then compute
        evaluation metric using a specific mask on the full predictions.

        """
        model.eval()

        # Full batch used in the transduction node classification setting.
        full_loader = loaders[1]
        batch = list(full_loader)[0]
        batch.split = "all"
        batch.to(torch.device(cfg_gg.device))
        pred, true = model(batch)
        pred, true = pred.detach().cpu().numpy(), true.detach().cpu().numpy()

        results = {}
        for metric_name, metric_func in self.metrics.items():
            for mask_name in masks:
                mask = batch[mask_name].detach().cpu().numpy()
                score = metric_func(true[mask], pred[mask])
                score_name = f"{mask_name.split(self.mask_suffix)[0]}_{metric_name}"
                results[score_name] = score

        results["time_per_epoch"] = self._elapse() / self.eval_steps

        return results

    def train(self, model, dataset, split_idx=0):
        """Train model using GraphGym.

        Note that because obnb only concerns transductive node classification
        (for now), the training procedure is reduced to this specific setting
        for the sake of runtime performance.

        """
        masks = [mask_name + self.mask_suffix for mask_name in dataset.masks]

        logger_gg = Logger_gg(name="train")
        loaders = self.get_loaders(dataset, split_idx)

        optimizer = pyg_gg.create_optimizer(model.parameters(), cfg_gg.optim)
        scheduler = pyg_gg.create_scheduler(optimizer, cfg_gg.optim)

        stats, best_stats, best_model_state = self.new_stats(masks)
        for cur_epoch in range(cfg_gg.optim.max_epoch):
            train_epoch(logger_gg, loaders[0], model, optimizer, scheduler)

            if self.is_eval_epoch(cur_epoch):
                new_results = self.evaluate(loaders, model, masks)
                self.update_stats(
                    model,
                    stats,
                    best_stats,
                    best_model_state,
                    new_results,
                    cur_epoch,
                    logger_gg.basic()["loss"],
                )
                logger_gg.reset()
                logging.info(new_results)

        logger_gg.close()
        logging.info(f"Task done, results saved in {cfg_gg.run_dir}")

        # Rewind back to best model
        model.load_state_dict(best_model_state)

        return best_stats


def graphgym_model_wrapper(model):
    """Wrap a GraphGym model to take PyG data as input."""

    @wraps(model)
    def wrapped_model(data):
        batch = Batch.from_data_list([data])
        batch.all_mask = torch.ones(data.y.shape[0], dtype=bool)
        batch.split = "all"
        pred, true = model(batch)
        return pred.detach().cpu().numpy(), true.detach().cpu().numpy()

    return wrapped_model
