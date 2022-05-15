"""Custom metrics comparible with GraphGym logger."""
from functools import wraps

import torch
from torch import Tensor

from ..typing import List, Metric
from .standard import auroc, log2_auprc_prior, precision_at_topk

__all__ = [
    "graphgym_auroc",
    "graphgym_log2_auprc_prior",
    "graphgym_precision_at_topk",
]


def graphgym_metric_wrapper(metric_func: Metric):
    """Wrap standard metric function into a GraphGym compatible one."""

    @wraps(metric_func)
    def graphgym_compatible_metric(
        y_true: List[Tensor],
        y_pred: List[Tensor],
        *args,
    ) -> float:
        y_true_np = torch.cat(y_true).detach().cpu().numpy()
        y_pred_np = torch.cat(y_pred).detach().cpu().numpy()
        return metric_func(y_true_np, y_pred_np)

    return graphgym_compatible_metric


graphgym_auroc = graphgym_metric_wrapper(auroc)
graphgym_log2_auprc_prior = graphgym_metric_wrapper(log2_auprc_prior)
graphgym_precision_at_topk = graphgym_metric_wrapper(precision_at_topk)
