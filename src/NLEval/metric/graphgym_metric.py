"""Custom metrics comparible with GraphGym logger."""
from functools import wraps
from typing import Callable
from typing import List

import numpy as np
import torch
from torch import Tensor

from .standard import auroc
from .standard import log2_auprc_prior
from .standard import precision_at_topk

__all__ = [
    "graphgym_auroc",
    "graphgym_log2_auprc_prior",
    "graphgym_precision_at_topk",
]

Metric = Callable[[np.ndarray, np.ndarray], float]
GraphGymMetric = Callable[[List[Tensor], List[Tensor], str], float]


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
