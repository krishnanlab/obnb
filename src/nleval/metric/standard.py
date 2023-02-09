"""Standard metric extending those available in sklearn."""
from functools import wraps

import numpy as np
import sklearn.metrics


def wrap_metric(metric_func):
    """Wrap metric function with common processing steps.

    - Skip computation if None
    - Perturn reduction when calculating metrics in a multi-class setting

    """

    @wraps(metric_func)
    def wrapped(y_true, y_pred, reduce="mean"):
        """Metric function with common processing steps.

        Args:
            y_true: True label.
            y_pred: Predicted values.
            reduce: Reduction strategy to use when y_true and y_pred are
                2-dimensional, with examples along the rows and label-class
                along the columns. Accepted options: ['none', 'mean', 'median']

        """
        if reduce not in ["none", "mean", "median"]:
            raise ValueError(f"Unknown reduce option {reduce!r}")

        if _skip(y_true, y_pred):
            return np.nan

        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            return metric_func(y_true, y_pred)
        else:
            scores = [metric_func(i, j) for i, j in zip(y_true.T, y_pred.T)]

            if reduce == "none":
                score = np.array(scores)
            elif reduce == "mean":
                score = np.mean(scores)
            elif reduce == "median":
                score = np.median(scores)
            else:
                raise ValueError("This should never happen")

            return score

    return wrapped


def _skip(y_true, y_predict):
    """Wehter to skip the metric computation or not."""
    if y_true is None and y_predict is None:
        return True
    return False


def prior(y_true: np.ndarray) -> float:
    """Return the prior of a label vector.

    The portion of positive examples.

    """
    return (y_true > 0).sum() / y_true.size


@wrap_metric
def log2_auprc_prior(y_true: np.ndarray, y_predict: np.ndarray) -> float:
    """Log2 auprc over prior."""
    return np.log2(
        sklearn.metrics.average_precision_score(y_true, y_predict) / prior(y_true),
    )


@wrap_metric
def precision_at_topk(y_true: np.ndarray, y_predict: np.ndarray) -> float:
    """Precision at top k."""
    k = (y_true > 0).sum()
    rank = y_predict.argsort()[::-1]
    nhits = (y_true[rank[:k]] > 0).sum()
    return -np.inf if nhits == 0 else np.log2(nhits * len(y_true) / k**2)


@wrap_metric
def auroc(y_true: np.ndarray, y_predict: np.ndarray) -> float:
    """AUROC metric."""
    return sklearn.metrics.roc_auc_score(y_true, y_predict)
