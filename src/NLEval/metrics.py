from typing import no_type_check

import numpy as np
import sklearn.metrics


def prior(y_true: np.ndarray) -> float:
    """Return the prior of a label vector.

    The portion of positive examples.
    """
    return (y_true > 0).sum() / len(y_true)


def log2_auprc_prior(y_true: np.ndarray, y_predict: np.ndarray) -> float:
    """Log2 auprc over prior."""
    if skip(y_true, y_predict):
        return np.nan
    return np.log2(
        sklearn.metrics.average_precision_score(y_true, y_predict)
        / prior(y_true),
    )


def precision_at_topk(y_true: np.ndarray, y_predict: np.ndarray) -> float:
    """Precision at top k."""
    if skip(y_true, y_predict):
        return np.nan
    k = (y_true > 0).sum()
    rank = y_predict.argsort()[::-1]
    nhits = (y_true[rank[:k]] > 0).sum()
    return -np.inf if nhits == 0 else np.log2(nhits * len(y_true) / k ** 2)


def auroc(y_true: np.ndarray, y_predict: np.ndarray) -> float:
    """AUROC metric."""
    if skip(y_true, y_predict):
        return np.nan
    return sklearn.metrics.roc_auc_score(y_true, y_predict)


@no_type_check
def skip(y_true, y_predict):
    """Wehter to skip the metric computation or not."""
    if y_true is None and y_predict is None:
        return True
    return False
