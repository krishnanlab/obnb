import sklearn.metrics
import numpy as np

def prior(y_true):
    return (y_true > 0).sum() / len(y_true)

def auPRC(y_true, y_predict):
    if skip(y_true, y_predict):
        return np.nan
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_predict)
    return np.log2(sklearn.metrics.auc(recall, precision) / prior(y_true))

def PTopK(y_true, y_predict):
    if skip(y_true, y_predict):
        return np.nan
    k = (y_true > 0).sum()
    rank = y_predict.argsort()[::-1]
    nhits = (y_true[rank[:k]] > 0).sum()
    return -np.inf if nhits == 0 else np.log2(nhits * len(y_true) / k**2)

def auROC(y_true, y_predict):
    if skip(y_true, y_predict):
        return np.nan
    return sklearn.metrics.roc_auc_score(y_true, y_predict)

def skip(y_true, y_predict):
    if y_true is None and y_predict is None:
        return True
    return False