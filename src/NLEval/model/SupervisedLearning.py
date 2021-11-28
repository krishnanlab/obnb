import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from NLEval.model.BaseModel import BaseModel
from copy import deepcopy

__all__ = ["SLBase", "LogReg", "SVM", "RF"]


class SLBase(BaseModel):
    def train(self, ID_ary, y):
        x = self.G[ID_ary]
        self.fit(x, y)

    def decision(self, ID_ary):
        x = self.G[ID_ary]
        decision_ary = self.decision_function(x)
        return decision_ary


class LogReg(SLBase, LogisticRegression):
    def __init__(self, G, **kwargs):
        SLBase.__init__(self, G)
        LogisticRegression.__init__(self, **kwargs)


class LogRegCV(SLBase, LogisticRegressionCV):
    def __init__(self, G, **kwargs):
        SLBase.__init__(self, G)
        LogisticRegressionCV.__init__(self, **kwargs)


class CombSLBase(BaseModel):
    def __init__(self, G, **kwargs):
        # G here should be multi feature set
        BaseModel.__init__(self, G)
        self.mdl_list = [self.base_mdl(**kwargs) for i in self.G.mat_list]

    def train(self, ID_ary, y):
        for i, mat in enumerate(self.G.mat_list):
            x = mat[self.G.IDmap[ID_ary]]
            self.mdl_list[i].fit(x, y)
        self.fit_master_mdl(ID_ary, y)


class CombLogRegCVBagging(CombSLBase):
    def __init__(self, G, **kwargs):
        self.base_mdl = LogisticRegressionCV
        CombSLBase.__init__(self, G, **kwargs)

    def fit_master_mdl(self, ID_ary, y):
        pass

    def decision(self, ID_ary):
        decision_ary = np.zeros((len(ID_ary)))
        for i, mat in enumerate(self.G.mat_list):
            x = mat[self.G.IDmap[ID_ary]]
            decision_ary += self.mdl_list[i].decision_function(x)
        decision_ary /= len(self.mdl_list)
        return decision_ary


class CombLogRegCVPredComb(CombSLBase):
    def __init__(self, G, mixing_ratio=0.5, **kwargs):
        if len(G.mat_list) != 2:
            raise ValueError(
                "PredComb only takes two input features sets, "
                + +f"but the input has {len(G.mat_list)} "
                + "number of feature sets",
            )
        self.mixing_ratio = mixing_ratio
        self.base_mdl = LogisticRegressionCV
        CombSLBase.__init__(self, G, **kwargs)

    def fit_master_mdl(self, ID_ary, y):
        pass

    def decision(self, ID_ary):
        decision_ary = np.zeros((len(ID_ary)))
        factors = [self.mixing_ratio, 1 - self.mixing_ratio]
        for i, mat in enumerate(self.G.mat_list):
            x = mat[self.G.IDmap[ID_ary]]
            decision_ary += factors[i] * self.mdl_list[i].decision_function(x)
        return decision_ary


class CombLogRegCVAdaBoost(CombSLBase):
    def __init__(self, G, exclude=True, n_mdl=None, **kwargs):
        """Initialize LogisticRegression AdaBoost type ensemble.

        Args:
            G (NLEval.graph.DenseGraph.MultiFeatureVec): multi-feature objects
                that contains multiple feature sets
                exclude (bool): whether or not to exclude feature set upon selection
            n_mdl (int): number of models to train, default is None, which uses
                the number of feature sets as n_mdl. Only used when exclude is
                set to be True
        """
        self.base_mdl = LogisticRegressionCV
        CombSLBase.__init__(self, G, **kwargs)
        self.coef_ = None
        self.exclude = exclude
        if self.exclude:
            if n_mdl is not None:
                print(
                    f"Warning: n_mdl set to be {repr(n_mdl)} "
                    + "with exclude=False, set to None implicitly.",
                )
                n_mdl = None
        self.n_mdl = n_mdl

    def fit_master_mdl(self, ID_ary, y):
        n_mdl = (
            len(self.mdl_list) if self.n_mdl is None else self.n_mdl
        )  # total number of models
        w = np.ones(len(ID_ary)) / len(ID_ary)  # data point weights
        coef = np.zeros(n_mdl)  # model boosting coefficients
        y_pred_mat = np.zeros(
            (len(ID_ary), n_mdl), dtype=bool,
        )  # predictions from all models
        idx_ary = self.G.IDmap[ID_ary]

        if self.exclude:
            selected_ind = np.zeros(
                n_mdl, dtype=bool,
            )  # inidvator for selected model
        else:
            mdl_idx_ary = np.zeros(
                n_mdl, dtype=int,
            )  # index of features of corresponding boosting coefficients
            mdl_list = self.mdl_list
            self.mdl_list = (
                []
            )  # need to make new model list, previously tied to feature set index

        # determine boosting coefficients
        for i in range(n_mdl):
            opt_err = np.inf
            opt_idx = None

            for j in range(len(self.G.mat_list)):
                if self.exclude:
                    if selected_ind[j]:
                        continue
                    mdl = self.mdl_list[j]
                else:
                    mdl = mdl_list[j]

                # retrain model using sample weights
                x = self.G.mat_list[j][idx_ary]
                if (
                    i > 0
                ):  # for first iteration, the model are already train with uniform weight
                    mdl.fit(x, y, sample_weight=w / w.sum())
                y_pred_mat[:, j] = mdl.predict(x)

                err = w[y_pred_mat[:, j] != y].sum()
                if err < opt_err:
                    opt_err = err
                    opt_idx = j

            a = 0.5 * np.log((1 - opt_err) / opt_err)  # model coefficient
            if a < 0:
                print(
                    f"Warning: encountered worse than random prediction, a = {a}, set to 0",
                )
                a = 0
            y_pred_opt = (
                y_pred_mat[:, opt_idx] == 1
            )  # predictions of optimal model
            w[y_pred_opt == y] *= np.exp(-a)  # down weight correct predictions
            w[y_pred_opt != y] *= np.exp(a)  # up weight incorrect predictions
            w[w < 0.01] = 0.01  # prevent zero sample weight
            w /= w.sum()  # normalize data point weights
            if self.exclude:
                selected_ind[
                    opt_idx
                ] = True  # remove selected model from candidates
                coef[opt_idx] = a
            else:
                mdl_idx_ary[i] = opt_idx
                self.mdl_list.append(deepcopy(mdl_list[opt_idx]))
                coef[i] = a
            # print(f"Iter = {i}, optidx = {opt_idx}, optimal error = {opt_err}, accuracy = {(y_pred_opt==y).sum() / len(y)}")

        if coef.sum() == 0:
            coef = np.ones(n_mdl) / n_mdl
        else:
            coef /= coef.sum()
        # print(coef)

        self.coef_ = coef  # set normalized bossting coefficients
        if not self.exclude:
            # print(mdl_idx_ary, '\n', coef, '\n')
            self.mdlidx_ = mdl_idx_ary

    def decision(self, ID_ary):
        if self.coef_ is None:
            raise ValueError(
                "Master model untrained, train first using fit_master_mdl",
            )

        idx_ary = self.G.IDmap[ID_ary]
        decision_ary = np.zeros((len(ID_ary)))
        iter_list = (
            list(range(len(self.G.mat_list))) if self.exclude else self.mdlidx_
        )
        for i, j in enumerate(iter_list):
            x = self.G.mat_list[j][idx_ary]
            decision_ary += self.coef_[i] * self.mdl_list[i].decision_function(
                x,
            )

        return decision_ary


class CombLogRegCVModifiedRankBoost(CombSLBase):
    def __init__(self, G, exclude=True, n_mdl=None, retrain=True, **kwargs):
        """Initialize LogisticRegression ModifiedRankBoost ensemble.

        Notes:
            This implementation follows from http://pages.cs.wisc.edu/~shavlik/abstracts/oliphant.ilp09.abstract.html

        Args:
            G (NLEval.graph.DenseGraph.MultiFeatureVec): multi-feature objects
                that contains multiple feature sets
                exclude (bool): whether or not to exclude feature set upon selection
            n_mdl (int): number of models to train, default is None, which uses
                the number of feature sets as n_mdl. Only used when exclude is
                set to be True
            retrain (bool): whether or not to retrain model in each iteration
                using sample weights, default is True

        """
        self.base_mdl = LogisticRegressionCV
        CombSLBase.__init__(self, G, **kwargs)
        self.coef_ = None
        self.exclude = exclude
        self.retrain = retrain
        if self.exclude:
            if n_mdl is not None:
                print(
                    f"Warning: n_mdl set to be {repr(n_mdl)} "
                    + "with exclude=False, set to None implicitly.",
                )
                n_mdl = None
        self.n_mdl = n_mdl

    def fit_master_mdl(self, ID_ary, y):
        n_mdl = (
            len(self.mdl_list) if self.n_mdl is None else self.n_mdl
        )  # total number of models
        n_pos = y.sum()
        n_neg = (~y).sum()
        skew = n_neg / n_pos

        w = np.ones(len(ID_ary)) / n_pos  # data point weights
        coef = np.zeros(n_mdl)  # model boosting coefficients
        y_pred_mat = np.zeros(
            (len(ID_ary), len(self.G.mat_list)), dtype=bool,
        )  # predictions from all features
        idx_ary = self.G.IDmap[ID_ary]

        if self.exclude:
            selected_ind = np.zeros(
                n_mdl, dtype=bool,
            )  # inidvator for selected model
        else:
            mdl_idx_ary = np.zeros(
                n_mdl, dtype=int,
            )  # index of features of corresponding boosting coefficients
            mdl_list = self.mdl_list
            self.mdl_list = (
                []
            )  # need to make new model list, previously tied to feature set index

        # determine boosting coefficients
        for i in range(n_mdl):
            opt_r = 0
            opt_idx = None

            for j in range(len(self.G.mat_list)):
                if self.exclude:
                    if selected_ind[j]:
                        continue
                    mdl = self.mdl_list[j]
                else:
                    mdl = mdl_list[j]

                # retrain model using sample weights
                if (i > 0) & self.retrain:
                    x = self.G.mat_list[j][idx_ary]
                    mdl.fit(x, y, sample_weight=w / w.sum())
                    y_pred_mat[:, j] = mdl.predict(x)

                r = average_precision_score(
                    y, y_pred_mat[:, j], sample_weight=w / w.sum(),
                )
                if r > opt_r:
                    opt_r = r
                    opt_idx = j

            opt_r = min(
                opt_r, 0.99,
            )  # prevent auprc of 1, causes divide by zero for a
            a = 0.5 * np.log((1 + opt_r) / (1 - opt_r))  # model coefficient
            y_pred_opt = y_pred_mat[:, opt_idx]  # decision scores of optim mdl

            w[y] *= w[y] * np.exp(-a * y_pred_opt[y])  # down weight positives
            w[~y] *= w[~y] * np.exp(
                a * y_pred_opt[~y],
            )  # up weight negatives, weighted by skew
            w[w < 0.01] = 0.01  # prevent zero sample weight
            w[y] /= w[y].sum()
            w[~y] /= w[~y].sum()
            w[~y] *= skew
            # if (w == 0).sum() > 0:
            #   print("FK", (w==0).sum(), (w==0).sum()/len(w))

            if self.exclude:
                selected_ind[
                    opt_idx
                ] = True  # remove selected model from candidates
                coef[opt_idx] = a
            else:
                mdl_idx_ary[i] = opt_idx
                self.mdl_list.append(deepcopy(mdl_list[opt_idx]))
                coef[i] = a

            # print(f"Iter = {i}, optidx = {opt_idx}, optimal r = {opt_r}, \
        # prior = {n_pos / (n_pos + n_neg)}, auPRC = {np.log2(r * (n_pos + n_neg) / n_pos)}")

        coef /= coef.sum()
        # print(coef)

        self.coef_ = coef  # set normalized bossting coefficients
        if not self.exclude:
            # print(mdl_idx_ary, '\n', coef, '\n')
            self.mdlidx_ = mdl_idx_ary

    def decision(self, ID_ary):
        if self.coef_ is None:
            raise ValueError(
                "Master model untrained, train first using fit_master_mdl",
            )

        idx_ary = self.G.IDmap[ID_ary]
        decision_ary = np.zeros((len(ID_ary)))
        iter_list = (
            list(range(len(self.G.mat_list))) if self.exclude else self.mdlidx_
        )
        for i, j in enumerate(iter_list):
            x = self.G.mat_list[j][idx_ary]
            decision_ary += self.coef_[i] * self.mdl_list[i].decision_function(
                x,
            )

        return decision_ary


class SVM(SLBase, LinearSVC):
    def __init__(self, G, **kwargs):
        SLBase.__init__(self, G)
        LinearSVC.__init__(self, **kwargs)


class RF(SLBase, RandomForestClassifier):
    def __init__(self, G, **kwargs):
        SLBase.__init__(self, G)
        RandomForestClassifier.__init__(self, **kwargs)

    def decision(self, ID_ary):
        x = self.get_x(ID_ary)
        decision_ary = self.predict_proba(x)[:, 1]  # take positive class
        return decision_ary
