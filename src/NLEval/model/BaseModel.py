import numpy as np
from NLEval.graph import BaseGraph
from NLEval.util import checkers

__all__ = ["BaseModel"]


class BaseModel:
    """Base model object."""

    def __init__(self, g):
        super(BaseModel, self).__init__()
        self.G = g

    @property
    def G(self):
        """:obj:`NLEval.Graph.BaseGraph`: graph object."""
        return self._G

    @G.setter
    def G(self, g):
        checkers.checkType("Graph", BaseGraph.BaseGraph, g)
        self._G = g

    def get_idx_ary(self, IDs):
        """Return indices of corresponding input IDs.

        Note:
            All ID in the input ID list must be in idmap of graph

        Args:
            IDs(:obj:`list` of str): list of ID in idmap

        Returns:
            (:obj:`numpy.ndarray`): numpy array of indices of input IDs

        """
        return self.G.idmap[IDs]

    def get_x(self, IDs):
        """Return features of input IDs as corresponding rows in graph."""
        idx_ary = self.get_idx_ary(IDs)
        return self.G.mat[idx_ary]

    def test(self, labelset_splitgen):
        """Model testing through validation split.

        Input:
            labelset_splitgen: validation split helper objects, see example

        Output:
            y_true: numpy array of true values
            y_predict: numpy array of decision values

        TODO:
            Add example here (the valsplit object is from lsc)
            Now it only supports single class, how about multiclass predcitions?

        """
        y_true = np.array([])
        y_predict = np.array([])
        for arys in labelset_splitgen:
            train_id_ary, train_label_ary, test_id_ary, test_label_ary = arys
            if train_id_ary is None:
                return None, None
            self.train(train_id_ary, train_label_ary)
            decision_ary = self.decision(test_id_ary)
            y_true = np.append(y_true, test_label_ary)
            y_predict = np.append(y_predict, decision_ary)
        return y_true, y_predict

    def test2(self, labelset_splitgen):
        """Model testing through validation split and separate by splits.

        Same as test() above, but return y_true and y_pred as list of lists,
        grouping based on fold/split instead of merging into a single list

        """
        y_true = []
        y_predict = []
        for arys in labelset_splitgen:
            train_id_ary, train_label_ary, test_id_ary, test_label_ary = arys
            if train_id_ary is None:
                return None, None
            self.train(train_id_ary, train_label_ary)
            decision_ary = self.decision(test_id_ary)
            y_true.append(test_label_ary)
            y_predict.append(decision_ary)
            # print(self.C_)
        return y_true, y_predict

    def predict(self, pos_ID_set, neg_ID_set):
        """Network wise prediction.

        Given positive and negative examples, train the model and then generate
        predictions for all nodes in the network and return as prediction score
        dictionary.

        Input:
            pos_ID_set (:obj:`set` of :obj:`str`): set of IDs of positive examples.
            neg_ID_set (:obj:`set` of :obj:`str`): set of IDs of negative examples.

        Output:
            score_dict (:obj:`dict` of :obj:`str` -> :obj:`float`): dictionary
                mapping node IDs to prediction scores

        """
        G = self.G
        ID_list = G.idmap.lst

        pos_ID_set = pos_ID_set & set(ID_list)
        neg_ID_set = neg_ID_set & set(ID_list)

        ID_ary = np.array(list(pos_ID_set | neg_ID_set))
        label_ary = np.zeros(len(ID_ary), dtype=bool)
        label_ary[: len(pos_ID_set)] = True

        self.train(ID_ary, label_ary)
        scores = self.decision(ID_list)
        score_dict = {ID: score for ID, score in zip(ID_list, scores)}

        return score_dict
