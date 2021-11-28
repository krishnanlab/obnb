import numpy as np
from NLEval.graph import BaseGraph
from NLEval.util import checkers

__all__ = ["BaseModel"]


class BaseModel:
    """Base model object."""

    def __init__(self, graph):
        super(BaseModel, self).__init__()
        self.graph = graph

    @property
    def graph(self):
        """:obj:`NLEval.Graph.BaseGraph`: graph object."""
        return self._G

    @graph.setter
    def graph(self, graph):
        checkers.checkType("Graph", BaseGraph.BaseGraph, graph)
        self._G = graph

    def get_idx_ary(self, node_ids):
        """Return indices of corresponding input IDs.

        Note:
            All ID in the input ID list must be in idmap of graph

        Args:
            node_ids(:obj:`list` of str): list of ID in idmap

        Returns:
            (:obj:`numpy.ndarray`): numpy array of indices of input IDs

        """
        return self.graph.idmap[node_ids]

    def get_x(self, node_ids):
        """Return features of input IDs as corresponding rows in graph."""
        idx_ary = self.get_idx_ary(node_ids)
        return self.graph.mat[idx_ary]

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

    def predict(self, pos_ids_set, neg_ids_set):
        """Network wise prediction.

        Given positive and negative examples, train the model and then generate
        predictions for all nodes in the network and return as prediction score
        dictionary.

        Input:
            pos_ids_set (:obj:`set` of :obj:`str`): set of IDs of positive examples.
            neg_ids_set (:obj:`set` of :obj:`str`): set of IDs of negative examples.

        Output:
            score_dict (:obj:`dict` of :obj:`str` -> :obj:`float`): dictionary
                mapping node IDs to prediction scores

        """
        graph = self.graph
        node_ids = graph.idmap.lst

        pos_ids_set = pos_ids_set & set(node_ids)
        neg_ids_set = neg_ids_set & set(node_ids)

        id_ary = np.array(list(pos_ids_set | neg_ids_set))
        label_ary = np.zeros(len(id_ary), dtype=bool)
        label_ary[: len(pos_ids_set)] = True

        self.train(id_ary, label_ary)
        scores = self.decision(node_ids)
        score_dict = {
            node_id: score for node_id, score in zip(node_ids, scores)
        }

        return score_dict
