from NLEval.graph import BaseGraph
from NLEval.util import checkers
import numpy as np

class BaseModel:
	"""Base model object"""
	def __init__(self, g):
		super(BaseModel, self).__init__()
		self.G = g

	@property
	def G(self):
		""":obj:`NLEval.Graph.BaseGraph`: graph object"""
		return self._G
	
	@G.setter
	def G(self, g):
		checkers.checkType('Graph', BaseGraph.BaseGraph, g)
		self._G = g

	def get_idx_ary(self, IDs):
		"""Return indices of corresponding input IDs

		Note:
			All ID in the input ID list must be in IDmap of graph

		Args:
			IDs(:obj:`list` of str): list of ID in IDmap

		Returns:
			(:obj:`numpy.ndarray`): numpy array of indices of input IDs

		"""
		return self.G.IDmap[IDs]

	def get_x(self, IDs):
		"""Return features of input IDs as corresponding rows in graph"""
		idx_ary = self.get_idx_ary(IDs)
		return self.G.mat[idx_ary]

	def test(self, labelset_splitgen):
		y_true = np.array([])
		y_predict = np.array([])
		for train_id_ary, train_label_ary, test_id_ary, test_label_ary in labelset_splitgen:
			if train_id_ary is None:
				return None, None
			self.train(train_id_ary, train_label_ary)
			decision_ary = self.decision(test_id_ary)
			y_true = np.append(y_true, test_label_ary)
			y_predict = np.append(y_predict, decision_ary)
		return y_true, y_predict
