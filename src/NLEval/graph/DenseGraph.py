from NLEval.graph.BaseGraph import BaseGraph
from NLEval.graph.SparseGraph import SparseGraph
from NLEval.util.IDmap import IDmap
from NLEval.util import checkers
import numpy as np

class DenseGraph(BaseGraph):
	"""Base Graph object that stores data using numpy array"""
	def __init__(self):
		super().__init__()
		self.mat = np.array([])
	
	def __getitem__(self, key):
		"""Return slice of graph

		Args:
			key(str): key of ID
			key(:obj:`list` of :obj:`str`): list of keys of IDs
		"""
		if isinstance(key, slice):
			raise NotImplementedError
		idx = self.IDmap[key]
		return self.mat[idx]

	@property
	def mat(self):
		return self._mat
	
	@mat.setter
	def mat(self, val):
		checkers.checkType('val', np.ndarray, val)
		self._mat = val

	def get_edge(self, ID1, ID2):
		return self.mat[self.IDmap[ID1], self.IDmap[ID2]]

	@classmethod
	def construct_graph(cls, idmap, mat):
		assert idmap.size == mat.shape[0]
		graph = cls()
		graph.IDmap = idmap
		graph.mat = mat
		return graph

	@classmethod
	def from_mat(cls, mat):
		"""Construct BaseGraph object from numpy array
		First column of mat encodes ID
		"""
		idmap = IDmap()
		for ID in mat[:,0]:
			idmap.addID(ID)
		return cls.construct_graph(idmap, mat[:,1:].astype(float))

	@classmethod
	def from_npy(cls, path_to_npy, **kwargs):
		"""Read numpy array from .npy file and construct BaseGraph"""
		mat = np.load(path_to_npy, **kwargs)
		return cls.from_mat(mat)

	@classmethod
	def from_edglst(cls, path_to_edglst, weighted, directed, **kwargs):
		"""Read from edgelist and construct BaseGraph"""
		graph = SparseGraph.from_edglst(path_to_edglst, weighted, directed, **kwargs)
		return cls.construct_graph(graph.IDmap, graph.to_adjmat())