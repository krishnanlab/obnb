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
		if val.size > 0:
			checkers.checkNumpyArrayNDim('val', 2, val)
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

class FeatureVec(DenseGraph):
	"""Feature vectors with ID maps"""
	def __init__(self, dim=None):
		super().__init__()
		self.dim = dim

	@property
	def dim(self):
		"""int: dimension of feature vectors"""
		return self._dim

	@dim.setter
	def dim(self, d):
		checkers.checkTypeAllowNone('d', (int, np.integer), d)
		if not self.isempty():
			if d != self.mat.shape[1]:
				if self.dim != self.mat.shape[1]:
					#self.dim should always in sync with actual dimension of feature vectors
					print("CRITICAL: This should never happen!")
				raise ValueError("Inconsistent dimension between input (%d) and data (%d)"%\
					(d, self.mat.shape[1]))
		self._dim = d

	@DenseGraph.mat.setter
	def mat(self, val):
		DenseGraph.mat.fset(self, val)
		if val.size > 0:
			self.dim = val.shape[1]

	def addVec(self, ID, vec):
		"""Add a new feature vector"""
		if self.isempty():
			if self.dim is not None:
				checkers.checkNumpyArrayShape('vec', self.dim, vec)
			else:
				self.dim = vec.shape[0]
			self.mat = vec.copy()
		else:
			self.mat = np.vstack([self.mat, vec])
		self.IDmap.addID(ID)

	@classmethod
	def from_emd(cls, path_to_emd, **kwargs):
		fvec_lst = []
		idmap = IDmap()
		with open(path_to_emd, 'r') as f:
			f.readline() # skip header
			for line in f:
				terms = line.split(' ')
				ID = terms[0].strip()
				idmap.addID(ID)
				fvec_lst.append(np.array(terms[1:], dtype=float))
		mat = np.asarray(fvec_lst)
		return cls.construct_graph(idmap, mat)
