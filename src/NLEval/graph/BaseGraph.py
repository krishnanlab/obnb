from NLEval.util.IDmap import IDmap
from NLEval.util import checkers
import numpy as np

class BaseGraph:
	"""Base Graph object that contains basic graph operations"""
	def __init__(self):
		super().__init__()
		self.IDmap = IDmap()
		
	@property
	def IDmap(self):
		return self._IDmap

	@IDmap.setter
	def IDmap(self, idmap):
		checkers.checkType('idmap', IDmap, idmap)
		self._IDmap = idmap

	def __contains__(self, graph):
		"""Return true if contains the input graph"""
		for ID in graph.IDmap: #check if containes all IDs in input graph
			if ID not in self.IDmap:
				return False
		for ID1 in graph.IDmap: #check if all connections match
			for ID2 in graph.IDmap:
				if self.get_edge(ID1, ID2) != graph.get_edge(ID1, ID2):
					return False
		return True

	def __eq__(self, graph):
		"""Return true if input graph is identical to self
		i.e. same set of IDs with same connections
		"""
		if self.IDmap == graph.IDmap:
			return graph in self
		return False

	@property
	def size(self):
		"""int: number of nodes in graph"""
		return self.IDmap.size