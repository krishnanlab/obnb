import numpy as np
from NLEval.util import checkers

class IDmap:
	def __init__(self):
		self._data = {}
		self._lst = []

	def __iter__(self):
		"""Yield all IDs"""
		return self.lst.__iter__()

	def __eq__(self, idmap):
		"""Return true if two idmaps have same set of IDs"""
		return set(self.lst) == set(idmap.lst)

	def __contains__(self, key):
		return key in self._data

	def __getitem__(self, key):
		"""Return (array of) index of key"""
		if isinstance(key, (list,np.ndarray)):
			idx = []
			for i in key:
				idx.append(self._data[i])
			return np.array(idx)
		else:
			checkers.checkType('key', (str, list, np.ndarray), key)
			return self._data[key]

	@property
	def size(self):
		"""int: number of IDs in map"""
		return len(self._data)

	@property
	def data(self):
		"""(dict of str:int): map from ID to index"""
		return self._data
	
	@property
	def lst(self):
		"""(:obj:`list` of :obj:`str`): list of IDs in index order"""
		return self._lst
	
	def addID(self, ID):
		"""Add new ID, append last"""
		#check if ID already exist
		assert ID not in self,"ID:\t'%s'\texist"%ID
		try:
			#convert to int string if numeric
			ID = str(int(ID))
		except ValueError:
			checkers.checkType('ID', (str,int), ID)
		self._data[ID] = self.size
		self._lst.append(ID)

	def idx2ID(self, idx):
		"""Return ID at index"""
		return self._lst[idx]