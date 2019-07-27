import numpy as np
from NLEval.util import checkers
from copy import deepcopy

class IDmap:
	def __init__(self):
		self._map = {}
		self._lst = []

	def __iter__(self):
		"""Yield all IDs"""
		return self.lst.__iter__()

	def __eq__(self, idmap):
		"""Return true if two idmaps have same set of IDs"""
		return set(self.lst) == set(idmap.lst)

	def __add__(self, other):
		new = self.copy()
		for ID in other:
			if ID not in new:
				new.addID(ID)
		return new

	def __sub__(self, other):
		new = self.__class__()
		for ID in self:
			if ID not in other:
				new.addID(ID)
		return new

	def __contains__(self, key):
		return key in self._map

	def __getitem__(self, key):
		"""Return (array of) index of key"""
		if isinstance(key, (list,np.ndarray)):
			idx = []
			for i in key:
				idx.append(self._map[i])
			return np.array(idx)
		else:
			checkers.checkType('key', (str, list, np.ndarray), key)
			return self._map[key]

	@property
	def size(self):
		"""int: number of IDs in map"""
		return len(self._map)

	@property
	def map(self):
		"""(dict of str:int): map from ID to index"""
		return self._map
	
	@property
	def lst(self):
		"""(:obj:`list` of :obj:`str`): list of IDs in index order"""
		return self._lst

	def copy(self):
		return deepcopy(self)

	def popID(self, ID):
		idx = self.map.pop(ID)
		self.lst.pop(idx)
		for i, ID in enumerate(self.lst[idx:]):
			self.map[ID] = idx + i
	
	def addID(self, ID):
		"""Add new ID as string, append last"""
		#check if ID already exist
		valid_type = checkers.INT_TYPE + checkers.FLOAT_TYPE + (str,)
		checkers.checkType('ID', valid_type, ID)
		try:
			num = float(ID)
			#convert to int string if numeric and is int
			ID = str(int(num)) if num % 1 == 0 else str(num)
		except ValueError:
			pass
		assert ID not in self, "ID %s exist"%repr(ID)
		self._map[ID] = self.size
		self._lst.append(ID)

	def getID(self, idx):
		"""Return ID at index"""
		return self._lst[idx]