import numpy as np
from NLEval.util import checkers
from copy import deepcopy

class IDlst(object):
	"""docstring for IDlst"""
	def __init__(self):
		super(IDlst, self).__init__()
		self._lst = []

	def __iter__(self):
		"""Yield all IDs"""
		return self.lst.__iter__()

	def __eq__(self, other):
		"""Return true if two IDlst have same set of IDs"""
		return set(self.lst) == set(other.lst)

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

	def __contains__(self, ID):
		return ID in self.lst

	def __getitem__(self, ID):
		"""Return (array of) index of ID"""
		if isinstance(ID, str):
			return self.__getitem_sinlge(ID)
		elif isinstance(ID, checkers.ITERABLE_TYPE):
			return self.__getitem_multiple(ID)
		else:
			raise TypeError("ID keys must be stirng or iterables of string, not %s"%\
					repr(type(ID)))

	def __getitem_sinlge(self, ID):
		assert ID in self, "Unknown ID: %s"%repr(ID)
		return self._lst.index(ID)

	def __getitem_multiple(self, IDs):
		checkers.checkTypesInIterable('IDs', str, IDs)
		idx_lst = []
		for ID in IDs:
			idx_lst.append(self.__getitem_sinlge(ID))
		return idx_lst

	@property
	def lst(self):
		return self._lst
	
	@property
	def size(self):
		return len(self.lst)

	def copy(self):
		return deepcopy(self)

	def popID(self, ID):
		checkers.checkType('ID', str, ID)
		assert ID in self, "Unknown ID: %s"%repr(ID)
		self.lst.pop(self[ID])

	def addID(self, ID):
		"""Add new ID as string, append last"""
		checkers.checkType('ID', checkers.NUMSTRING_TYPE, ID)
		try:
			num = float(ID)
			#convert to int string if numeric and is int
			ID = str(int(num)) if num % 1 == 0 else str(num)
		except ValueError:
			pass
		#check if ID already exist after clean up format
		assert ID not in self, "ID %s exist"%repr(ID)
		self._lst.append(ID)

	def getID(self, idx):
		return self.lst[idx]

class IDmap(IDlst):
	"""IDmap object that implements dictionary for more efficient mapping
	from ID to index"""
	def __init__(self):
		super(IDmap, self).__init__()
		self._map = {}

	def __contains__(self, ID):
		return ID in self.map

	def __getitem_sinlge(self, ID):
		assert ID in self, "Unknown ID: %s"%repr(ID)
		return self.map[ID]

	@property
	def map(self):
		"""(dict of str:int): map from ID to index"""
		return self._map

	def popID(self, ID):
		super(IDmap, self).popID(ID)
		idx = self.map.pop(ID)
		for i, ID in enumerate(self.lst[idx:]):
			self.map[ID] = idx + i
	
	def addID(self, ID):
		new_idx = self.size
		super(IDmap, self).addID(ID)
		self._map[self.lst[-1]] = new_idx