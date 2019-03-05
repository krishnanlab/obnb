import numpy as np

class IDmap:
	def __init__(self):
		self._data = {}
		self._lst = []

	def __contains__(self, key):
		return key in self._data

	def __getitem__(self, key):
		if isinstance(key, (list,np.ndarray)):
			idx = []
			for i in key:
				idx.append(self._data[i])
			return np.array(idx)
		else:
			return self._data[key]

	@property
	def size(self):
		return len(self._data)

	@property
	def data(self):
		return self._data
	
	@property
	def lst(self):
		return self._lst
	
	def addID(self, ID):
		assert ID not in self,"ID:\t'%s'\texist"%ID
		self._data[ID] = self.size
		self._lst.append(ID)

	def idx2ID(self, idx):
		return self._lst[idx]		