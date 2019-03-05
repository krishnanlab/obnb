import numpy as np

class IDmap:
	def __init__(self):
		self._data = {}
		self._lst = []

	def __contains__(self, key):
		return key in self._data

	@property
	def size(self):
		return len(self._data)

	@property
	def data(self):
		return self._data
	
	@property
	def lst(self):
		return self._lst
	
	def newID(self, ID):
		assert ID not in self,"ID:\t'%s'\texist"%ID
		self._data[ID] = self.size
		self._lst.append(ID)

	def ID2idx(self, ID):
		return self._data[ID]

	def idx2ID(self, idx):
		return self._lst[idx]

	def IDary2idxary( self, ID ):
		idx = []
		try:
			for i in ID:
				idx.append(self.ID2idx(i))
		except TypeError:
			idx.append(self.ID2idx(ID))
		return np.array(idx)