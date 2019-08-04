from NLEval.util import checkers, IDHandler
import numpy as np

class BaseValSplit:
	def __init__(self, shuffle=False):
		super(BaseValSplit, self).__init__()
		self.shuffle = shuffle

	@property
	def shuffle(self):
		"""bool: shufle train/test indices when splitting"""
		return self._shuffle
	
	@shuffle.setter
	def shuffle(self, val):
		checkers.checkType("shuffle", bool, val)
		self._shuffle = val

	def split(self, ID_ary, label_ary):
		"""Split samples into training and testing sets

		Note:
			ID_ary and label_ary are coulpled, such that a particular entry
			in label_ary corresponds to the label of the ID in the same entry
			in ID_ary.

		Args:
			ID_ary(:obj:`numpy.ndarray` of :obj:`str`): array of entity IDs
			label_ary(:obj:`numpy.ndarray` of :obj:`bool`):
				boolean/binary array of indicating positive samples

		Yields:
			train_ID_ary(:obj:`numpy.ndarray` of :obj:`str`):
				numpy array of training IDs
			train_label_ary(:obj:`numpy.ndarray` of :obj:`bool`):
				numpy array of training labels
			test_ID_ary(:obj:`numpy.ndarray` of :obj:`str`):
				numpy array of testing IDs
			test_label_ary(:obj:`numpy.ndarray` of :obj:`bool`):
				numpy array of testing labels

		"""
		for train_idx_ary, test_idx_ary in self.get_split_idx_ary(ID_list, label_ary):
			if self.shuffle:
				np.random.shuffle(train_idx_ary)
				np.random.shuffle(test_idx_ary)
			train_ID_ary = ID_ary[train_idx_ary]
			train_label_ary = label_ary[train_idx_ary]
			test_ID_ary = ID_ary[test_idx_ary]
			test_label_ary = label_ary[test_idx_ary]
			yield train_ID_ary, train_label_ary, test_ID_ary, test_label_ary

class BaseHoldout(BaseValSplit):
	def __init__(self, train_on='top', shuffle=False):
		"""Generic holdout validation object

		Split based on some numeric properties of the samples, train on 
		either top or bottom set and test on the other, depending on 
		user specification of `train_on`. If 'top' specified, those samples 
		with properties of larger values are used for training, and those 
		with smaller values are used for testing, vice versa.

		"""
		super(BaseHoldout, self).__init__(shuffle=shuffle)
		self.train_on = train_on
		self._test_ID_ary = None
		self._train_ID_ary = None

	@property
	def train_ID_ary(self):
		return self._train_ID_ary.copy()
	
	@property
	def test_ID_ary(self):
		return self._test_ID_ary.copy()

	@property
	def train_on(self):
		"""str: train on top or bottom sample sets
			- 'top': train on top, test on bottom
			- 'bot': train on bottom, test on top
		"""
		return self._train_on
	
	@train_on.setter
	def train_on(self, val):
		checkers.checkTypeErrNone('train_on', str, val)
		if val not in ['top','bot']:
			raise ValueError("Train on must be 'top' or 'bot', not '%s'"%repr(val))
		self._train_on = val
		self._reverse = val == 'bot'

	@property
	def reverse(self):
		return self._reverse

	def get_common_ID_list(self, lscIDs, graphIDs):
		"""Get list of common IDs between labelset collection and graph

		Args:
			lscIDs(:obj:`NLEval.util.IDHandler.IDprop`):
			graphIDs(:obj:`NLEval.util.IDHandler.IDmap`):

		"""
		checkers.checkType("ID for labelset collection entities", IDHandler.IDprop, lscIDs)
		checkers.checkType("ID for graph entities", IDHandler.IDmap, graphIDs)
		common_ID_list = []
		for ID in graphIDs.lst:
			if ID in lscIDs:
				#make sure entity is part of at least one labelset
				if lscIDs.getProp(ID, 'Noccur') > 0:
					common_ID_list.append(ID)
		return common_ID_list

	def get_split_idx_ary(ID_ary, label_ary):
		assert (self._test_idx_ary is not None) | (self._train_idx_ary is not None),\
			"Training or testing sets not available, run `train_test_setup` first"
		train_idx_ary = np.in1d(ID_list, self.train_ID_ary)
		test_idx_ary = np.in1d(ID_list, self.test_ID_ary)
		return train_idx_ary, test_idx_ary
	

class BaseInterface(BaseValSplit):
	"""Base interface with user defined validation split generator"""
	def __init__(self, shuffle=False):
		super(BaseInterface, self).__init__(shuffle=shuffle)

	def setup_split_func(self, split_func):
		self.get_split_idx_ary = lambda ID_ary, label_ary: split_func(ID_ary, label_ary)
