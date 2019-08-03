from sklearn import model_selection as ms
from NLEval.valsplit.Base import *

class sklInterface(BaseInterface):
	"""Dedicated interface for Scikit Learn splitter"""
	def __init__(self, sklSplitter):
		super(sklInterface, self).__init__()
		self.setup_split_func(sklSplitter.split)

class sklSKF(sklInterface):
	"""Dedicated interface for Stratified K-Fold in SKLearn"""
	def __init__(self, n_splits=5, shuffle=False, random_state=None):
		super(sklSKF, self).__init__(\
			ms.StratifiedKFold(n_splits=n_splits, \
				shuffle=shuffle, random_state=random_state))

class sklSSS(sklInterface):
	"""Dedicated interface for Stratified Shuffle Split in SKLearn"""
	def __init__(self, n_splits=5, test_size=0.1, \
				train_size=None, random_state=None):
		super(sklSSS, self).__init__(\
			ms.StratifiedShuffleSplit(n_splits=n_splits, \
				test_size=test_size, train_size=train_size, \
				random_state=random_state))

class sklLOO(sklInterface):
	"""Dedicated interface for Leave One Out in SKLearn"""
	def __init__(self, P):
		super(sklLOO, self).__init__(ms.LeavePOut(P))

class sklLPO(sklInterface):
	"""Dedicated interface for Leave P Out in SKLearn"""
	def __init__(self):
		super(sklLPO, self).__init__(ms.LeaveOneOut())
