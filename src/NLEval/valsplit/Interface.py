from sklearn import model_selection as ms
from .Base import *

class sklSKF(BaseInterface):
	'''
	Dedicated interace for Stratified K-Fold in SKLearn
	'''
	def __init__(self, min_pos=10, n_splits=5, shuffle=False, random_state=None):
		super().__init__(\
			ms.StratifiedKFold(n_splits=n_splits, \
					shuffle=shuffle, random_state=random_state), \
			min_pos=min_pos)

class sklSSS(BaseInterface):
	'''
	Dedicated interface for Stratified Shuffle Split in SKLearn
	'''
	def __init__(self, min_pos=10, n_splits=5, test_size=0.1, \
				train_size=None, random_state=None):
		super().__init__(\
			ms.StratifiedShuffleSplit(n_splits=n_splits, \
						test_size=test_size, train_size=train_size, \
						random_state=random_state), \
			min_pos=min_pos)

class sklLOO(BaseInterface):
	'''
	Dedicated interface for Leave One Out in SKLearn
	'''
	def __init__(self, P, min_pos=10):
		super().__init__(ms.LeavePOut(P), min_pos=min_pos)

class sklLPO(BaseInterface):
	'''
	Dedicated interface for Leave P Out in SKLearn
	'''
	def __init__(self, min_pos=10):
		super().__init__(ms.LeaveOneOut(), min_pos=min_pos)


