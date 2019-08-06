from sklearn import model_selection as ms
from NLEval.valsplit.Base import *

__all__ = ['sklSKF', 'sklSSS', 'sklLOO', 'sklLPO']

class sklInterface(BaseInterface):
	"""Dedicated interface for Scikit Learn splitter"""
	def __init__(self, sklSplitter, skl_kws={}, shuffle=False):
		super(sklInterface, self).__init__(shuffle=shuffle)
		splitter = sklSplitter(**skl_kws)
		self.setup_split_func(splitter.split)

class sklSKF(sklInterface):
	"""Dedicated interface for Stratified K-Fold in SKLearn"""
	def __init__(self, skl_kws={}, shuffle=False):
		super(sklSKF, self).__init__(ms.StratifiedKFold)

class sklSSS(sklInterface):
	"""Dedicated interface for Stratified Shuffle Split in SKLearn"""
	def __init__(self, skl_kws={}, shuffle=False):
		super(sklSSS, self).__init__(ms.StratifiedShuffleSplit)

class sklLOO(sklInterface):
	"""Dedicated interface for Leave One Out in SKLearn"""
	def __init__(self, skl_kws={}, shuffle=False):
		super(sklLOO, self).__init__(ms.LeavePOut)

class sklLPO(sklInterface):
	"""Dedicated interface for Leave P Out in SKLearn"""
	def __init__(self, skl_kws={}, shuffle=False):
		super(sklLPO, self).__init__(ms.LeaveOneOu)
