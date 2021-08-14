from NLEval.util import checkers, IDHandler
from NLEval.valsplit.Base import *
import numpy as np

__all__ = ['BinHold', 'ThreshHold', 'CustomHold', 'TrainTestAll']

class TrainValTest(BaseHoldout):
	"""Split into train-val-test sets by ratios

	Sort the entities based on the desired properties and then prepare the 
	splits according to the train-val-test ratio.

	"""
	def __init__(self, train_ratio, test_ratio, train_on='top', shuffle=False):
		super(TrainValTest, self).__init__(train_on=train_on, shuffle=shuffle)
		self.train_ratio = train_ratio
		self.test_ratio = test_ratio

	def __repr__(self):
		# TODO: make repr a super magic fun, automatically generate for children.
		return 'TrainValTest(train_ratio=%s, test_ratio=%s, train_on=%s)'%\
		(repr(self.train_ratio), repr(self.test_ratio), repr(self.train_on))

	@property
	def train_ratio(self):
		return self._train_ratio

	@property
	def test_ratio(self):
		return self._test_ratio
	
	@train_ratio.setter
	def train_ratio(self, val):
		checkers.checkTypeErrNone('Training ratio', checkers.FLOAT_TYPE, val)
		if (val <= 0) | (val > 1):
			raise ValueError("Training ratio must be between 0 and 1, received value %f"%val)
		self._train_ratio = val
	
	@test_ratio.setter
	def test_ratio(self, val):
		checkers.checkTypeErrNone('Testing ratio', checkers.FLOAT_TYPE, val)
		if (val <= 0) | (val > 1):
			raise ValueError("Testing ratio must be between 0 and 1, received value %f"%val)
		if self.train_ratio + val >=1:
			raise ValueError("Sum of training and testing ratio must be less than 1" + 
				", received train_raio = %f, and test_ratio = %f"%(self.train_ratio, val))
		self._test_ratio = val

	def train_test_setup(self, lscIDs, nodeIDs, prop_name, **kwargs):
		lscIDs._check_prop_existence(prop_name, True)
		common_ID_list = self.get_common_ID_list(lscIDs, nodeIDs)
		sorted_ID_list = sorted(common_ID_list, reverse=self.train_on=='bot', \
			key=lambda ID: lscIDs.getProp(ID, prop_name))

		n = len(sorted_ID_list)
		train_size = np.floor(n * self.train_ratio).astype(int)
		test_size = np.floor(n * self. test_ratio).astype(int)
		val_size = n - train_size - test_size

		self._test_ID_ary = np.array(sorted_ID_list[:test_size])
		self._val_ID_ary = np.array(sorted_ID_list[test_size:-train_size])
		self._train_ID_ary = np.array(sorted_ID_list[-train_size:])

class BinHold(BaseHoldout):
	def __init__(self, bin_num, train_on='top', shuffle=False):
		"""

		Args:
			bin_num(int): num of bins for bin_num mode (see mode)

		"""
		super(BinHold, self).__init__(train_on=train_on, shuffle=shuffle)
		self.bin_num = bin_num

	def __repr__(self):
		return 'BinHold(bin_num=%s, train_on=%s)'%\
		(repr(self.bin_num), repr(self.train_on))

	@property
	def bin_num(self):
		return self._bin_num
	
	@bin_num.setter
	def bin_num(self, val):
		checkers.checkTypeErrNone('Number of bins', checkers.INT_TYPE, val)
		if val < 1:
			raise ValueError("Number of bins must be greater than 1, not '%d'"%val)
		self._bin_num = val

	def train_test_setup(self, lscIDs, nodeIDs, prop_name, **kwargs):
		"""

		Args:
			lscIDs(:obj:`NLEval.util.IDHandler.IDprop`)
			nodeIDs(:obj:`NLEval.util.IDHandler.IDmap`)
			prop_name(str): name of property to be used for splitting

		"""
		lscIDs._check_prop_existence(prop_name, True)
		common_ID_list = self.get_common_ID_list(lscIDs, nodeIDs)
		sorted_ID_list = sorted(common_ID_list, reverse=self.train_on=='bot', \
			key=lambda ID: lscIDs.getProp(ID, prop_name))
		bin_size = np.floor(len(sorted_ID_list) / self.bin_num).astype(int)
		self._test_ID_ary = np.array(sorted_ID_list[:bin_size])
		self._train_ID_ary = np.array(sorted_ID_list[-bin_size:])

class ThreshHold(BaseHoldout):
	def __init__(self, cut_off, train_on='top', shuffle=False):
		"""

		Args:
			cut_off:		cut-off point for cut mode, num of bins for bin mode (see mode)

		"""
		super(ThreshHold, self).__init__(train_on=train_on, shuffle=shuffle)
		self.cut_off = cut_off

	def __repr__(self):
		return 'ThreshHold(cut_off=%s, prop_name=%s, train_on=%s)'%\
		(repr(self.cut_off), repr(self.prop_name), repr(self.train_on))

	@property
	def cut_off(self):
		return self._cut_off
	
	@cut_off.setter
	def cut_off(self, val):
		checkers.checkTypeErrNone('Cut off', checkers.NUMERIC_TYPE, val)
		self._cut_off = val

	def train_test_setup(self, lscIDs, nodeIDs, prop_name, **kwargs):
		"""

		Args:
			lscIDs(:obj:`NLEval.util.IDHandler.IDprop`)
			nodeIDs(:obj:`NLEval.util.IDHandler.IDmap`)
			prop_name(str): name of property to be used for splitting

		"""
		lscIDs._check_prop_existence(prop_name, True)
		top_list = []
		bot_list = []
		for ID in nodeIDs.lst:
			if ID in lscIDs:
				if lscIDs.getProp(ID, 'Noccur') > 0:
					if lscIDs.getProp(ID, prop_name) >= self.cut_off:
						top_list.append(ID)
					else:
						bot_list.append(ID)

		if self.train_on == 'top':
			self._train_ID_ary, self._test_ID_ary = top_list, bot_list
		else:
			self._train_ID_ary, self._test_ID_ary = bot_list, top_list

class CustomHold(BaseHoldout):
	def __init__(self, custom_train_ID_ary, custom_test_ID_ary, shuffle=False):
		"""User defined training and testing samples"""
		super(CustomHold, self).__init__(shuffle=shuffle)
		self.custom_train_ID_ary = custom_train_ID_ary
		self.custom_test_ID_ary = custom_test_ID_ary

	def __repr__(self):
		return 'CustomHold(min_pos=%s)'%repr(self.min_pos)

	@property
	def custom_train_ID_ary(self):
		return self._custom_train_ID_ary

	@custom_train_ID_ary.setter
	def custom_train_ID_ary(self, ID_ary):
		checkers.checkTypesInNumpyArray("Training data ID list", str, ID_ary)
		self._custom_train_ID_ary = ID_ary

	@property
	def custom_test_ID_ary(self):
		return self._custom_test_ID_ary

	@custom_test_ID_ary.setter
	def custom_test_ID_ary(self, ID_ary):
		checkers.checkTypesInNumpyArray('Testing data ID list', str, ID_ary)
		self._custom_test_ID_ary = ID_ary

	def train_test_setup(self, lscIDs, nodeIDs, **kwargs):
		common_ID_list = self.get_common_ID_list(lscIDs, nodeIDs)
		self._train_ID_ary = np.intersect1d(self.custom_train_ID_ary, common_ID_list)
		self._test_ID_ary = np.intersect1d(self.custom_test_ID_ary, common_ID_list)

class TrainTestAll(BaseHoldout):
	def __init__(self, shuffle=False):
		"""Train and test on all data"""
		super(TrainTestAll, self).__init__(shuffle=shuffle)

	def train_test_setup(self, lscIDs, nodeIDs, **kwargs):
		common_ID_list = self.get_common_ID_list(lscIDs, nodeIDs)
		self._train_ID_ary = self._test_ID_ary = np.array(common_ID_list)

'''
class HoldoutChildTemplate(BaseHoldout):
	"""
	This is a template for BaseHoldout children class
	"""
	def __init__(self, **args, min_pos=10, **kwargs):
		super().__init__(min_pos=min_pos)
	
	def __repr__(self):
		return 'HoldoutChildTemplate(min_pos=%s, train_on=%s)'%repr(self.min_pse)

	@property
	def foo(self):
		return self._foo
	
	@foo.setter
	def foo(self, val):
		self._foo = val

	def train_test_setup(self, lscIDs, nodeIDs, prop_name, **kwargs):
		#setup train_ID_ary and test_ID_ary
'''
