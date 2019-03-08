from NLEval.util import checkers
from NLEval.valsplit.Base import *
import numpy as np

class BinHold(BaseHoldout):
	def __init__(self, bin_num, prop_name, min_pos=10, train_on='top'):
		'''
		Input:
			- bin_num:		num of bins for bin_num mode (see mode)
			- prop_name:	name of property for guiding holdout
			- min_pos:		minimum number of positives in both testing and training
			- train_on
				- 'top':	train on top, test on bottom
				- 'bot':	train on bottom, test on top
		'''
		super().__init__(bin_num, prop_name, min_pos=min_pos, train_on=train_on)
		self.bin_num = bin_num

	def __repr__(self):
		return 'BinHold(bin_num=%s, prop_name=%s, min_pos=%s, train_on=%s)'%\
		(repr(self.bin_num), repr(self.prop_name), repr(self.min_pse), repr(self.train_on))

	@property
	def bin_num(self):
		return self._bin_num
	
	@bin_num.setter
	def bin_num(self, val):
		checkers.checkTypeErrNone('bin_num', int, val)
		assert val > 1,"Number of bins must be greater than 1, not '%d'"%val
		#TODO:max num of bins??
		self._bin_num = val
		self.split_criterion = val

	def train_test_setup(self, pos_ID, node_property, **kwargs):
		'''
		Input:
		-   pos_ID: union of entities in a labelset
		-   node_property:  EmdEval.node_property, dictionary of dictionaries of node property
				{ property_name: { entity_id: property_value } }
		'''
		assert self.prop_name in node_property, 'Unknown property %s'%self.prop_name
		intersection = set(pos_ID) & set(node_property[self.prop_name])
		prop = {ID:node_property[self.prop_name][ID] for ID in intersection}

		sorted_id_lst = sorted(prop, key=prop.get, reverse=self.reverse)
		bin_size = len(sorted_id_lst) // self.bin_num

		test_IDlst = list(set(sorted_id_lst[:bin_size]))
		train_IDlst = list(set(sorted_id_lst[(bin_size * (self.bin_num - 1)):]))

		self.make_train_test_ary(pos_ID, train_IDlst, test_IDlst)

class ThreshHold(BaseHoldout):
	def __init__(self, cut_off, prop_name, min_pos=10, train_on='top'):
		'''
		Input:
			- cut_off:		cut-off point for cut mode, num of bins for bin mode (see mode)
			- prop_name:	name of node property for guiding holdout
			- min_pos:		minimum number of positives in both testing and training
			- train_on
				- 'top':	train on top, test on bottom
				- 'bot':	train on bottom, test on top
		'''
		super().__init__(cut_off, prop_name, min_pos=min_pos, train_on=train_on)

	def __repr__(self):
		return 'ThreshHold(cut_off=%s, prop_name=%s, min_pos=%s, train_on=%s)'%\
		(repr(self.cut_off), repr(self.prop_name), repr(self.min_pos), repr(self.train_on))

	def train_test_setup(self, pos_ID, node_property, **kwargs):
		'''
		Input:
		-   pos_ID: union of entities in a labelset
		-   node_property:  EmdEval.node_property, dictionary of dictionaries of node property
				{ property_name: { entity_id: property_value } }
		'''
		assert self.holdout in node_property, 'Unknown property %s'%self.holdout
		intersection = set(pos_ID) & set(node_property[self.holdout])
		prop = {ID:node_property[self.holdout][ID] for ID in intersection}

		top_lst = []
		bot_lst = []
		for ID, prop_val in prop.items():
			if prop_val >= self.split:
				top_lst.append( ID )
			else:
				bot_lst.append( ID )
		if self.reverse:
			self.make_train_test_ary(pos_ID, top_lst, bot_lst)
		else:
			self.make_train_test_ary(pos_ID, bot_lst, top_lst)

class CustomHold(BaseHoldout):
	def __init__(self, train_IDlst=[], test_IDlst=[], min_pos=10):
		'''
		User defined training and testing
		'''
		super().__init__(0, 'NA', min_pos=min_pos)
		self.train_IDlst = train_IDlst
		self.test_IDlst = test_IDlst

	def __repr__(self):
		return 'CustomHold(min_pos=%s)'%repr(self.min_pos)

	@property
	def train_IDlst(self):
		return self._train_IDlst

	@property
	def test_IDlst(self):
		return self._test_IDlst

	@train_IDlst.setter
	def train_IDlst(self, IDlst):
		checkers.checkTypeErrNone('train_IDlst', list, IDlst)
		for ID in IDlst:
			self.addTrainID(ID)

	@test_IDlst.setter
	def test_IDlst(self, IDlst):
		checkers.checkTypeErrNone('test_IDlst', list, IDlst)
		for ID in IDlst:
			self.addTestID(ID)

	def addTrainID(self, ID):
		checkers.checkTypeErrNone('Training ID', str, IDlst)
		self._train_IDlst.append(ID)

	def addTestID(self, ID):
		checkers.checkTypeErrNone('Testing ID', str, IDlst)
		self._test_IDlst.append(ID)

	def train_test_setup(self, pos_ID, **kwargs):
		self.make_train_test_ary(pos_ID, self.train_IDlst, self.test_IDlst)

class TrainTestAll(BaseHoldout):
	def __init__(self, min_pos=10):
		'''
		Train and test on all
		'''
		super().__init__(0, 'NA', min_pos=min_pos)

	def __repr__(self):
		return 'TrainTestAll(min_pos=%s)'%repr(self.min_pos)

	def train_test_setup(self, pos_ID, **kwargs):
		self.make_train_test_ary(pos_ID, pos_ID, pos_ID)

"""
class HoldoutChildTemplate(BaseHoldout):
	'''
	This is a template for BaseHoldout children class
	'''
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

	def train_test_setup(self, pos_ID, **kwargs):
		#setup train_IDlst and test_IDlst and 
		pass
"""
