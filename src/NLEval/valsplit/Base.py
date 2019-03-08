from ..util import checkers
import numpy as np

class BaseValSplit:
	class Wrapper:
		def __init__(self, split_gen, min_pos):
			self.split_gen = split_gen
			self.min_pos = min_pos

		def filter_pos_below_min(self, pad, label):
			num_pos_test = label.sum()
			num_pos_train = label.sum()
			for train, test in self.split_gen(pad, label):
				if (num_pos_train < self.min_pos) | (num_pos_test < self.min_pos):
					yield None, None
				else:
					yield train, test

	def __init__(self, min_pos=10):
		'''
		Input:
			- min_pos:		minimum number of positives in both testing and training
		'''
		self.min_pos = min_pos

	@property
	def min_pos(self):
		return self._min_pos
	
	@min_pos.setter
	def min_pos(self, val):
		checkers.checkTypeAllowNone('min_pos', int, val)
		if val is None:
			val = 0
		self._min_pos = val
	
	def __repr__(self):
		return 'ValSplit(min_pos=%s)'%repr(self.min_pos)

	def train_test_setup(self, pos_ID, node_property):
		pass

	def wrap(self, split_gen):
		wrapper = self.Wrapper(split_gen, self.min_pos)
		return wrapper.filter_pos_below_min

	def get_split_gen(self):
		assert False,"This is base validation split object, no split generators are defined"

class BaseHoldout(BaseValSplit):
	def __init__(self, split_criterion, prop_name, min_pos=10, train_on='top'):
		'''
		Generic holdout validation object
			- min_pos:			minimum number of positives in both testing and training
			- split_criterion:	splitting criterion, type int or float
			- prop_name:		name of node property for guiding holdout
		'''
		super().__init__(min_pos=min_pos)
		self.min_pos = min_pos
		self.split_criterion = split_criterion
		self.prop_name = prop_name
		self.train_on = train_on
		self._test_idx_ary = None
		self._train_idx_ary = None

	def __repr__(self):
		return 'HoldoutSplit(split_criterion=%s, prop_name=%s, \
		min_pos=%s, train_on=%s)'%(repr(self.split_criterion), \
		repr(self.prop_name), repr(self.min_pos), repr(self.train_on))

	@property
	def test_idx_ary(self):
		return self._test_idx_ary.copy()
	
	@property
	def train_idx_ary(self):
		return self._train_idx_ary.copy()

	def make_train_test_ary(self, pos_ID, train_IDlst, test_IDlst):
		self._test_idx_ary = np.array([idx for idx,ID in enumerate(pos_ID) if ID in test_IDlst], dtype=int)
		self._train_idx_ary = np.array([idx for idx,ID in enumerate(pos_ID) if ID in train_IDlst], dtype=int)

	@property
	def split_criterion(self):
		return self._cut_off
	
	@split_criterion.setter
	def split_criterion(self, val):
		checkers.checkTypeErrNone('split_criterion', (float,int), val)
		self._cut_off = val

	@property
	def prop_name(self):
		return self._prop_name
	
	@prop_name.setter
	def prop_name(self, val):
		checkers.checkTypeErrNone('prop_name', str, val)
		self._prop_name = val

	@property
	def train_on(self):
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
	
	def holdout_split(self, pad, label):
		'''
		Input:
			- pad:		for compatibility with StratifiedKFold, 
						zeros with size n (number of total sample)
			- label:	boolean/binary array of indicator of positives, size n
		'''
		assert (self._test_idx_ary is not None) & (self._train_idx_ary is not None),\
		 "Training and testing sets not available, run train_test_setup first"
		num_pos_test = label[self._test_idx_ary].sum()
		num_pos_train = label[self._train_idx_ary].sum()
		yield self.train_idx_ary, self.test_idx_ary

	def train_test_setup(self, **kwargs):
		assert False,"This is base holdout split object, train_test_setup is not defined"

	def get_split_gen(self):
		return self.wrap(self.holdout_split)

class BaseInterface(BaseValSplit):
	def __init__(self, obj, min_pos=10):
		'''
		Interface for other split generating objects
		'''
		super().__init__(min_pos=min_pos)
		self.model = obj

	def __repr__(self):
		return "BaseInterface(%s, min_pos=%s)"%(repr(obj), repr(min_pos))

	def get_split_gen(self):
		return self.wrap(self.model.split)


