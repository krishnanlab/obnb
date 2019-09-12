class BaseFilter:
	"""Base Filter object containing basic filter operations

	Notes:
		Loop through all instances (IDs) retrieved by `self.get_IDs` and decide
		whether or not to apply modification using `self.criterion`, and finally
		apply modification if passes criterion using `mod_fun`.

	Basic components (methods) needed for children filter classes:
		criterion: retrun true if the corresponding value of an instance passes
			the criterion
		get_IDs: return list of IDs to scan through
		get_val_getter: return a function that map ID of an instance to some 
			corresponding values
		get_mod_fun: return a function that modifies an instance

	All three 'get' methods above take a `LabelsetCollection` object as input

	"""
	def __init__(self):
		super(BaseFilter, self).__init__()

	def __call__(self, lsc):
		IDs = self.get_IDs(lsc)
		val_getter = self.get_val_getter(lsc)
		mod_fun = self.get_mod_fun(lsc)

		for ID in IDs:
			if self.criterion(val_getter(ID)):
				mod_fun(ID)

class RangeFilter(BaseFilter):
	"""Filter entities in labelset collection by range of values

	Notes:
		If `None` specified for `min_val` or `max_val`, no filtering
		will be done on upper/lower bound.

	Attributes:
		min_val: minimum below which entities are removed
		max_val: maximum beyound which entiteis are removed

	"""

	def __init__(self, min_val=None, max_val=None):
		super(RangeFilter, self).__init__()
		self.min_val = min_val
		self.max_val = max_val

	def criterion(self, val):
		if self.min_val is not None:
			if val < self.min_val:
				return True
		if self.max_val is not None:
			if val > self.max_val:
				return True
		return False
	
class EntityRangeFilterNoccur(RangeFilter):
	"""Pop entities based on number of occurance"""
	def __init__(self, min_val=None, max_val=None):
		super(EntityRangeFilterNoccur, self).__init__(min_val, max_val)

	@staticmethod
	def get_val_getter(lsc):
		return lsc.getNoccur

	@staticmethod
	def get_IDs(lsc):
		return lsc.entityIDlst

	@staticmethod
	def get_mod_fun(lsc):
		return lsc.popEntity

class LabelsetRangeFilterSize(RangeFilter):
	"""Pop labelsets based on size"""
	def __init__(self, min_val=None, max_val=None):
		super(LabelsetRangeFilterSize, self).__init__(min_val, max_val)

	@staticmethod
	def get_val_getter(lsc):
		return lambda ID: len(lsc.getLabelset(ID))

	@staticmethod
	def get_IDs(lsc):
		return lsc.labelIDlst

	@staticmethod
	def get_mod_fun(lsc):
		return lsc.popLabelset

class LabelsetRangeFilterTrainTestPos(RangeFilter):
	"""Pop labelsets based on number of positives in train/test sets"""
	def __init__(self, min_val):
		super(LabelsetRangeFilterTrainTestPos, self).__init__(min_val=min_val)

	@staticmethod
	def get_val_getter(lsc):
		return lambda labelID: min([min(tr.sum(), ts.sum()) for \
			_,tr,_,ts in lsc.splitLabelset(labelID)])

	@staticmethod
	def get_IDs(lsc):
		return lsc.labelIDlst

	@staticmethod
	def get_mod_fun(lsc):
		return lsc.popLabelset #replace with soft filter

class ValueFilter(BaseFilter):
	"""Filter based on certain values

	Attributes:
		val: target value
		remove(bool): if true, remove any ID with matched value,
			else remove any ID with mismatched value

	"""
	def __init__(self, val, remove=True):
		super(RangeFilter, self).__init__()
		self.val = val
		self.remove = remove

	def criterion(self, val):
		return True if (val == self.val) is self.remove else False
