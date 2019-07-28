import numpy as np
from NLEval.util import checkers
from copy import deepcopy

class IDlst(object):
	"""docstring for IDlst"""
	def __init__(self):
		super(IDlst, self).__init__()
		self._lst = []

	def __iter__(self):
		"""Yield all IDs"""
		return self.lst.__iter__()

	def __eq__(self, other):
		"""Return true if two IDlst have same set of IDs"""
		checkers.checkType('other', self.__class__, other)
		return set(self.lst) == set(other.lst)

	def __add__(self, other):
		checkers.checkType('other', self.__class__, other)
		new = self.copy()
		for ID in other:
			if ID not in new:
				new.addID(ID)
		return new

	def __sub__(self, other):
		checkers.checkType('other', self.__class__, other)
		new = self.__class__()
		for ID in self:
			if ID not in other:
				new.addID(ID)
		return new

	def __contains__(self, ID):
		"""Return true if ID exist in current list"""
		checkers.checkType('ID', str, ID)
		return ID in self.lst

	def __getitem__(self, ID):
		"""Return (array of) index of ID"""
		if isinstance(ID, str):
			return self.__getitem_sinlge(ID)
		elif isinstance(ID, checkers.ITERABLE_TYPE):
			return self.__getitem_multiple(ID)
		else:
			raise TypeError("ID keys must be stirng or iterables of string, not %s"%\
					repr(type(ID)))

	def __getitem_sinlge(self, ID):
		assert ID in self, "Unknown ID: %s"%repr(ID)
		return self._lst.index(ID)

	def __getitem_multiple(self, IDs):
		checkers.checkTypesInIterable('IDs', str, IDs)
		idx_lst = []
		for ID in IDs:
			idx_lst.append(self.__getitem_sinlge(ID))
		return np.array(idx_lst)

	@property
	def lst(self):
		""":obj:`list` of :obj:`str`: list of IDs.
		No setter, use `addID` or `popID` to modify"""
		return self._lst
	
	@property
	def size(self):
		"""int: number of IDs in list"""
		return len(self.lst)

	def copy(self):
		"""Return a deepcopy of self"""
		return deepcopy(self)

	def isempty(self):
		"""Return true if ID list is empty"""
		return self.size == 0

	def popID(self, ID):
		"""Pop an ID out of list of IDs"""
		checkers.checkType('ID', str, ID)
		assert ID in self, "Unknown ID: %s"%repr(ID)
		idx = self[ID]
		self.lst.pop(idx)
		return idx

	def addID(self, ID):
		"""Add new ID as string, append last"""
		checkers.checkType('ID', checkers.NUMSTRING_TYPE, ID)
		try:
			num = float(ID)
			#convert to int string if numeric and is int
			ID = str(int(num)) if num % 1 == 0 else str(num)
		except ValueError:
			ID = ID.strip()
		#check if ID already exist after clean up format
		assert ID not in self, "ID %s exists"%repr(ID)
		self._lst.append(ID)

	def getID(self, idx):
		"""Return ID by its index"""
		return self.lst[idx]

	@classmethod
	def from_list(cls, lst):
		checkers.checkType("ID list", list, lst)
		obj = cls()
		for ID in lst:
			obj.addID(ID)
		return obj

class IDmap(IDlst):
	"""IDmap object that implements dictionary for more efficient mapping
	from ID to index"""
	def __init__(self):
		super(IDmap, self).__init__()
		self._map = {}

	def __contains__(self, ID):
		return ID in self.map

	def __getitem_sinlge(self, ID):
		assert ID in self, "Unknown ID: %s"%repr(ID)
		return self.map[ID]

	@property
	def map(self):
		"""(dict of str:int): map from ID to index"""
		return self._map

	def popID(self, ID):
		super(IDmap, self).popID(ID)
		idx = self.map.pop(ID)
		for i, ID in enumerate(self.lst[idx:]):
			self.map[ID] = idx + i
		return idx
	
	def addID(self, ID):
		new_idx = self.size
		super(IDmap, self).addID(ID)
		self._map[self.lst[-1]] = new_idx

class IDprop(IDmap):
	"""ID properties object that stores property information of IDs"""
	def __init__(self):
		super(IDprop, self).__init__()
		self._prop_default_val = {}
		self._prop_default_type = {}
		self._prop = {}

	def __eq__(self, other):
		"""Return true if two object have same set of IDs with same properties"""
		#check if two objects have same set of IDs
		if not super(IDprop, self).__eq__(other):
			return False
		#check if two objects have same set of properties
		if not set(self.propLst) == set(other.propLst):
			return False
		#check if properties have same values
		for prop in self.propLst:
			for ID in self.lst:
				if self.getProp(ID, prop) != other.getProp(ID, prop):
					return False
		return True

	def __add__(self, other):
		raise NotImplementedError

	def __sub__(self, other):
		raise NotImplementedError

	@property
	def prop_default_val(self):
		"""(dict of str:obj): dictionary mapping from property name to 
		default property value"""
		return self._prop_default_val.copy()
	
	@property
	def prop_default_type(self):
		return self._prop_default_type.copy()

	@property
	def prop(self):
		"""(dict of str: :obj:`list` of :obj:): dictionary mapping from 
		property name to list of property values in the order of ID list

		Note: the returned list is a copy of self._prop to prevent userside
		maniputation on data, use self.setProp to modify properties

		"""
		return self._prop.copy()

	@property
	def propLst(self):
		""":obj:`list` of :obj:`str`: list of properties names"""
		return list(self._prop)

	def newProp(self, prop_name, default_val=None, default_type=None):
		"""Create a new property
		
		Args:
			prop_name(str): name of property
			default_val: default value to set if property not specified

		"""
		checkers.checkType("Property name", str, prop_name)
		assert prop_name not in self.propLst, "Property %s exists"%prop_name
		if default_type is not None:
			checkers.checkType("Default type", type, default_type)
			if not isinstance(default_val, default_type):
				raise TypeError("Inconsistent type between default values %s and default type %s"%\
					(type(default_val), default_type))
		if not self.isempty():
			prop_lst = [deepcopy(default_val) for i in range(self.size)]
		else:
			prop_lst = []
		self._prop_default_val[prop_name] = default_val
		self._prop_default_type[prop_name] = default_type
		self._prop[prop_name] = prop_lst

	def setProp(self, ID, prop_name, prop_val):
		"""Set a pericif property value of an ID, must match default type if available"""
		self.getProp(ID, prop_name) #check ID and prop_name validity
		if self.prop_default_type[prop_name] is not None:
			checkers.checkType("Property value for %s"%repr(prop_name), \
				self.prop_default_type[prop_name], prop_val)
		self._prop[prop_name][self[ID]] = prop_val

	def getProp(self, ID, prop_name):
		"""Return a specific properties associated with an ID"""
		checkers.checkType('ID', str, ID)
		assert ID in self, "Unknown ID: %s"%repr(ID)
		checkers.checkType('Property name', str, prop_name)
		assert prop_name in self.propLst, "Unknown property name: %s"%repr(prop_name)
		return self._prop[prop_name][self[ID]]

	def getAllProp(self, ID):
		"""Return all properties associated with an ID"""
		return {prop:self.getProp(ID, prop) for prop in self.propLst}

	def popID(self, ID):
		idx = super(IDprop, self).popID(ID)
		for prop in self.propLst:
			self._prop[prop].pop(idx)
		return idx

	def addID(self, ID, prop=None):
		if prop is not None:
			checkers.checkType("Properties", dict, prop)
			checkers.checkTypesInIterable("Properties Keys", str, prop)
			assert set(prop) == set(self.propLst), \
				"Input properties must be in %s, not %s"%\
				(set(self.propLst), set(prop))
			#chekc type of prop val
			for prop_name, default_type in self.prop_default_type.items():
				if default_type is not None:
					checkers.checkType("Properties Values", default_type, prop[prop_name])
		else:
			prop = self.prop_default_val
		super(IDprop, self).addID(ID)
		for prop_name, prop_val in prop.items():
			self._prop[prop_name].append(prop_val)
