from NLEval.util.Exceptions import IDExistsError, IDNotExistError
from NLEval.util import checkers, IDHandler
from NLEval.label import Filter
from NLEval.valsplit import Base
import numpy as np

class BaseLSC(IDHandler.IDprop):
	"""Collection of labelsets

	This class is used for managing collection of labelsets.


	Example GMT (Gene Matrix Transpose):

	'''
	Geneset1	Description1	Gene1	Gene2	Gene3
	Geneset2	Description2	Gene2	Gene4	Gene5	Gene6
	'''

	Example internal data for a label collection with above GMT data:

	self.entityIDlst = ['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5', 'Gene6']
	self.entity.prop = {'Noccur': [1, 2, 1, 1, 1, 1]}
	self.labelIDlst = ['Geneset1', 'Geneset2']
	self.prop = {
		'Info':['Description1', 'Description2']
		'Labelset':[
			{'Gene1', 'Gene2', 'Gene3'},
			{'Gene2', 'Gene4', 'Gene5', 'Gene6'}
		]
	}

	"""
	def __init__(self):
		super(BaseLSC, self).__init__()
		self.entity = IDHandler.IDprop()
		self.entity.newProp('Noccur', 0, int)
		self.newProp('Info', 'NA', str)
		self.newProp('Labelset', set(), set)
		self.newProp('Negative', set(), set)

	def _show(self):
		"""Debugging prints"""
		print("Labelsets IDs:"); print(self._lst)
		print("Labelsets Info:"); print(self._prop['Info'])
		print("Labelsets:")
		for lbset in self._prop['Labelset']:
			print(lbset)
		print("Entities IDs:"); print(self.entity._lst)
		print("Entities occurances:"); print(self.entity._prop)

	@property
	def entityIDlst(self):
		""":obj:`list` of :obj:`str`: list of all entity IDs"""
		return self.entity.lst

	@property
	def labelIDlst(self):
		""":obj:`list` of :obj:`str`: list of all labelset names"""
		return self.lst

	def addLabelset(self, lst, labelID, labelInfo=None):
		"""Add new labelset

		Args:
			lst(:obj:`list` of :obj:`str`): list of IDs of entiteis belong
				to the input label
			labelID(str): name of label
			labelInfo(str): description of label

		"""
		self.addID(labelID, {} if labelInfo is None else {'Info':labelInfo})
		try:
			self.entity.update(lst)
		except Exception as e:
			#if entity list not updated successfully, pop the new labelset
			self.popID(labelID)
			raise e
		self.updateLabelset(lst, labelID)

	def popLabelset(self, labelID):
		"""Pop a labelset and remove entities that no longer belong to any labelset"""
		self.resetLabelset(labelID)
		self.popID(labelID)

	def updateLabelset(self, lst, labelID):
		"""Update existing labelset

		Take list of entities IDs and update current labelset with a label
		name matching `labelID`. Any ID in the input list `lst` that does
		not exist in the entity list will be added to the entity list. Increment
		the `Noccur` property of any newly added entites to the labelset by 1.

		Note: labelID must already existed, use `.addLabelset()` for adding
		new labelset

		Args:
			lst(:obj:`list` of :obj:`str`): list of entiteis IDs to be 
				added to the labelset, can be redundant.

		Raises:
			TypeError: if `lst` is not `list` type or any element within `lst`
				is not `str` type

		"""
		checkers.checkTypesInList("Entity list", str, lst)
		lbset = self.getLabelset(labelID)
		for ID in lst:
			if ID not in self.entity:
				self.entity.addID(ID)
			if ID not in lbset:
				lbset.update([ID])
				self.entity.setProp(ID, 'Noccur', self.getNoccur(ID) + 1)

	def resetLabelset(self, labelID):
		"""Reset an existing labelset to an empty set
		Decrement `Noccur` property of all entites belonging to the labelset,
		any entity specific to the popped labelset (`Noccur` == 1) is popped.
		"""
		lbset = self.getLabelset(labelID)
		for ID in lbset:
			self.entity.setProp(ID, 'Noccur', self.getNoccur(ID) - 1)
			if self.entity.getAllProp(ID) == self.entity.prop_default_val:
				self.entity.popID(ID)
		lbset.clear()

	def popEntity(self, ID):
		"""Pop an entity, remove from all labelsets"""
		self.entity.popID(ID)
		for labelID in self.labelIDlst:
			lbset = self.getLabelset(labelID).difference_update([ID])

	def getInfo(self, labelID):
		"""Return description of a labelset"""
		return self.getProp(labelID, 'Info')

	def getLabelset(self, labelID):
		"""Return set of entities associated with a label"""
		return self.getProp(labelID, 'Labelset')

	def getNegative(self, labelID):
		"""Return set of negative samples of a labelset
		
		Note:
			If negative samples not available, use complement of labelset

		"""
		neg = self.getProp(labelID, 'Negative')
		return neg if neg else set(self.entity.map) - self.getLabelset(labelID)

	def setNegative(self, lst, labelID):
		checkers.checkTypesInList("Negative entity list", str, lst)
		lbset = self.getLabelset(labelID)
		for ID in lst:
			self.entity._check_ID_existence(ID, True)
			if ID in lbset:
				#raise Exception(repr(ID), repr(labelID))
				raise IDExistsError("Entity %s is positive in labelset %s, "%\
				 	(repr(ID), repr(labelID)) + "cannot be set to negative")
		self.setProp(labelID, 'Negative', set(lst))

	def getNoccur(self, ID):
		"""Return the number of labelsets in which an entity participates"""
		return self.entity.getProp(ID, 'Noccur')

	def apply(self, filter_func):
		"""Apply filter"""
		checkers.checkType("Filter", Filter.BaseFilter, filter_func)
		filter_func(self)

	def export(self, fp):
		pass

	def load_entity_properties(self, fp, prop_name, \
			default_val, default_type, interpreter=int):
		"""Load entity properties from file.
		The file is tab seprated with two columns, first column 
		contains entities IDs, second column contains corresponding 
		properties of entities.

		Args:
			fp(str): path to the entity properties file.
			default_val: default value of property of an entity 
				if not specified.
			default_type(type): default type of the property.
			interpreter: function to transfrom property value from
				string to some other value

		"""
		self.entity.newProp(prop_name, default_val, default_type)
		with open(fp, 'r') as f:
			for line in f:
				ID, val = line.strip().split()
				if ID not in self.entity:
					self.entity.addID(ID)
				self.entity.setProp(ID, prop_name, interpreter(val))

	@classmethod
	def from_gmt(cls, fp):
		"""Load data from Gene Matrix Transpose `.gmt` file
		https://software.broadinstitute.org/cancer/software/gsea/wiki/index.php/Data_formats

		Args:
			fg(str): path to the `.gmt` file

		"""
		lsc = cls()
		with open(fp, 'r') as f:
			for line in f:
				labelID, labelInfo, *lst = line.strip().split('\t')
				lsc.addLabelset(lst, labelID, labelInfo)
		return lsc
