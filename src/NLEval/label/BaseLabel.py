from NLEval.util.IDmap import IDmap
from NLEval.util import checkers
import numpy as np

class BaseLabel:
	def __init__(self, lb_file_pth=None, drop_threshold=None, spec_threshold=None, reader='gmt'):
		self.lb_file_pth = lb_file_pth
		self.drop_threshold = drop_threshold
		self.spec_threshold = spec_threshold
		self.reader = reader

		self._class_IDlst = []
		self._classname = []
		self._classinfo = []

		self.IDmap = IDmap()

		if lb_file_pth is not None:
			self.construct()

	def __getitem__(self, ID):
		return self.mat[self.IDmap[ID]]

	def __repr__(self):
		return "BaseLabel(lb_file_pth=%s, drop_threshold=%s, \n\
		spec_threshold=%s, reader=%s)"%\
		(repr(self.lb_file_pth), repr(self.drop_threshold), \
		repr(self.spec_threshold), repr(self.reader))

	@property
	def size(self):
		return len(self.classname)
	
	@property
	def class_IDlst(self):
		return self._class_IDlst
	
	@property
	def classname(self):
		return self._classname

	@property
	def classinfo(self):
		return self._classinfo

	@property
	def lb_file_pth(self):
		return self._lb_file_pth
	
	@lb_file_pth.setter
	def lb_file_pth(self, val):
		checkers.checkTypeAllowNone('lb_file_pth', str, val)
		self._lb_file_pth = val

	@property
	def drop_threshold(self):
		return self._drop_threshold
	
	@drop_threshold.setter
	def drop_threshold(self, val):
		checkers.checkTypeAllowNone('drop_threshold', int, val)
		self._drop_threshold = val

	@property
	def spec_threshold(self):
		return self._spec_threshold
	

	@spec_threshold.setter
	def spec_threshold(self, val):
		checkers.checkTypeAllowNone('spec_threshold', int, val)
		self._spec_threshold = val

	def add_set(self, classname, classinfo, IDlst):
		self._classname.append(classname)
		self._classinfo.append(classinfo)
		self._class_IDlst.append(IDlst)

	def remove_set(self):
		pass

	@staticmethod
	def gmt_reader(fp):
		'''
		Read from gmt files (tab separated): 
			first item -> name
			second item -> info
			rest -> ID set
		'''
		with open(fp, 'r') as lbfile:
			for line in lbfile:
				raw_lb_lst = line.rstrip('\n').split('\t')
				classname = raw_lb_lst[0].strip()
				classinfo = raw_lb_lst[1].strip()
				#make sure no redundant entities within each set
				IDlst = list({ID.strip() for ID in raw_lb_lst[2:]})
				yield classname, classinfo, IDlst

	def read(self):
		'''
		Construct label sets from file
		Input:
			- lb_file_pth:	(str) path to file
			- reader:		user defined generator function that yield classname, classinfo, IDlst
							default is gmt reader
		'''
		checkers.checkTypeErrNone('lb_file_pth', str, self.lb_file_pth)
		print('Loading %s...'%self.lb_file_pth.split('/')[-1])
		if self.reader == 'gmt':
			reader = self.gmt_reader
		else:
			reader = self.reader

		for classname, classinfo, IDlst in reader(self.lb_file_pth):
			self.add_set(classname, classinfo, IDlst)

	def drop_entity_lst(self):
		'''
		Count the number of times an entity apprear in sets
		Return a list of entities that exceed spec_threshold
		'''
		count = {}
		for IDlst in self._class_IDlst:
			for ID in IDlst:
				if ID not in count:
					count[ID] = 1
				else:
					count[ID] += 1
		pop_lst = [ID for ID in count if count[ID] >= self.spec_threshold]
		return pop_lst

	def drop_entity(self):
		'''
		Drop entities that are commonly appear acrose sets to increase specificity
		'''
		if self.spec_threshold is not None:
			pop_lst = self.drop_entity_lst()
			for ID in pop_lst:
				for IDlst in self._class_IDlst:
					try:
						IDlst.remove(ID)
					except ValueError:
						continue
			print('%d entities dropped with specificity threshold = %d'%\
				(len(pop_lst), self.spec_threshold))

	def drop_class(self):
		'''
		Drop classes with # of positives less than or higher thana spcific threshold value
		'''
		if self.drop_threshold is not None:
			idx = -1
			num_poped = 0
			while True:
				try:
					if len(self._class_IDlst[idx]) < self.drop_threshold:
						self._class_IDlst.pop(idx)
						self._classname.pop(idx)
						self._classinfo.pop(idx)
						num_poped += 1
					else:
						idx -= 1
				except IndexError:
					break
			print('%d classes dropped with drop threshold = %d'%(num_poped, self.drop_threshold))

	def get_IDmap(self):
		IDset = set()
		for IDlst in self.class_IDlst:
			IDset.update(IDlst)
		for ID in IDset:
			self.IDmap.addID(ID)

	def get_mat(self):
		NIDs = self.IDmap.size
		Nclass = len(self._classname)
		mat = np.zeros((NIDs,Nclass), dtype=bool)
		for class_idx in range(Nclass):
			for id in self._class_IDlst[class_idx]:
				mat[self.IDmap[id], class_idx] = 1
		self.mat = mat

	def construct(self):
		self.reset()
		self.read()
		self.drop_entity()
		self.drop_class()
		self.get_IDmap()
		self.get_mat()

	def reset(self):
		self._class_IDlst = []
		self._classname = []
		self._classinfo = []


