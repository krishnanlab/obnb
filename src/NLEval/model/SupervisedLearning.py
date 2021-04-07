import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from NLEval.model.BaseModel import BaseModel

__all__ = ['SLBase', 'LogReg', 'SVM', 'RF']

class SLBase(BaseModel):
	def train(self, ID_ary, y):
		x = self.G[ID_ary]
		self.fit(x, y)

	def decision(self, ID_ary):
		x = self.G[ID_ary]
		decision_ary = self.decision_function(x)
		return decision_ary

class LogReg(SLBase, LogisticRegression):
	def __init__(self, G, **kwargs):
		SLBase.__init__(self, G)
		LogisticRegression.__init__(self, **kwargs)

class LogRegCV(SLBase, LogisticRegressionCV):
	def __init__(self, G, **kwargs):
		SLBase.__init__(self, G)
		LogisticRegressionCV.__init__(self, **kwargs)

class CombLogRegCV(BaseModel):
	"""LogRegCV with multiple feature sets"""
	def __init__(self, G, **kwargs):
		# G here should be multi feature set
		BaseModel.__init__(self, G)
		self.mdl_list = [LogisticRegressionCV(**kwargs) for i in self.G.mat_list]
		self.master_mdl = LogisticRegressionCV(**kwargs)

	def train(self, ID_ary, y):
		x_master = np.zeros((len(ID_ary), len(self.G.mat_list)))
		for i, mat in enumerate(self.G.mat_list):
			x = mat[self.G.IDmap[ID_ary]]
			self.mdl_list[i].fit(x,y)
			x_master[:,i] = self.mdl_list[i].decision_function(x)
		self.master_mdl.fit(x_master, y)
		#print(self.master_mdl.coef_)
		#for i, j in zip(self.G.name_list, self.master_mdl.coef_[0]):
				#print(i,j)
		#print('')

	def decision(self, ID_ary):
		x_master = np.zeros((len(ID_ary), len(self.G.mat_list)))
		for i, mat in enumerate(self.G.mat_list):
			x = mat[self.G.IDmap[ID_ary]]
			x_master[:,i] = self.mdl_list[i].decision_function(x)
		decision_ary = self.master_mdl.decision_function(x_master)
		return decision_ary

class SVM(SLBase, LinearSVC):
	def __init__(self, G, **kwargs):
		SLBase.__init__(self, G)
		LinearSVC.__init__(self, **kwargs)

class RF(SLBase, RandomForestClassifier):
	def __init__(self, G, **kwargs):
		SLBase.__init__(self, G)
		RandomForestClassifier.__init__(self, **kwargs)

	def decision(self, ID_ary):
		x = self.get_x(ID_ary)
		decision_ary = self.predict_proba(x)[:,1] # take positive class
		return decision_ary
