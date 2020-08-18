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
