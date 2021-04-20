import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

class CombSLBase(BaseModel):
	def __init__(self, G, **kwargs):
		# G here should be multi feature set
		BaseModel.__init__(self, G)
		self.mdl_list = [self.base_mdl(**kwargs) for i in self.G.mat_list]

	def train(self, ID_ary, y):
		for i, mat in enumerate(self.G.mat_list):
			x = mat[self.G.IDmap[ID_ary]]
			self.mdl_list[i].fit(x, y)
		self.fit_master_mdl(ID_ary, y)

class CombLogRegCVBagging(CombSLBase):
	def __init__(self, G, **kwargs):
		self.base_mdl = LogisticRegressionCV
		CombSLBase.__init__(self, G, **kwargs)

	def fit_master_mdl(self, ID_ary, y):
		pass

	def decision(self, ID_ary):
		decision_ary = np.zeros((len(ID_ary)))
		for i, mat in enumerate(self.G.mat_list):
			x = mat[self.G.IDmap[ID_ary]]
			decision_ary += self.mdl_list[i].decision_function(x)
		decision_ary /= len(self.mdl_list)
		return decision_ary

class CombLogRegCVAdaBoost(CombSLBase):
	def __init__(self, G, **kwargs):
		self.base_mdl = LogisticRegressionCV
		CombSLBase.__init__(self, G, **kwargs)
		self.coef_ = None

	def fit_master_mdl(self, ID_ary, y):
		n_mdl = len(self.mdl_list)  # total number of models
		selected_ind = np.zeros(n_mdl, dtype=bool)  # inidvator for selected model
		w = np.ones(len(ID_ary)) / len(ID_ary)  # data point weights
		coef = np.zeros(n_mdl)  # model boosting coefficients
		y_pred_mat = np.zeros((len(ID_ary), n_mdl), dtype=bool)  # predictions from all models

		# generate predictions from individual models
		idx_ary = self.G.IDmap[ID_ary]
		for i, mdl in enumerate(self.mdl_list):
			x = self.G.mat_list[i][idx_ary]
			y_pred_mat[:,i] = mdl.predict(x)

		# determine boosting coefficients
		for i in range(n_mdl):
			opt_err = np.inf
			opt_idx = None

			for j in range(n_mdl):
				if selected_ind[j]:
					continue

				# retrain model using sample weights
				mdl = self.mdl_list[j]
				x = self.G.mat_list[j][idx_ary]
				mdl.fit(x, y, sample_weight=w)
				y_pred_mat[:,j] = mdl.predict(x)

				err = w[y_pred_mat[:,j] != y].sum()
				if err < opt_err:
					opt_err = err
					opt_idx = j

			a = 0.5 * np.log((1 - opt_err) / opt_err)  # model coefficient
			if a < 0: 
				print(f"Warning: encountered worse than random prediction, a = {a}, set to 0")
				a = 0
			y_pred_opt = y_pred_mat[:, opt_idx] == 1  # predictions of optimal model
			w[y_pred_opt == y] *= np.exp(-a)  # down weight correct predictions
			w[y_pred_opt != y] *= np.exp(a)  # up weight incorrect predictions
			w /= w.sum()  # normalize data point weights
			selected_ind[opt_idx] = True  # remove selected model from candidates
			coef[opt_idx] = a
			#print(f"Iter = {i}, optidx = {opt_idx}, optimal error = {opt_err}, accuracy = {(y_pred_opt==y).sum() / len(y)}")

		if coef.sum() == 0:
			coef = np.ones(n_mdl) / n_mdl
		else:
			coef /= coef.sum()
		#print(coef)

		self.coef_ = coef  # set normalized bossting coefficients

	def decision(self, ID_ary):
		if self.coef_ is None:
			raise ValueError("Master model untrained, train first using fit_master_mdl")

		decision_ary = np.zeros((len(ID_ary)))
		for i, mat in enumerate(self.G.mat_list):
			x = mat[self.G.IDmap[ID_ary]]
			decision_ary += self.coef_[i] * self.mdl_list[i].decision_function(x)

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
