from NLEval.util import checkers

class BaseFlags:
	def __init__(self):
		self.verbose = True

	@property
	def verbose(self):
		return self._verbose
	
	@verbose.setter
	def verbose(self, val):
		checkers.checkType('verbose', bool, val)

