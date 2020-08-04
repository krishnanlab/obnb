import time
from NLEval.util import checkers

class TimeIt:
	def __init__(self, verbose=True):
		self.verbose = verbose

	def __call__(self, fun):
		def wrapper(*args):
			start = time.time()
			fun(*args)
			end = time.time()
			time_interval = end - start
			print("Took %.2f seconds to run function '%s'"%(time_interval, repr(fun)))
		if self.verbose:
			return wrapper
		else:
			return fun

	@property
	def verbose(self):
		return self._verbose
	
	@verbose.setter
	def verbose(self, val):
		checkers.checkTypeErrNone('verbose', bool, val)
		self._verbose = val
	
