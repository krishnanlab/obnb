import time
from NLEval.util import checkers

class TimeIt:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def __call__(self, func):
        def wrapper(*args):
            start = time.time()
            func(*args)
            end = time.time()
            time_interval = end - start
            print(f"Took {time_interval:.2f} seconds to run function {repr(func)}")
        if self.verbose:
            return wrapper
        else:
            return func

    @property
    def verbose(self):
        return self._verbose
    
    @verbose.setter
    def verbose(self, val):
        checkers.checkTypeErrNone('verbose', bool, val)
        self._verbose = val
    
