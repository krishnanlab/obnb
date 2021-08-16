from sys import path
path.append("../")
from NLEval.wrapper.ParWrap import ParDat
import time

lst = list(range(10))
verbose = True
verb_kws = {'bar_length':40, 'log_steps':3}

for n_workers in 5, 12, 1, 0:
    print(f"Start testing n_workers = {n_workers}")

    @ParDat(lst, n_workers=n_workers, verbose=verbose, verb_kws=verb_kws)
    def test(a,b,c):
        time.sleep(1)
        return a, b, c

    for i in test(b=3,c=5):
        if verbose:
            pass
        else:
            print(i)
    print('')

