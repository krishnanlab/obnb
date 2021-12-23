from sys import path

path.append("../")
from NLEval.wrapper.ParWrap import ParDat
import time

lst = list(range(10))

for verbose in True, False:
    for n_workers in 5, 12, 1, 0:
        print(f"Start testing {n_workers=}, {verbose=}")

        @ParDat(lst, n_workers=n_workers, verbose=verbose, bar_length=40)
        def test(a, b, c):
            time.sleep(0.5)
            return a, b, c

        for i in test(b=3, c=5):
            if not verbose:
                print(i)
