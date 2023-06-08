import time

from obnb.util.parallel import ParDat, ParDatExe, ParDatMap

lst = list(range(10))


def test(a, b, c):
    time.sleep(0.1)
    return a, b, c


for verbose in True, False:
    for n_workers in 5, 12, 1, 0:
        print(f"Start testing {n_workers=}, {verbose=}")

        pardat = ParDat(lst, n_workers=n_workers, verbose=verbose)
        pardatmap = ParDatMap(lst, n_workers=n_workers, verbose=verbose)
        pardatexe = ParDatExe(lst, n_workers=n_workers, verbose=verbose)

        print("ParDat")
        for i in pardat(test)(b=3, c=5):
            if not verbose:
                print(i)

        print("ParDatMap")
        print(pardatmap(test)(b=3, c=5))

        print("ParDatExe")
        pardatexe(test)(b=3, c=5)
