from sys import path
path.append("../")
from NLEval.wrapper.ParWrap import ParDat
import time

lst = list(range(10))
verbose = True

for n_workers in 5, 12, 1, 0:
	print(f"Start testing n_workers = {n_workers}")

	@ParDat(lst, n_workers=n_workers, verbose=verbose)
	def test(a,b,c):
		time.sleep(1)
		return a, b, c

	for i in test(b=3,c=5):
		if verbose:
			pass
		else:
			print(i)
	print('')

