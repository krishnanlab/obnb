from sys import path
path.append("../")
from NLEval.wrapper.ParWrap import ParDat
import time

lst = list(range(10))

for n_workers in 5, 12, 1:
	print(f"Start testing n_workers = {n_workers}")

	@ParDat(lst, n_workers=n_workers)
	def test(a,b,c):
		time.sleep(2)
		return a, b, c

	for i in test(b=3,c=5):
		print(i)
	print('')

