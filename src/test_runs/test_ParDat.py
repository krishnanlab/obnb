from sys import path
path.append("../")
from NLEval.wrapper.ParWrap import ParDat
import time

lst = list(range(10))

@ParDat(lst, n_jobs=5)
def test5(a,b,c):
	time.sleep(2)
	return a, b, c

@ParDat(lst, n_jobs=12)
def test12(a,b,c):
	time.sleep(2)
	return a, b, c

@ParDat(lst, n_jobs=1)
def test1(a,b,c):
	time.sleep(2)
	return a, b, c

for test in test5, test12, test1:
	for i in test(b=3,c=5):
		print(i)
	print('')

