from sys import path
path.append("../")
from NLEval.wrapper import ParWrap
import time

lst = list(range(20))
@ParWrap.ParDat(lst)
def test(a,b,c):
	time.sleep(2)
	#print("a=%s"%repr(a))
	return a

for i in test(b=3,c=5):
	print(i)

