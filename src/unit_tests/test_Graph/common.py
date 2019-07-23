import os
from sys import path
pwd = os.path.dirname(os.path.abspath(__file__))
lst = pwd.split('/')
idx = lst.index('src') + 1
src_pth = '/'.join(lst[:idx])
path.append(src_pth)
from NLEval.util import IDmap
import unittest
import numpy as np
