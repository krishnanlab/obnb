import numpy as np
from collections.abc import Iterable

INT_TYPE = (int, np.integer)
FLOAT_TYPE = (float, np.floating)
NUMERIC_TYPE = INT_TYPE + FLOAT_TYPE
NUMSTRING_TYPE = INT_TYPE + FLOAT_TYPE + (str,)
ITERABLE_TYPE = Iterable

def checkType(name, targetType, val):
	if not isinstance(val, targetType):
		raise TypeError("%s should be %s, not %s: %s"%
				(repr(name), repr(targetType), repr(type(val)), repr(val)))

def checkTypeErrNone(name, targetType, val):
	if val is not None:
		checkType(name, targetType, val)
	else:
		raise ValueError("Value for '%s' has not yet been provided"%name)

def checkTypeAllowNone(name, targetType, val):
	if val is not None:
		checkType(name, targetType, val)

def checkTypesInIterable(name, targetType, val):
	checkType(name, ITERABLE_TYPE, val)
	for idx, i in enumerate(val):
		if not isinstance(i, targetType):
			raise TypeError("All instances in %s must be type %s, invalid type %s found at position (%d)"%\
				(repr(name), repr(targetType), repr(type(i)), idx))

def checkTypesInList(name, targetType, val):
	checkType(name, list, val)
	checkTypesInIterable(name, targetType, val)

def checkNumpyArrayIsNumeric(name, ary):
	checkType(name, np.ndarray, ary)
	if not any([ary.dtype == i for i in NUMERIC_TYPE]):
		raise TypeError("%s should be numeric, not type %s"%
			(repr(name), repr(ary.dtype)))

def checkNumpyArrayNDim(name, targetNDim, ary):
	checkType("targetNDim", INT_TYPE, targetNDim)
	checkType(name, np.ndarray, ary)
	NDim = len(ary.shape)
	if NDim != targetNDim:
		raise ValueError("%s should be %d dimensional array, not %d dimensional"%
			(repr(name), targetNDim, NDim))

def checkNumpyArrayShape(name, targetShape, ary):
	if isinstance(targetShape, ITERABLE_TYPE):
		checkTypesInIterable("targetShape", INT_TYPE, targetShape)
		NDim = len(targetShape)
	else:
		checkType("targetShape", INT_TYPE, targetShape)
		NDim = 1
		targetShape = (targetShape,)
	checkNumpyArrayNDim(name, NDim, ary)
	shape = ary.shape
	if shape != targetShape:
		raise ValueError("%s should be in shape %s, not %s"%
			(repr(name), repr(targetShape), repr(shape)))