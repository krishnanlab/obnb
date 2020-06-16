"""Type checking functions

This module contains a collection of checkers which could be used to ensure input 
values to function calls or class initializations are valid.

Attributes:
	INT_TYPE(:obj:`tuple` of :obj:`type`): integer types
	FLOAT_TYPE(:obj:`tuple` of :obj:`type`): float types
	NUMERIC_TYPE(:obj:`tuple` of :obj:`type`): numeric types (int or float)
	NUMSTRING_TYPE(:obj:`tuple` of :obj:`type`): numeric string types (int or float or str)
	ITERABLE_TYPE(:obj:`tuple` of :obj:`type`): iterable type

"""

import numpy as np
from collections.abc import Iterable

__all__ = [
	'INT_TYPE',
	'FLOAT_TYPE',
	'NUMERIC_TYPE',
	'NUMSTRING_TYPE',
	'ITERABLE_TYPE',
	'checkType',
	'checkTypeErrNone',
	'checkTypeAllowNone',
	'checkTypesInIterable',
	'checkTypesInList',
	'checkTypesInSet',
	'checkTypesInNumpyArray',
	'checkNumpyArrayIsNumeric',
	'checkNumpyArrayNDim',
	'checkNumpyArrayShape'
]

INT_TYPE = (int, np.integer)
FLOAT_TYPE = (float, np.floating, np.float32)
NUMERIC_TYPE = INT_TYPE + FLOAT_TYPE
NUMSTRING_TYPE = INT_TYPE + FLOAT_TYPE + (str,)
ITERABLE_TYPE = Iterable

def checkType(name, targetType, val):
	"""Check the type of an input value

	Args:
		name(str): name of the value
		targetType(type): desired type of the value
		val(any): value to be checked

	Raises:
		TypeError: if `val` is not an instance of `targetType`

	"""
	if not isinstance(val, targetType):
		raise TypeError("%s should be %s, not %s: %s"%
						(repr(name), repr(targetType), repr(type(val)), repr(val)))

def checkTypeErrNone(name, targetType, val):
	"""Type cheking with `checkType` and raises `ValueError` `val` is `None`
	
	Raises:
		ValueError: if `val` is `None`

	"""
	if val is not None:
		checkType(name, targetType, val)
	else:
		raise ValueError("Value for %s has not yet been provided"%repr(name))

def checkTypeAllowNone(name, targetType, val):
	"""Type cheking with `checkType` and allow `None` for `val`"""
	if val is not None:
		checkType(name, targetType, val)

def checkTypesInIterable(name, targetType, val):
	"""Check the types of all elements in an iterable"""
	for idx, i in enumerate(val):
		if not isinstance(i, targetType):
			raise TypeError("All instances in %s must be type %s, "%\
							(repr(name), repr(targetType)) + \
							"invalid type %s found at position (%d): %s"%\
							(repr(type(i)), idx, repr(i)))

def checkTypesInList(name, targetType, val):
	"""Check types of all elements in a list"""
	checkType(name, list, val)
	checkTypesInIterable(name, targetType, val)

def checkTypesInSet(name, targetType, val):
	"""Check types of all elements in a set"""
	checkType(name, set, val)
	checkTypesInIterable(name, targetType, val)	

def checkTypesInNumpyArray(name, targetType, val):
	"""Check types of all elements in a numpy array"""
	checkType(name, np.ndarray, val)
	checkTypesInIterable(name, targetType, val)	

def checkNumpyArrayIsNumeric(name, ary):
	"""Check if numpy array is numeric type"""
	checkType(name, np.ndarray, ary)
	if not any([ary.dtype == i for i in NUMERIC_TYPE]):
		raise TypeError("%s should be numeric, not type %s"%
						(repr(name), repr(ary.dtype)))

def checkNumpyArrayNDim(name, targetNDim, ary):
	"""Check the rank (number of dimension) of a numpy array
	
	Args:
		name(str): name of the input array
		targetNDim(int): desired number of dimensions
		ary(:obj:`numpy.ndarray`): numpy array to be checked

	Raises:
		ValueError: if the number of dimensions of the input array is different 
			from the target number of dimensions

	"""
	checkType("targetNDim", INT_TYPE, targetNDim)
	checkType(name, np.ndarray, ary)
	NDim = len(ary.shape)
	if NDim != targetNDim:
		raise ValueError("%s should be %d dimensional array, not %d dimensional"%
						(repr(name), targetNDim, NDim))

def checkNumpyArrayShape(name, targetShape, ary):
	"""Check the shape of a numpy array

	Args:
		name(str): name of the input array
		targetShape: desired shape of the array
		ary(:obj:`numpy.ndarray`): numpy array to be checked

	Raises:
		ValueError: if the sape of the input array differ from the target shape

	"""
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
