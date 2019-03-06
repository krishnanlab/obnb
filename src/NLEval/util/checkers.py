

def checkType(name, targetType, val):
	if not isinstance(val, targetType):
		raise TypeError("'%s' should be type '%s', not '%s'"%\
			(name, targetType, type(val)))

def checkTypeErrNone(name, targetType, val):
	if val is not None:
		checkType(name, targetType, val)
	else:
		raise ValueError("Value for '%s' has not yet been provided"%name)

def checkTypeAllowNone(name, targetType, val):
	if val is not None:
		checkType(name, targetType, val)

def checkFileExist():
	pass


