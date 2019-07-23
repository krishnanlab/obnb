

def checkType(name, targetType, val):
	if not isinstance(val, targetType):
		if isinstance(targetType, tuple):
			#multiple types
			typeString = ""
			for idx, typeName in enumerate(targetType):
				if idx == 0:
					typeString += "type %s"%repr(typeName)
				else:
					typeString += " or type %s"%repr(typeName)
		else:
			typeString = "type %s"%repr(targetType)

		raise TypeError("'%s' should be %s, not %s"%(name, typeString, repr(type(val))))

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


