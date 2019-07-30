class IDNotExistError(Exception):
	"""Raised when query ID not exist"""
	pass

class IDExistsError(Exception):
	"""Raised when try to add new ID that already exists"""
	pass