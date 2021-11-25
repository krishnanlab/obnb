import numpy as np
from NLEval.util.Exceptions import IDNotExistError, IDExistsError
from NLEval.util import checkers
from copy import deepcopy

__all__ = ["IDlst", "IDmap", "IDprop"]


class IDlst(object):
    """ID list object that stores a list of IDs"""

    def __init__(self):
        super(IDlst, self).__init__()
        self._lst = []

    def __iter__(self):
        """Yield all IDs"""
        return self._lst.__iter__()

    def __eq__(self, other):
        """Return true if two IDlst have same set of IDs"""
        checkers.checkType("other", self.__class__, other)
        return set(self._lst) == set(other._lst)

    def __add__(self, other):
        """Combine two ID list and return a copy"""
        checkers.checkType("other", self.__class__, other)
        new = self.copy()
        for ID in other:
            if ID not in new:
                new.addID(ID)
        return new

    def __sub__(self, other):
        """Return a copy of ID list that does not contain any
        IDs from the `other` ID list"""
        checkers.checkType("other", self.__class__, other)
        new = self.__class__()
        for ID in self:
            if ID not in other:
                new.addID(ID)
        return new

    def __and__(self, other):
        """Return a copy of ID list with IDs that exist in both lists"""
        checkers.checkType("other", self.__class__, other)
        new = self.__class__()
        for ID in set(self._lst) & set(other._lst):
            new.addID(ID)
        return new

    def __or__(self, other):
        """Same as `__add__`"""
        return self.__add__(other)

    def __xor__(self, other):
        """Return a copy of ID list with IDs that are unique"""
        checkers.checkType("other", self.__class__, other)
        new = self.__class__()
        for ID in set(self._lst) ^ set(other._lst):
            new.addID(ID)
        return new

    def __contains__(self, ID):
        """Return true if ID exist in current list"""
        checkers.checkType("ID", str, ID)
        return ID in self._lst

    def __getitem__(self, ID):
        """Return single or multiple (as numpy array) indices of ID
        depending on input type"""
        if isinstance(ID, str):
            return self._getitem_sinlge(ID)
        elif isinstance(ID, checkers.ITERABLE_TYPE):
            return self._getitem_multiple(ID)
        else:
            raise TypeError(
                "ID key(s) must be stirng or iterables of "
                + "string, not %s" % repr(type(ID))
            )

    def _getitem_sinlge(self, ID):
        self._check_ID_existence(ID, True)
        return self._lst.index(ID)

    def _getitem_multiple(self, IDs):
        checkers.checkTypesInIterable("IDs", str, IDs)
        idx_lst = []
        for ID in IDs:
            idx_lst.append(self._getitem_sinlge(ID))
        return np.array(idx_lst)

    def _check_ID_existence(self, ID, existence):
        """Check existence of ID and raise exceptions depending on
        desired existence of ID"""
        if (not existence) & (ID in self):
            raise IDExistsError("Existing ID %s" % repr(ID))
        elif existence & (ID not in self):
            raise IDNotExistError("Unknown ID %s" % repr(ID))

    @property
    def lst(self):
        """:obj:`list` of :obj:`str`: list of IDs.

        Note: the returned list is a copy of self._lst to prevent userside
        maniputation on data, use `.addID()` or `.popID()` to modify data

        """
        return self._lst.copy()

    @property
    def size(self):
        """int: number of IDs in list"""
        return len(self._lst)

    def copy(self):
        """Return a deepcopy of self"""
        return deepcopy(self)

    def isempty(self):
        """Return true if ID list is empty"""
        return self.size == 0

    def popID(self, ID):
        """Pop an ID out of list of IDs"""
        self._check_ID_existence(ID, True)
        idx = self[ID]
        self._lst.pop(idx)
        return idx

    def addID(self, ID):
        """Add new ID to end of list

        Note: raises IDExistsError if ID exists

        """
        self._check_ID_existence(ID, False)
        self._lst.append(ID)

    def update(self, IDs):
        """Update ID list
        Loop over all IDs and add to list if not yet existed

        Args:
            IDs(:obj:`list` of :obj:`str`): list of ID to be added,
            can be redundant

        Returns:
            n(int): number of newly added IDs

        """
        checkers.checkTypesInList("IDs", str, IDs)
        n = 0
        for ID in IDs:
            if ID not in self:
                self.addID(ID)
                n += 1
        return n

    def getID(self, idx):
        """Return ID by its index"""
        return self._lst[idx]

    @classmethod
    def from_list(cls, lst):
        checkers.checkType("ID list", list, lst)
        obj = cls()
        for ID in lst:
            obj.addID(ID)
        return obj


class IDmap(IDlst):
    """IDmap object that implements dictionary for more efficient mapping
    from ID to index"""

    def __init__(self):
        super(IDmap, self).__init__()
        self._map = {}

    def __contains__(self, ID):
        checkers.checkType("ID", str, ID)
        return ID in self._map

    def _getitem_sinlge(self, ID):
        self._check_ID_existence(ID, True)
        return self._map[ID]

    @property
    def map(self):
        """(dict of str:int): map from ID to index

        Note: the returned dict is a copy of self._map to prevent userside
        maniputation on data, use `.addID()` or `.popID()` to modify data

        """
        return self._map.copy()

    def popID(self, ID):
        self._check_ID_existence(ID, True)
        super(IDmap, self).popID(ID)
        idx = self._map.pop(ID)
        for i, ID in enumerate(self.lst[idx:]):
            self._map[ID] = idx + i
        return idx

    def addID(self, ID):
        new_idx = self.size
        super(IDmap, self).addID(ID)
        self._map[self._lst[-1]] = new_idx


class IDprop(IDmap):
    """ID properties object that stores property information of IDs"""

    def __init__(self):
        super(IDprop, self).__init__()
        self._prop_default_val = {}
        self._prop_default_type = {}
        self._prop = {}

    def __eq__(self, other):
        """Return true if two object have same set of IDs with same properties"""
        # check if two objects have same set of IDs
        if not super(IDprop, self).__eq__(other):
            return False
        # check if two objects have same set of properties
        if not set(self.propLst) == set(other.propLst):
            return False
        # check if properties have same values
        for prop in self.propLst:
            for ID in self:
                if self.getProp(ID, prop) != other.getProp(ID, prop):
                    return False
        return True

    def __add__(self, other):
        """Not Implemented"""
        raise NotImplementedError

    def __sub__(self, other):
        """Not Implemented"""
        raise NotImplementedError

    def _check_prop_existence(self, prop_name, existence):
        """Check existence of property name and raise exceptions depending
        on desired existence of property name

        Raises:
            IDExistsError: if desired existence of `prop_name` int `self._prop`
                is `False` and `prop_name` exists
            IDNotExitError: if desired existence of `prop_name` in `self._prop`
                is `True` and `prop_name` not yet existed

        """
        checkers.checkType("Property name", str, prop_name)
        if (not existence) & (prop_name in self._prop):
            raise IDExistsError("Existing property name %s" % repr(prop_name))
        elif existence & (prop_name not in self._prop):
            raise IDNotExistError("Unknown property name %s" % repr(prop_name))

    @property
    def prop_default_val(self):
        """(dict of str:obj): dictionary mapping from property name to
        default property value"""
        return self._prop_default_val.copy()

    @property
    def prop_default_type(self):
        return self._prop_default_type.copy()

    @property
    def prop(self):
        """(dict of str: :obj:`list` of :obj:): dictionary mapping from
        property name to list of property values in the order of ID list

        Note: the returned dict is a copy of self._prop to prevent userside
        maniputation on data, use `.setProp` to modify properties

        """
        return self._prop.copy()

    @property
    def propLst(self):
        """:obj:`list` of :obj:`str`: list of properties names"""
        return list(self._prop)

    def newProp(self, prop_name, default_val=None, default_type=None):
        """Create a new property

        Args:
            prop_name(str): name of property
            default_val: default value to set if property not specified

        Raises:
            TypeError: if prop_name is not `str` type, or if default_type
                is not `type` type, or if default_val type is inconsistent
                withspecified default_type

        """
        self._check_prop_existence(prop_name, False)
        if default_type is not None:
            checkers.checkType("Default type", type, default_type)
            if not isinstance(default_val, default_type):
                raise TypeError(
                    "Inconsistent type between default values "
                    + "%s and default type %s"
                    % (type(default_val), default_type)
                )
        if not self.isempty():
            prop_lst = [deepcopy(default_val) for i in range(self.size)]
        else:
            prop_lst = []
        self._prop_default_val[prop_name] = default_val
        self._prop_default_type[prop_name] = default_type
        self._prop[prop_name] = prop_lst

    def setProp(self, ID, prop_name, prop_val):
        """Set a pericif property value of an ID, must match default type if available"""
        self.getProp(ID, prop_name)  # check ID and prop_name validity
        if self.prop_default_type[prop_name] is not None:
            checkers.checkType(
                "Property value for %s" % repr(prop_name),
                self.prop_default_type[prop_name],
                prop_val,
            )
        self._prop[prop_name][self[ID]] = prop_val

    def getProp(self, ID, prop_name):
        """Return a specific properties associated with an ID

        Raises:
            IDNotExistError: if either ID or prop_name does not exist
            TypeError: if either ID or prop_name is not string type

        """
        self._check_ID_existence(ID, True)
        self._check_prop_existence(prop_name, True)
        return self._prop[prop_name][self[ID]]

    def delProp(self, prop_name):
        """Delete a property, along with its default type and value"""
        self._check_prop_existence(prop_name, True)
        self._prop.pop(prop_name)
        self._prop_default_val.pop(prop_name)
        self._prop_default_type.pop(prop_name)

    def getAllProp(self, ID):
        """Return all properties associated with an ID"""
        return {prop: self.getProp(ID, prop) for prop in self.propLst}

    def popID(self, ID):
        """Pop ID from ID list, and all properties lists."""
        idx = super(IDprop, self).popID(ID)
        for prop in self.propLst:
            self._prop[prop].pop(idx)
        return idx

    def addID(self, ID, prop=None):
        """Add a new ID to list, optional input of properties

        Note: input properties must be one of the existing properties,
        `IDNotExistError` raised other wise. Use `.newProp()` to add new
        property.

        Args:
            ID(str): ID to be added
            prop(:obj:`dict` of str:obj): dictionary specifying property(s)
                of the input ID. Corresponding properties must follow default
                type as specified in `.prop_default_type` if any. If `None`
                specified for `prop` (or `prop` doesn't contain all properties
                needed), default input from `.prop_default_val` is used to set
                (or fill in missing properties of) new ID properties.

        """
        if prop is not None:
            checkers.checkType("Properties", dict, prop)
            prop = prop.copy()
            for prop_name in prop:
                self._check_prop_existence(prop_name, True)
            # chekc type of prop val and fill in missing properties
            for prop_name, default_type in self.prop_default_type.items():
                if prop_name in prop:
                    if default_type is not None:
                        checkers.checkType(
                            "Properties Values", default_type, prop[prop_name]
                        )
                else:
                    prop[prop_name] = deepcopy(self.prop_default_val[prop_name])
        else:
            prop = deepcopy(self.prop_default_val)
        super(IDprop, self).addID(ID)
        for prop_name, prop_val in prop.items():
            self._prop[prop_name].append(prop_val)
