from __future__ import annotations

from copy import deepcopy

import numpy as np

from nleval.exception import IDExistsError, IDNotExistError
from nleval.typing import Dict, Iterable, List
from nleval.util import checkers

__all__ = ["IDlst", "IDmap", "IDprop"]


class IDlst:
    """ID list object that stores a list of IDs."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset ID list."""
        self._lst = []

    def __iter__(self):
        """Yield all IDs."""
        return self._lst.__iter__()

    def __len__(self):
        """Return the size of the id list."""
        return self.size

    def __eq__(self, other):
        """Return true if two IDlst have same set of IDs."""
        checkers.checkType("other", self.__class__, other)
        return set(self._lst) == set(other._lst)

    def __add__(self, other):
        """Combine two ID list and return a copy."""
        checkers.checkType("other", self.__class__, other)
        new = self.copy()
        for identifier in other:
            if identifier not in new:
                new.add_id(identifier)
        return new

    def __sub__(self, other):
        """Return a copy of ID list that does not contain any IDs from the `other` ID
        list."""
        checkers.checkType("other", self.__class__, other)
        new = self.__class__()
        for identifier in self:
            if identifier not in other:
                new.add_id(identifier)
        return new

    def __and__(self, other):
        """Return a copy of ID list with IDs that exist in both lists."""
        checkers.checkType("other", self.__class__, other)
        new = self.__class__()
        for identifier in set(self._lst) & set(other._lst):
            new.add_id(identifier)
        return new

    def __or__(self, other):
        """Same as `__add__.`"""
        return self.__add__(other)

    def __xor__(self, other):
        """Return a copy of ID list with IDs that are unique."""
        checkers.checkType("other", self.__class__, other)
        new = self.__class__()
        for identifier in set(self._lst) ^ set(other._lst):
            new.add_id(identifier)
        return new

    def __contains__(self, identifier):
        """Return true if ID exist in current list."""
        checkers.checkType("ID", str, identifier)
        return identifier in self._lst

    def __getitem__(self, identifier):
        """Return single or multiple (as numpy array) indices of ID depending on input
        type."""
        if isinstance(identifier, str):
            return self._getitem_sinlge(identifier)
        elif isinstance(identifier, Iterable):
            return self._getitem_multiple(identifier)
        else:
            raise TypeError(
                f"ID key(s) must be stirng or iterables of "
                f"string, not {type(identifier)!r}",
            )

    def _getitem_sinlge(self, identifier):
        self._check_ID_existence(identifier, True)
        return self._lst.index(identifier)

    def _getitem_multiple(self, identifiers):
        checkers.checkTypesInIterable("IDs", str, identifiers)
        idx_lst = []
        for identifier in identifiers:
            idx_lst.append(self._getitem_sinlge(identifier))
        return np.array(idx_lst)

    def _check_ID_existence(self, identifier, existence):
        """Check existence of ID and raise exceptions depending on desired existence of
        ID."""
        if (not existence) & (identifier in self):
            raise IDExistsError(f"Existing ID {identifier!r}")
        elif existence & (identifier not in self):
            raise IDNotExistError(f"Unknown ID {identifier!r}")

    @property
    def lst(self):
        """:obj:`list` of :obj:`str`: list of IDs.

        Note: the returned list is a copy of self._lst to prevent userside
        maniputation on data, use `.add_id()` or `.pop_id()` to modify data

        """
        return self._lst.copy()

    @property
    def size(self):
        """int: number of IDs in list."""
        return len(self._lst)

    def copy(self):
        """Return a deepcopy of self."""
        return deepcopy(self)

    def isempty(self):
        """Return true if ID list is empty."""
        return self.size == 0

    def pop_id(self, identifier):
        """Pop an ID out of list of IDs."""
        self._check_ID_existence(identifier, True)
        idx = self[identifier]
        self._lst.pop(idx)
        return idx

    def add_id(self, identifier):
        """Add new ID to end of list.

        Note: raises IDExistsError if ID exists

        """
        self._check_ID_existence(identifier, False)
        self._lst.append(identifier)

    def update(self, identifiers):
        """Update ID list.

        Loop over all IDs and add to list if not yet existed

        Args:
            identifiers(:obj:`list` of :obj:`str`): list of IDs to be added,
                can be redundant.

        Returns:
            n(int): number of newly added IDs

        """
        checkers.checkTypesInList("IDs", str, identifiers)
        n = 0
        for identifier in identifiers:
            if identifier not in self:
                self.add_id(identifier)
                n += 1
        return n

    def get_id(self, idx: int) -> str:
        """Return ID by its index."""
        return self._lst[idx]

    def get_ids(self, idxs: Iterable[int]) -> List[str]:
        """Return a list of IDs given indexes."""
        return list(map(self._lst.__getitem__, idxs))

    @classmethod
    def from_list(cls, lst):
        checkers.checkType("ID list", list, lst)
        obj = cls()
        for identifier in lst:
            obj.add_id(identifier)
        return obj


class IDmap(IDlst):
    """IDmap object that implements dictionary for more efficient mapping from ID to
    index."""

    def __init__(self):
        super().__init__()

    def __contains__(self, identifier):
        checkers.checkType("ID", str, identifier)
        return identifier in self._map

    def _getitem_sinlge(self, identifier):
        self._check_ID_existence(identifier, True)
        return self._map[identifier]

    @property
    def map(self):
        """(dict of str:int): map from ID to index.

        Note: the returned dict is a copy of self._map to prevent userside
        maniputation on data, use `.add_id()` or `.pop_id()` to modify data

        """
        return self._map.copy()

    def pop_id(self, identifier):
        self._check_ID_existence(identifier, True)
        super().pop_id(identifier)
        idx = self._map.pop(identifier)
        for i, identifier in enumerate(self.lst[idx:]):
            self._map[identifier] = idx + i
        return idx

    def add_id(self, identifier):
        new_idx = self.size
        super().add_id(identifier)
        self._map[self._lst[-1]] = new_idx

    def reset(self, identifiers: list[str] | None = None):
        """Reset the IDmap with a list of IDs.

        Args:
            identifiers (list of str): List of IDs to be used to construct the
                new IDmap. If set to None, then leave as empty.

        """
        super().reset()
        self._map: Dict[str, int] = {}

        if identifiers is not None:
            for identifier in identifiers:
                self.add_id(identifier)

    def align(
        self,
        new_idmap: IDmap,
        join: str = "right",
        update: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align current idmap with another idmap.

        Alignt the IDmaps and return alignment index (left, right).

        Args:
            new_idmap: The new idmap to align.
            join (str): Strategy of selecting the IDs, choices:
                * "right": Use all IDs from the new IDmap
                * "left": Use all IDs from the old IDmap
                * "intersection": Use common IDs from the old and ndew IDmaps.
                * "union": Use all IDs from both the new and old IDmap
            update (bool): Whether or not to update the IDmap passed in
                (default: :obj:`False`)

        Raises:
            TypeError: If new_idmap is not of type IDmap
            ValueError: If new_idmap is None; or no common IDs found (except
                when join method is "union")

        """
        checkers.checkType("IDmap", IDmap, new_idmap)
        if join in ["right", "left", "intersection"]:
            common_ids = sorted(set(self._map) & set(new_idmap._map))
            if not common_ids:
                raise ValueError("Alignment failed since no common ID found")

            left_idx = self[common_ids]
            right_idx = new_idmap[common_ids]

            if join == "right":
                self.reset(new_idmap.lst)
            elif join == "left" and update:
                new_idmap.reset(self.lst)
            elif join == "intersection":
                self.reset(common_ids)
                if update:
                    new_idmap.reset(common_ids)

        elif join == "union":
            left_ids = self.lst
            right_ids = new_idmap.lst
            full_ids = sorted(set(left_ids) | set(right_ids))

            self.reset(full_ids)
            if update:
                new_idmap.reset(full_ids)

            left_idx = self[left_ids]
            right_idx = self[right_ids]
        else:
            raise ValueError(f"Unknwon join type: {join!r}")

        return left_idx, right_idx


class IDprop(IDmap):
    """ID properties object that stores property information of IDs."""

    def __init__(self):
        super().__init__()

    def reset(self):
        """Reset ID properties."""
        super().reset()
        self._prop_default_val = {}
        self._prop_default_type = {}
        self._prop = {}

    def __eq__(self, other):
        """Return true if two object have same set of IDs with same properties."""
        # check if two objects have same set of IDs
        if not super().__eq__(other):
            return False
        # check if two objects have same set of properties
        if not set(self.properties) == set(other.properties):
            return False
        # check if properties have same values
        for prop in self.properties:
            for identifier in self:
                if self.get_property(identifier, prop) != other.get_property(
                    identifier,
                    prop,
                ):
                    return False
        return True

    def __add__(self, other):
        """Not Implemented."""
        raise NotImplementedError

    def __sub__(self, other):
        """Not Implemented."""
        raise NotImplementedError

    def _check_prop_existence(self, prop_name, existence):
        """Check existence of property name and raise exceptions depending on desired
        existence of property name.

        Raises:
            IDExistsError: if desired existence of `prop_name` int `self._prop`
                is `False` and `prop_name` exists
            IDNotExitError: if desired existence of `prop_name` in `self._prop`
                is `True` and `prop_name` not yet existed

        """
        checkers.checkType("Property name", str, prop_name)
        if (not existence) & (prop_name in self._prop):
            raise IDExistsError(f"Existing property name {prop_name!r}")
        elif existence & (prop_name not in self._prop):
            raise IDNotExistError(f"Unknown property name {prop_name!r}")

    @property
    def prop_default_val(self):
        """(dict of str:obj): dictionary mapping from property name to default property
        value."""
        return self._prop_default_val.copy()

    @property
    def prop_default_type(self):
        return self._prop_default_type.copy()

    @property
    def prop(self):
        """(dict of str: :obj:`list` of :obj:): dictionary mapping from property name to
        list of property values in the order of ID list.

        Note: the returned dict is a copy of self._prop to prevent userside
        maniputation on data, use `.set_property` to modify properties

        """
        return self._prop.copy()

    @property
    def properties(self):
        """:obj:`list` of :obj:`str`: list of properties names."""
        return list(self._prop)

    def new_property(self, prop_name, default_val=None, default_type=None):
        """Create a new property.

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
                    f"Inconsistent type between default values "
                    f"{type(default_val)!r} and default type {default_type!r}",
                )
        if not self.isempty():
            prop_lst = [deepcopy(default_val) for i in range(self.size)]
        else:
            prop_lst = []
        self._prop_default_val[prop_name] = default_val
        self._prop_default_type[prop_name] = default_type
        self._prop[prop_name] = prop_lst

    def set_property(self, identifier, prop_name, prop_val):
        """Set a specific property value of an ID.

        Note: must match default type if available.

        """
        self.get_property(identifier, prop_name)  # check ID and prop_name
        if self.prop_default_type[prop_name] is not None:
            checkers.checkType(
                f"Property value for {prop_name!r}",
                self.prop_default_type[prop_name],
                prop_val,
            )
        self._prop[prop_name][self[identifier]] = prop_val

    def get_property(self, identifier, prop_name):
        """Return a specific properties associated with an ID.

        Raises:
            IDNotExistError: if either ID or prop_name does not exist
            TypeError: if either ID or prop_name is not string type

        """
        self._check_ID_existence(identifier, True)
        self._check_prop_existence(prop_name, True)
        return self._prop[prop_name][self[identifier]]

    def remove_property(self, prop_name):
        """Remove a property along with its default type and value."""
        self._check_prop_existence(prop_name, True)
        self._prop.pop(prop_name)
        self._prop_default_val.pop(prop_name)
        self._prop_default_type.pop(prop_name)

    def get_all_properties(self, identifier):
        """Return all properties associated with an ID."""
        return {i: self.get_property(identifier, i) for i in self.properties}

    def pop_id(self, identifier):
        """Pop ID from ID list, and all properties lists."""
        idx = super().pop_id(identifier)
        for prop in self.properties:
            self._prop[prop].pop(idx)
        return idx

    def add_id(self, identifier, prop=None):
        """Add a new ID to list, optional input of properties.

        Note: input properties must be one of the existing properties,
        `IDNotExistError` raised other wise. Use `.new_property()` to add new
        property.

        Args:
            identifier(str): ID to be added
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
                            "Properties Values",
                            default_type,
                            prop[prop_name],
                        )
                else:
                    prop[prop_name] = deepcopy(self.prop_default_val[prop_name])
        else:
            prop = deepcopy(self.prop_default_val)
        super().add_id(identifier)
        for prop_name, prop_val in prop.items():
            self._prop[prop_name].append(prop_val)
