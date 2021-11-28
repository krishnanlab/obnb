import numpy as np
from NLEval.label import Filter
from NLEval.util import IDHandler, checkers
from NLEval.util.Exceptions import IDExistsError
from NLEval.valsplit import Base

__all__ = ["BaseLSC", "SplitLSC"]


class BaseLSC(IDHandler.IDprop):
    """Collection of labelsets.

    This class is used for managing collection of labelsets.


    Example GMT (Gene Matrix Transpose):

    '''
    Geneset1    Description1    Gene1   Gene2   Gene3
    Geneset2    Description2    Gene2   Gene4   Gene5   Gene6
    '''

    Example internal data for a label collection with above GMT data:

    self.entity_ids = ['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5', 'Gene6']
    self.entity.prop = {'Noccur': [1, 2, 1, 1, 1, 1]}
    self.label_ids = ['Geneset1', 'Geneset2']
    self.prop = {
        'Info':['Description1', 'Description2']
        'Labelset':[
            {'Gene1', 'Gene2', 'Gene3'},
            {'Gene2', 'Gene4', 'Gene5', 'Gene6'}
        ]
    }

    """

    def __init__(self):
        super(BaseLSC, self).__init__()
        self.entity = IDHandler.IDprop()
        self.entity.newProp("Noccur", 0, int)
        self.newProp("Info", "NA", str)
        self.newProp("Labelset", set(), set)
        self.newProp("Negative", {None}, set)

    def _show(self):
        """Debugging prints."""
        print("Labelsets IDs:")
        print(self._lst)
        print("Labelsets Info:")
        print(self._prop["Info"])
        print("Labelsets:")
        for lbset in self._prop["Labelset"]:
            print(lbset)
        print("Entities IDs:")
        print(self.entity._lst)
        print("Entities occurances:")
        print(self.entity._prop)

    @property
    def entity_ids(self):
        """List of all entity IDs that are part of at least one labelset."""
        return [i for i in self.entity if self.get_noccur(i) > 0]

    @property
    def label_ids(self):
        """:obj:`list` of :obj:`str`: list of all labelset names."""
        return self.lst

    def add_labelset(self, lst, label_id, label_info=None):
        """Add a new labelset.

        Args:
            lst(:obj:`list` of :obj:`str`): list of IDs of entiteis belong
                to the input label
            label_id(str): name of label
            label_info(str): description of label

        """
        self.add_id(
            label_id,
            {} if label_info is None else {"Info": label_info},
        )
        try:
            self.entity.update(lst)
        except Exception as e:
            # if entity list not updated successfully, pop the new labelset
            self.popID(label_id)
            raise e
        self.update_labelset(lst, label_id)

    def pop_labelset(self, label_id):
        """Pop a labelset.

        Note:
            This also removes any entity that longer belongs to any labelset.

        """
        self.reset_labelset(label_id)
        self.popID(label_id)

    def update_labelset(self, lst, label_id):
        """Update an existing labelset.

        Take list of entities IDs and update current labelset with a label
        name matching `label_id`. Any ID in the input list `lst` that does
        not exist in the entity list will be added to the entity list. Increment
        the `Noccur` property of any newly added entites to the labelset by 1.

        Note: label_id must already existed, use `.add_labelset()` for adding
        new labelset

        Args:
            lst(:obj:`list` of :obj:`str`): list of entiteis IDs to be
                added to the labelset, can be redundant.

        Raises:
            TypeError: if `lst` is not `list` type or any element within `lst`
                is not `str` type

        """
        checkers.checkTypesInList("Entity list", str, lst)
        lbset = self.get_labelset(label_id)
        for entity_id in lst:
            if entity_id not in self.entity:
                self.entity.add_id(entity_id)
            if entity_id not in lbset:
                lbset.update([entity_id])
                self.entity.setProp(
                    entity_id,
                    "Noccur",
                    self.get_noccur(entity_id) + 1,
                )

    def reset_labelset(self, label_id):
        """Reset an existing labelset to an empty set.

        Setting the labelset back to empty and deecrement `Noccur` of all
        entites belonging to the labelset by 1.

        """
        lbset = self.get_labelset(label_id)
        for entity_id in lbset:
            self.entity.setProp(
                entity_id,
                "Noccur",
                self.get_noccur(entity_id) - 1,
            )
            if (
                self.entity.getAllProp(entity_id)
                == self.entity.prop_default_val
            ):
                self.entity.popID(entity_id)
        lbset.clear()

    def pop_entity(self, entity_id):
        """Pop an entity from entity list, and also remove it from all labelsets.

        Note: Unlike `pop_labelset`, if after removal, a labelset beomes empty, the
        labelset itself is NOT removed. This is for more convenient comparison of
        labelset sizes before and after filtering.

        """
        self.entity.popID(entity_id)
        for label_id in self.label_ids:
            self.get_labelset(label_id).difference_update([entity_id])

    def get_info(self, label_id):
        """Return description of a labelset."""
        return self.getProp(label_id, "Info")

    def get_labelset(self, label_id):
        """Return set of entities associated with a label."""
        return self.getProp(label_id, "Labelset")

    def get_negative(self, label_id):
        """Return set of negative samples of a labelset.

        Note:
            If negative samples not available, use complement of labelset

        """
        neg = self.getProp(label_id, "Negative")

        if neg == {None}:
            all_positives = {
                i for i in self.entity.map if self.get_noccur(i) > 0
            }
            return all_positives - self.get_labelset(label_id)

        return neg

    def set_negative(self, lst, label_id):
        checkers.checkTypesInList("Negative entity list", str, lst)
        lbset = self.get_labelset(label_id)
        for entity_id in lst:
            self.entity._check_ID_existence(entity_id, True)
            if entity_id in lbset:
                raise IDExistsError(
                    f"Entity {entity_id!r} is positive in labelset, "
                    f"{label_id!r}, cannot be set to negative",
                )
        self.setProp(label_id, "Negative", set(lst))

    def get_noccur(self, entity_id):
        """Return the number of labelsets in which an entity participates."""
        return self.entity.getProp(entity_id, "Noccur")

    def apply(self, filter_func, inplace=False):
        """Apply filter to labelsets, see `NLEval.label.Filter` for more info.

        Args:
            filter_func
            inplace(bool): whether or not to modify original object
                - `True`: apply filter directly on the original object
                - `False`: apply filter on a copy of the original object

        Returns:
            Labelset coolection object after filtering.

        """
        checkers.checkType("Filter", Filter.BaseFilter, filter_func)
        checkers.checkType("inplace", bool, inplace)
        obj = self if inplace else self.copy()
        filter_func(obj)
        return obj

    def export(self, fp):
        """Export self as a '.lsc' file.

        Notes:
            '.lsc' is a csv file storing entity labels in matrix form, where
            first column is entity IDs, first and second rows correspond to
            label ID and label information respectively. If an entity 'i' is
            annotated with a label 'j', the corresponding 'ij' entry is marked
            as '1', else if it is considered a negative for that label, it is
            marked as '-1', otherwise it is '0', standing for neutral.

            entity_idmap is necessary since not all entities are guaranteed to
            be part of at least one label.

        Input:
            fp(str): path to file to save, including file name, with/without
                extension.

        """
        entity_ids = self.entity_ids
        entity_idmap = {
            entity_id: idx for idx, entity_id in enumerate(entity_ids)
        }
        label_ids = self.label_ids
        label_info_list = [self.get_info(label_id) for label_id in label_ids]
        mat = np.zeros((len(entity_ids), len(label_ids)), dtype=int)

        for j, label_id in enumerate(label_ids):
            positive_set = self.get_labelset(label_id)
            negative_set = self.get_negative(label_id)

            for sign, labelset in zip(
                ["1", "-1"],
                [positive_set, negative_set],
            ):
                for entity_id in labelset:
                    i = entity_idmap[entity_id]
                    mat[i, j] = sign

        fp = fp if fp.endswith(".lsc") else fp + ".lsc"
        with open(fp, "w") as f:
            # headers
            label_ids = "\t".join(label_ids)
            label_info_str = "\t".join(label_info_list)
            f.write(f"Label ID\t{label_ids}\n")
            f.write(f"Label Info\t{label_info_str}\n")

            # annotations
            for i, entity_id in enumerate(entity_ids):
                indicator_string = "\t".join(map(str, mat[i]))
                f.write(f"{entity_id}\t{indicator_string}\n")

    def export_gmt(self, fp):
        """Export self as a '.gmt' (Gene Matrix Transpose) file.

        Input:
            fp(str): path to file to save, including file name, with/without extension.
        """
        fp += "" if fp.endswith(".gmt") else ".gmt"
        with open(fp, "w") as f:
            for label_id in self.label_ids:
                label_info = self.get_info(label_id)
                labelset_str = "\t".join(self.get_labelset(label_id))
                f.write(f"{label_id}\t{label_info}\t{labelset_str}\n")

    def load_entity_properties(
        self,
        fp,
        prop_name,
        default_val,
        default_type,
        interpreter=int,
        comment="#",
        skiprows=0,
    ):
        """Load entity properties from file.

        The file is tab seprated with two columns, first column
        contains entities IDs, second column contains corresponding
        properties of entities.

        Args:
            fp(str): path to the entity properties file.
            default_val: default value of property of an entity
                if not specified.
            default_type(type): default type of the property.
            interpreter: function to transfrom property value from
                string to some other value

        """
        self.entity.newProp(prop_name, default_val, default_type)
        with open(fp, "r") as f:
            for i, line in enumerate(f):
                if (i < skiprows) | line.startswith(comment):
                    continue
                entity_id, val = line.strip().split()
                if entity_id not in self.entity:
                    self.entity.add_id(entity_id)
                self.entity.setProp(entity_id, prop_name, interpreter(val))

    @classmethod
    def from_gmt(cls, fp):
        """Load data from Gene Matrix Transpose `.gmt` file.

        Args:
            fg(str): path to the `.gmt` file

        """
        lsc = cls()
        with open(fp, "r") as f:
            for line in f:
                label_id, label_info, *lst = line.strip().split("\t")
                lsc.add_labelset(lst, label_id, label_info)
        return lsc


class SplitLSC(BaseLSC):
    """Labelset collection equipped with split generator."""

    def __init__(self):
        super(SplitLSC, self).__init__()
        self._valsplit = None
        self._filter_switch = False

    @property
    def valsplit(self):
        """Ssplit generator for generating train/(val)/test splits."""
        return self._valsplit

    @valsplit.setter
    def valsplit(self, obj):
        checkers.checkType("Validation split generator", Base.BaseValSplit, obj)
        self._valsplit = obj

    def train_test_setup(self, graph, prop_name=None, min_pos=10):
        """Set up training and testing.

        Filter labelsets based on train/test samples

        Args:
            prop_name(str): name of entity properties used for generating splits
            min_pos(int): minimum number of positive in both training and testing
                sets of a given labelset below which labelset is discarded. If
                `None` specified, no filtering will be done.

        """
        if self.valsplit is None:
            raise AttributeError(
                "'valsplit' not configured, please assign validation split "
                "generator first. See `NLEval.valsplit` for more info.",
            )

        num_labelsets = None
        while num_labelsets != len(self.label_ids):
            num_labelsets = len(self.label_ids)
            # print(num_labelsets)
            # label_id_set = set(self.label_ids)
            self.valsplit.train_test_setup(self.entity, graph.idmap, prop_name)
            self.apply(
                Filter.LabelsetRangeFilterTrainTestPos(min_pos),
                inplace=True,
            )
            # for i in label_id_set - set(self.label_ids):
            #     print(f"Pop {i}")

    def split_labelset(self, label_id, entity_ids=None):
        """Split up a labelset by training and testing sets.

        Returns:
            A generator that yeilds train/test IDs and labels, see
            `NLEval.valsplit.Base.BaseValSplit.split` for more info.

        """
        if entity_ids is None:
            entity_ids = self.entity_ids.copy()

        pos_ids_set = set(self.get_labelset(label_id)) & set(entity_ids)
        neg_ids_set = set(self.get_negative(label_id)) & set(entity_ids)

        id_ary = np.array(list(pos_ids_set) + list(neg_ids_set))
        label_ary = np.zeros(len(id_ary), dtype=bool)
        label_ary[: len(pos_ids_set)] = True
        return self.valsplit.split(id_ary, label_ary)

    def export_splits(self, fp, graph):
        """Export (holdout) split information to npz file.

        Notes:
            * Only allow ``Holdout`` split type for now, since it is not
                specific for each label
            * Ignores neutral and set everything not positives as negatives,
                in the future, add an option to allow neutral labels by
                setting positive, neutral, and negative as +1, 0, and -1,
                respectively
            * Currently not checking whether the validation split is aligned
                with the graph, in the future, need to think of a way to make
                sure this is checked

        Args:
            fp(str): output file path
            graph(:obj:`NLEval.graph`): graph object, more specifically the IDs
                of the nodes in the graph, used for filtering IDs

        """
        checkers.checkType(
            "Labelset collection splitter (only support Holdout split now)",
            Base.BaseHoldout,
            self.valsplit,
        )
        valid = False if self.valsplit.valid_index is None else True
        self.valsplit.check_split_setup(valid)

        train_idx = graph.idmap[self.valsplit.train_index]
        test_idx = graph.idmap[self.valsplit.test_index]
        valid_idx = graph.idmap[self.valsplit.valid_index] if valid else np.NaN

        y = np.zeros((graph.size, len(self.label_ids)), dtype=bool)
        for i, label_id in enumerate(self.label_ids):
            pos_id_ary = np.array(list(self.get_labelset(label_id)))
            pos_idx_ary = graph.idmap[pos_id_ary]
            y[pos_idx_ary, i] = True

        np.savez(
            fp,
            y=y,
            train_idx=train_idx,
            valid_idx=valid_idx,
            test_idx=test_idx,
            IDs=graph.idmap.lst,
            label_ids=self.label_ids,
        )
