from NLEval.util.Exceptions import IDExistsError, IDNotExistError
from NLEval.util import checkers, IDHandler
from NLEval.label import Filter
from NLEval.valsplit import Base
import numpy as np

__all__ = ["BaseLSC", "SplitLSC"]


class BaseLSC(IDHandler.IDprop):
    """Collection of labelsets

    This class is used for managing collection of labelsets.


    Example GMT (Gene Matrix Transpose):

    '''
    Geneset1    Description1    Gene1   Gene2   Gene3
    Geneset2    Description2    Gene2   Gene4   Gene5   Gene6
    '''

    Example internal data for a label collection with above GMT data:

    self.entityIDlst = ['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5', 'Gene6']
    self.entity.prop = {'Noccur': [1, 2, 1, 1, 1, 1]}
    self.labelIDlst = ['Geneset1', 'Geneset2']
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
        """Debugging prints"""
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
    def entityIDlst(self):
        """:obj:`list` of :obj:`str`: list of all entity IDs that
        are part of at least one labelset"""
        return [i for i in self.entity if self.getNoccur(i) > 0]

    @property
    def labelIDlst(self):
        """:obj:`list` of :obj:`str`: list of all labelset names"""
        return self.lst

    def addLabelset(self, lst, labelID, labelInfo=None):
        """Add new labelset

        Args:
            lst(:obj:`list` of :obj:`str`): list of IDs of entiteis belong
                to the input label
            labelID(str): name of label
            labelInfo(str): description of label

        """
        self.addID(labelID, {} if labelInfo is None else {"Info": labelInfo})
        try:
            self.entity.update(lst)
        except Exception as e:
            # if entity list not updated successfully, pop the new labelset
            self.popID(labelID)
            raise e
        self.updateLabelset(lst, labelID)

    def popLabelset(self, labelID):
        """Pop a labelset and remove entities that no longer belong to any labelset"""
        self.resetLabelset(labelID)
        self.popID(labelID)

    def updateLabelset(self, lst, labelID):
        """Update existing labelset

        Take list of entities IDs and update current labelset with a label
        name matching `labelID`. Any ID in the input list `lst` that does
        not exist in the entity list will be added to the entity list. Increment
        the `Noccur` property of any newly added entites to the labelset by 1.

        Note: labelID must already existed, use `.addLabelset()` for adding
        new labelset

        Args:
            lst(:obj:`list` of :obj:`str`): list of entiteis IDs to be
                added to the labelset, can be redundant.

        Raises:
            TypeError: if `lst` is not `list` type or any element within `lst`
                is not `str` type

        """
        checkers.checkTypesInList("Entity list", str, lst)
        lbset = self.getLabelset(labelID)
        for ID in lst:
            if ID not in self.entity:
                self.entity.addID(ID)
            if ID not in lbset:
                lbset.update([ID])
                self.entity.setProp(ID, "Noccur", self.getNoccur(ID) + 1)

    def resetLabelset(self, labelID):
        """Reset an existing labelset to an empty set, and deecrement `Noccur` of
        all entites belonging to the labelset.

        Note: Any entity specific to the popped labelset is popped, which is
        indicated by the equality of properties of entity with default property
        values. The most important property is `Noccur`, the number of labelset
        an entity is presented in, with default value of 0 (see init), which implies
        that the entity is no longer present in any labelsets. The use of all
        default properfies for comparison to determine popping of entity is just
        for generalization purpose.

        """
        lbset = self.getLabelset(labelID)
        for ID in lbset:
            self.entity.setProp(ID, "Noccur", self.getNoccur(ID) - 1)
            if self.entity.getAllProp(ID) == self.entity.prop_default_val:
                self.entity.popID(ID)
        lbset.clear()

    def popEntity(self, ID):
        """Pop an entity from entity list, and also remove it from all labelsets.

        Note: Unlike `popLabelset`, if after removal, a labelset beomes empty, the
        labelset itself is NOT removed. This is for more convenient comparison of
        labelset sizes before and after filtering.

        """
        self.entity.popID(ID)
        for labelID in self.labelIDlst:
            self.getLabelset(labelID).difference_update([ID])

    def getInfo(self, labelID):
        """Return description of a labelset"""
        return self.getProp(labelID, "Info")

    def getLabelset(self, labelID):
        """Return set of entities associated with a label"""
        return self.getProp(labelID, "Labelset")

    def getNegative(self, labelID):
        """Return set of negative samples of a labelset

        Note:
            If negative samples not available, use complement of labelset

        """
        neg = self.getProp(labelID, "Negative")

        if neg == {None}:
            all_positives = set(
                [i for i in self.entity.map if self.getNoccur(i) > 0]
            )
            return all_positives - self.getLabelset(labelID)

        return neg

    def setNegative(self, lst, labelID):
        checkers.checkTypesInList("Negative entity list", str, lst)
        lbset = self.getLabelset(labelID)
        for ID in lst:
            self.entity._check_ID_existence(ID, True)
            if ID in lbset:
                # raise Exception(repr(ID), repr(labelID))
                raise IDExistsError(
                    f"Entity {ID!r} is positive in labelset {labelID!r}, "
                    f"cannot be set to negative"
                )
        self.setProp(labelID, "Negative", set(lst))

    def getNoccur(self, ID):
        """Return the number of labelsets in which an entity participates"""
        return self.entity.getProp(ID, "Noccur")

    def apply(self, filter_func, inplace=False):
        """Apply filter to labelsets, see `NLEval.label.Filter` for more info

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
        """Export as '.lsc' file

        Notes:
            '.lsc' is a csv file storing entity labels in matrix form, where
            first column is entity IDs, first and second rows correspond to
            label ID and label information respectively. If an entity 'i' is
            annotated with a label 'j', the corresponding 'ij' entry is marked
            as '1', else if it is considered a negative for that label, it is
            marked as '-1', otherwise it is '0', standing for neutral.

            entityIDmap is necessary since not all entities are guaranteed to
            be part of at least one labels.

        Input:
            fp(str): path to file to save, including file name, with/without extension.

        """
        entityIDlst = self.entityIDlst
        entityIDmap = {ID: idx for idx, ID in enumerate(entityIDlst)}
        labelIDlst = self.labelIDlst
        labelInfolst = [self.getInfo(labelID) for labelID in labelIDlst]
        mat = np.zeros((len(entityIDlst), len(labelIDlst)), dtype=int)

        for j, labelID in enumerate(labelIDlst):
            positive_set = self.getLabelset(labelID)
            negative_set = self.getNegative(labelID)

            for sign, labelset in zip(
                ["1", "-1"], [positive_set, negative_set]
            ):
                for entityID in labelset:
                    i = entityIDmap[entityID]
                    mat[i, j] = sign

        fp += "" if fp.endswith(".lsc") else ".lsc"
        with open(fp, "w") as f:
            # headers
            labelIDlst_str = "\t".join(labelIDlst)
            labelInfolst_str = "\t".join(labelInfolst)
            f.write(f"Label ID\t{labelIDlst_str}\n")
            f.write(f"Label Info\t{labelInfolst_str}\n")

            # annotations
            for i, entityID in enumerate(entityIDlst):
                indicator_string = "\t".join(map(str, mat[i]))
                f.write("{entityID}\t{indicator_string}\n")

    def export_gmt(self, fp):
        """Export as '.gmt' (Gene Matrix Transpose) file

        Input:
            fp(str): path to file to save, including file name, with/without extension.
        """
        fp += "" if fp.endswith(".gmt") else ".gmt"
        with open(fp, "w") as f:
            for labelID in self.labelIDlst:
                labelInfo = self.getInfo(labelID)
                labelset_str = "\t".join(self.getLabelset(labelID))
                f.write("{labelID}\t{labelInfo}\t{labelset_str}\n")

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
                ID, val = line.strip().split()
                if ID not in self.entity:
                    self.entity.addID(ID)
                self.entity.setProp(ID, prop_name, interpreter(val))

    @classmethod
    def from_gmt(cls, fp):
        """Load data from Gene Matrix Transpose `.gmt` file
        https://software.broadinstitute.org/cancer/software/gsea/wiki/index.php/Data_formats

        Args:
            fg(str): path to the `.gmt` file

        """
        lsc = cls()
        with open(fp, "r") as f:
            for line in f:
                labelID, labelInfo, *lst = line.strip().split("\t")
                lsc.addLabelset(lst, labelID, labelInfo)
        return lsc


class SplitLSC(BaseLSC):
    """Labelset collection with more functionality including negative selection and
    splitting utility to generate train/test split for each labelset"""

    def __init__(self):
        super(SplitLSC, self).__init__()
        self._valsplit = None
        self._filter_switch = False

    @property
    def valsplit(self):
        """:obj:`NLEval.valsplit.Base.BaseValSplit`: validation split
        generator used to generat train/test split for labelsets"""
        return self._valsplit

    @valsplit.setter
    def valsplit(self, obj):
        checkers.checkType("Validation split generator", Base.BaseValSplit, obj)
        self._valsplit = obj

    def train_test_setup(self, graph, prop_name=None, min_pos=10):
        """Setup training and testing IDs, filter labelsets based on train/test samples

        Args:
            prop_name(str): name of entity properties used for generating splits
            min_pos(int): minimum number of positive in both training and testing
                sets of a given labelset below which labelset is discarded. If
                `None` specified, no filtering will be done.

        """
        if self.valsplit is None:
            raise AttributeError(
                "'valsplit' not configured, "
                + "assign validation split generator first. "
                + "See `NLEval.valsplit` for more info."
            )

        num_labelsets = None
        while num_labelsets != len(self.labelIDlst):
            num_labelsets = len(self.labelIDlst)
            # print(num_labelsets)
            # labelIDset = set(self.labelIDlst)
            self.valsplit.train_test_setup(self.entity, graph.IDmap, prop_name)
            self.apply(
                Filter.LabelsetRangeFilterTrainTestPos(min_pos), inplace=True
            )
            # for i in labelIDset - set(self.labelIDlst):
            #     print(f"Pop {i}")

    def splitLabelset(self, labelID, entityIDlst=None):
        """Split up a labelset by training and testing sets

        Returns:
            A generator that yeilds train/test IDs and labels, see
            `NLEval.valsplit.Base.BaseValSplit.split` for more info.

        """
        if entityIDlst is None:
            entityIDlst = self.entityIDlst.copy()

        pos_ID_set = set(list(self.getLabelset(labelID))) & set(entityIDlst)
        neg_ID_set = set(list(self.getNegative(labelID))) & set(entityIDlst)

        ID_ary = np.array(list(pos_ID_set) + list(neg_ID_set))
        label_ary = np.zeros(len(ID_ary), dtype=bool)
        label_ary[: len(pos_ID_set)] = True
        return self.valsplit.split(ID_ary, label_ary)

    def export_splits(self, fp, graph):
        """Export (holdout) split information to npz file

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
            "Labelset collection splitter "
            + "(only support Holdout split now)",
            Base.BaseHoldout,
            self.valsplit,
        )
        valid = False if self.valsplit.valid_ID_ary is None else True
        self.valsplit.check_split_setup(valid)

        train_idx = graph.IDmap[self.valsplit.train_ID_ary]
        test_idx = graph.IDmap[self.valsplit.test_ID_ary]
        valid_idx = graph.IDmap[self.valsplit.valid_ID_ary] if valid else np.NaN

        y = np.zeros((graph.size, len(self.labelIDlst)), dtype=bool)
        for i, labelID in enumerate(self.labelIDlst):
            pos_ID_ary = np.array(list(self.getLabelset(labelID)))
            pos_idx_ary = graph.IDmap[pos_ID_ary]
            y[pos_idx_ary, i] = True

        np.savez(
            fp,
            y=y,
            train_idx=train_idx,
            valid_idx=valid_idx,
            test_idx=test_idx,
            IDs=graph.IDmap.lst,
            labelIDs=self.labelIDlst,
        )
