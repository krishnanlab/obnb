from functools import lru_cache

import numpy as np

from NLEval.exception import IDExistsError
from NLEval.graph import OntologyGraph
from NLEval.label.filters.base import BaseFilter
from NLEval.typing import Dict, Iterator, List, Optional, Set, Splitter, Tuple
from NLEval.util import checkers, idhandler


class LabelsetCollection(idhandler.IDprop):
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
        """Initialize LabelsetCollection object."""
        super().__init__()

    def reset(self):
        """Reset all labelsets and entities."""
        super().reset()
        self.entity = idhandler.IDprop()
        self.entity.new_property("Noccur", 0, int)
        self.new_property("Info", "NA", str)
        self.new_property("Labelset", set(), set)
        self.new_property("Negative", {None}, set)

    def __hash__(self):
        """Hash a LabelsecCollection object.

        Hash using the followings
        - Entity IDs and number of occurances
        - Labelsets along with negatives and their information

        Note:
            This is used for lru cach that decorates the split method.

        """
        eids = (*self.entity,)
        enoccur = (*self.entity._prop["Noccur"],)
        linfo = (*self._prop["Info"],)
        llbs = (*map(frozenset, self._prop["Labelset"]),)
        lneg = (*map(frozenset, self._prop["Negative"]),)
        return hash(eids + enoccur + linfo + llbs + lneg)

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

    def stats(self) -> str:
        """Print basic stats for the labelset collection."""
        sizes = self.sizes
        return (
            f"Number of labelsets: {len(self)}\n"
            f"max: {max(sizes)}\n"
            f"min: {min(sizes)}\n"
            f"med: {np.median(sizes):.2f}\n"
            f"avg: {np.mean(sizes):.2f}\n"
            f"std: {np.std(sizes):.2f}\n"
        )

    def items(self) -> Iterator[Tuple[int, Set[str]]]:
        """Yield label name and the corresponding label set."""
        for label in self:
            yield label, self.get_labelset(label)

    @property
    def sizes(self) -> List[int]:
        """Sizes of the labelsets."""
        return [len(labelset) for _, labelset in self.items()]

    @property
    def entity_ids(self):
        """List of all entity IDs that are part of at least one labelset."""
        return [i for i in self.entity if self.get_noccur(i) > 0]

    @property
    def label_ids(self):
        """:obj:`list` of :obj:`str`: list of all labelset names."""
        return self.lst

    def new_labelset(self, label_id, label_info=None):
        """Create a new empty labelset."""
        self.add_id(
            label_id,
            {} if label_info is None else {"Info": label_info},
        )

    def add_labelset(self, lst, label_id, label_info=None):
        """Add a new labelset.

        Args:
            lst(:obj:`list` of :obj:`str`): list of IDs of entiteis belong
                to the input label
            label_id(str): name of label
            label_info(str): description of label

        """
        self.new_labelset(label_id, label_info=label_info)
        try:
            self.entity.update(lst)
        except Exception as e:
            # if entity list not updated successfully, pop the new labelset
            self.pop_id(label_id)
            raise e
        self.update_labelset(lst, label_id)

    def pop_labelset(self, label_id):
        """Pop a labelset.

        Note:
            This also removes any entity that longer belongs to any labelset.

        """
        self.reset_labelset(label_id)
        self.pop_id(label_id)

    def update_labelset(self, lst, label_id):
        """Update an existing labelset.

        Take list of entities IDs and update current labelset with a label
        name matching `label_id`. Any ID in the input list `lst` that does
        not exist in the entity list will be added to the entity list.
        Increment the `Noccur` property of any newly added entites to the
        labelset by 1.

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
                self.entity.set_property(
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
            self.entity.set_property(
                entity_id,
                "Noccur",
                self.get_noccur(entity_id) - 1,
            )
            if (
                self.entity.get_all_properties(entity_id)
                == self.entity.prop_default_val
            ):
                self.entity.pop_id(entity_id)
        lbset.clear()

    def pop_entity(self, entity_id):
        """Pop an entity from entity list and remove it from all labelsets.

        Note:
            Unlike `pop_labelset`, if after removal, a labelset beomes empty,
            the labelset itself is NOT removed. This is for more convenient
            comparison of labelset sizes before and after filtering.

        """
        self.entity.pop_id(entity_id)
        for label_id in self.label_ids:
            self.get_labelset(label_id).difference_update([entity_id])

    def get_noccur(self, entity_id):
        """Return the number of labelsets in which an entity participates."""
        return self.entity.get_property(entity_id, "Noccur")

    def get_info(self, label_id):
        """Return description of a labelset."""
        return self.get_property(label_id, "Info")

    def get_labelset(self, label_id):
        """Return set of entities associated with a label."""
        return self.get_property(label_id, "Labelset")

    def get_negative(self, label_id):
        """Return set of negative samples of a labelset.

        Note:
            If negative samples not available, use complement of labelset

        """
        neg = self.get_property(label_id, "Negative")

        if neg == {None}:
            all_positives = {i for i in self.entity.map if self.get_noccur(i) > 0}
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
        self.set_property(label_id, "Negative", set(lst))

    def get_y(
        self,
        target_ids: Tuple[str, ...],
        labelset_name: Optional[str] = None,
    ) -> np.ndarray:
        """Return the y matrix.

        Args:
            target_ids: Tuple of entity ids used to order the rows.
            labelset_name: A specific labelset to use, if not set, use all the
                labelests (default: :obj:`None`).

        """
        # TODO: Clean up this, reduce redundancy with split
        target_idmap = {j: i for i, j in enumerate(target_ids)}
        entity_idmap = {j: i for i, j in enumerate(self.entity_ids)}
        to_target_idx = np.array([target_idmap[i] for i in self.entity_ids])

        if labelset_name is None:
            labelsets = list(map(self.get_labelset, self.label_ids))
            y = np.zeros((len(self.entity_ids), len(labelsets)), dtype=bool)
            for i, labelset in enumerate(labelsets):
                y[list(map(entity_idmap.get, labelset)), i] = True
        else:
            labelset = self.get_labelset(labelset_name)
            y = np.zeros(len(self.entity_ids), dtype=bool)
            y[list(map(entity_idmap.get, labelset))] = True

        if labelset_name is not None or len(y.shape) == 1:
            y_out = np.zeros(len(target_ids), dtype=bool)
        else:
            y_out = np.zeros((len(target_ids), y.shape[1]), dtype=bool)
        y_out[to_target_idx] = y

        return y_out

    @lru_cache  # noqa: B019
    def split(  # TODO: Reduce cylic complexity..
        self,
        splitter: Splitter,
        property_name: Optional[str],
        target_ids: Optional[Tuple[str, ...]] = None,
        labelset_name: Optional[str] = None,
        mask_names: Optional[Tuple[str, ...]] = None,
        consider_negative: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Split the entities based on the labelsets.

        Args:
            splitter: A splitter function that split the entities based on
                their labels and optionally the an entity.
            property_name: Name of the entity property used by the splitter. Do
                not use any property and use only the labels if set to
                :obj:`None`.
            target_ids: Tuple of entity ids for the output masks and label
                vector to align with. Use ``self.entity_ids`` if not specified.
            labelset_name: Indicate which specific labelset to split. Split
                based on all available sets if not specified.
            mask_names: Name of maskes for splits generated by the splitter. If
                not specified, use ``('train', 'test')`` when the splitter
                generates two splits and use ``('train', 'val', 'test')`` when
                the splitter generates three splits.
            consider_negative (bool): Only use annotated negatives and remove
                neutral data points where we do not know for sure they are
                negatives (default: :obj:`False`).

        Note:
            The ``consider_negative`` option currently only works when one
            explicitly specify the ``labelset_name``. In the future, might also
            support this option with multiple labelsets.

        Raises:
            ValueError: If the length of the specified ``mask_names`` does not
                match that of the number of splits generated by the splitter,
                or if the number of splits generated by the splitter is neither
                two or three but ``mask_names`` is not specified. Or the
                specified ``target_ids`` does not catain all of ``entity_ids``.
            IDNotExistError: If the specified ``labelset_name`` does not exist
                or the specified ``property_name` does not exist.

        """
        if target_ids is not None:
            target_idset = set(target_ids)
            # Check if target_ids contains all entity_ids
            for entity_id in self.entity_ids:
                if entity_id not in target_idset:
                    raise ValueError(
                        f"target_ids must contain all of entity_ids, "
                        f"but {entity_id!r} is missing",
                    )
        else:
            target_ids = self.entity_ids

        # Prepare mapping from entity id to target index
        target_idmap = {j: i for i, j in enumerate(target_ids)}
        # TODO: fix this entity_idmap...
        entity_idmap = {j: i for i, j in enumerate(self.entity_ids)}
        to_target_idx = np.array([target_idmap[i] for i in self.entity_ids])

        # Prepare 'x' and 'y' and pass to splitter
        if labelset_name is None:
            labelsets = list(map(self.get_labelset, self.label_ids))
            y = np.zeros((len(self.entity_ids), len(labelsets)), dtype=bool)
            for i, labelset in enumerate(labelsets):
                y[list(map(entity_idmap.get, labelset)), i] = True
        else:
            labelset = self.get_labelset(labelset_name)
            y = np.zeros(len(self.entity_ids), dtype=bool)
            y[list(map(entity_idmap.get, labelset))] = True

        if property_name is not None:
            x = np.array(
                [self.entity.get_property(i, property_name) for i in self.entity_ids],
            )
        else:
            x = y

        # Iterate over splits generated by splitter and align with target_ids
        splits = list(zip(*[*splitter(x, y)]))
        split_size = len(splits)
        if mask_names is not None:
            if split_size != len(mask_names):
                raise ValueError(
                    f"{len(mask_names)} mask names specified: {mask_names!r}, "
                    f"but got {split_size} from the splitter.",
                )
        elif split_size == 1:
            mask_names = ("test",)
        elif split_size == 2:
            mask_names = ("train", "test")
        elif split_size == 3:
            mask_names = ("train", "val", "test")
        else:
            raise ValueError(
                f"Default mask_names expected split size of 2 or 3, "
                f"got {split_size} instead.",
            )

        masks = {}
        for mask_name, split in zip(mask_names, splits):
            mask = np.zeros((len(target_ids), len(split)), dtype=bool)
            for i, j in enumerate(split):
                mask[to_target_idx[j], i] = True
            masks[mask_name] = mask

        if consider_negative:
            if labelset_name is None:
                # TODO: option for consider negatives with multiple labelsets
                raise ValueError(
                    "Considering multiple labelsets with negatives is not "
                    "supported currently, specify labelset_name to pick one "
                    "single labelset to consider negatives.",
                )
            else:
                positives = self.get_labelset(labelset_name)
                negatives = self.get_negative(labelset_name)
                to_remove = set(self.entity_ids) - (positives | negatives)
                if len(to_remove) > 0:  # skip if nothing to be removed
                    target_idx_to_remove = to_target_idx[
                        list(map(entity_idmap.get, to_remove))
                    ]
                    for mask in masks.values():
                        mask[target_idx_to_remove] = False

        # Map back to the order of target_ids
        if labelset_name is not None or len(y.shape) == 1:
            y_out = np.zeros(len(target_ids), dtype=bool)
        else:
            y_out = np.zeros((len(target_ids), y.shape[1]), dtype=bool)
        y_out[to_target_idx] = y

        return y_out, masks

    def apply(
        self,
        filter_func,
        inplace: bool = False,
        progress_bar: bool = False,
    ):
        """Apply filter to labelsets.

        See `NLEval.label.filters` for more info.

        Args:
            filter_func
            inplace (bool): whether or not to modify original object, if True,
                then apply the filter directly on the original object;
                otherwise, apply the filter on a copy of the original object
                and return that object (default: :obj:`False`).
            progress_bar (bool): whether or not to display progress bar for
                filtering (default: :obj:`False`).

        Returns:
            Labelset coolection object after filtering.

        """
        checkers.checkType(
            "filters",
            BaseFilter,
            filter_func,
        )
        checkers.checkType("inplace", bool, inplace)
        obj = self if inplace else self.copy()
        filter_func(obj, progress_bar)
        return obj

    def iapply(self, filter_func, progress_bar: bool = False):
        """Apply filter to labelsets inplace.

        This is a shortcut for calling self.apply(filter_func, inplace=True).

        """
        self.apply(filter_func, inplace=True, progress_bar=progress_bar)

    def export(self, path):
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
            path(str): path to file to save, including file name, with/without
                extension.

        """
        entity_ids = self.entity_ids
        entity_idmap = {entity_id: idx for idx, entity_id in enumerate(entity_ids)}
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

        path = path if path.endswith(".lsc") else path + ".lsc"
        with open(path, "w") as f:
            # headers
            label_ids = "\t".join(label_ids)
            label_info_str = "\t".join(label_info_list)
            f.write(f"Label ID\t{label_ids}\n")
            f.write(f"Label Info\t{label_info_str}\n")

            # annotations
            for i, entity_id in enumerate(entity_ids):
                indicator_string = "\t".join(map(str, mat[i]))
                f.write(f"{entity_id}\t{indicator_string}\n")

    def export_gmt(self, path):
        """Export self as a '.gmt' (Gene Matrix Transpose) file.

        Input:
            path(str): path to file to save, including file name, with/without
                extension.

        """
        path += "" if path.endswith(".gmt") else ".gmt"
        with open(path, "w") as f:
            for label_id in self.label_ids:
                label_info = self.get_info(label_id)
                labelset_str = "\t".join(self.get_labelset(label_id))
                f.write(f"{label_id}\t{label_info}\t{labelset_str}\n")

    def load_entity_properties(
        self,
        path,
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
            path(str): path to the entity properties file.
            default_val: default value of property of an entity if not
                specified.
            default_type(type): default type of the property.
            interpreter: function to transfrom property value from string to
                some other value

        """
        # TODO: option to skip non-existing entities
        self.entity.new_property(prop_name, default_val, default_type)
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if (i < skiprows) | line.startswith(comment):
                    continue
                entity_id, val = line.strip().split()
                if entity_id not in self.entity:
                    self.entity.add_id(entity_id)
                self.entity.set_property(
                    entity_id,
                    prop_name,
                    interpreter(val),
                )

    def read_ontology_graph(
        self,
        graph: OntologyGraph,
        min_size: int = 10,
        namespace: Optional[str] = None,
    ):
        """Load labelset collection from an annotated ontology graph.

        Args:
            graph (OntologyGraph): The annotated ontology graph to be read.
            min_size (int): Minimum number of positive examples in order to be
                loaded as a label set (default: 10).
            namespace (str, optional): If set, only load terms that are
                inherited from the term specified in as namespace, otherwise
                load all terms (default: :obj:`None`).

        """
        with graph.cache_on_static():
            for label_id in graph.node_ids:
                if namespace is None or namespace in graph.ancestors(label_id):
                    label_info = graph.get_node_name(label_id)
                    label_set = graph.get_node_attr(label_id) or []
                    if len(label_set) >= min_size:
                        self.add_labelset(label_set, label_id, label_info)

    @classmethod
    def from_ontology_graph(
        cls,
        graph: OntologyGraph,
        min_size: int = 10,
        namespace: Optional[str] = None,
    ):
        """Construct LabelsetCollection object from an annotated ontology."""
        lsc = cls()
        lsc.read_ontology_graph(graph, min_size=min_size, namespace=namespace)
        return lsc

    def read_gmt(self, path: str, sep: str = "\t", reload: bool = False):
        """Load data from Gene Matrix Transpose `.gmt` file.

        Args:
            path: path to the `.gmt` file.
            sep: seperator used in the GMT file.
            reload: Remove existing labelsets before loading if set to True.

        """
        if reload:
            self.reset()
        with open(path, "r") as f:
            for line in f:
                label_id, label_info, *lst = line.strip().split(sep)
                self.add_labelset(lst, label_id, label_info)

    @classmethod
    def from_gmt(cls, path: str, sep: str = "\t"):
        """Construct LabelsetCollection object from GMT file.

        Args:
            path: path to the `.gmt` file.
            sep: seperator used in the GMT file.

        """
        lsc = cls()
        lsc.read_gmt(path, sep=sep)
        return lsc

    @classmethod
    def from_dict(cls, input_dict: Dict[str, str]):
        """Load data from entity label dictionary.

        Args:
            input_dict (:obj:`dict` from :obj:`str` to :obj:`str): A dictionary
                mapping from entities to their unique label IDs.

        """
        lsc = cls()
        for entity, label_id in input_dict.items():
            if label_id not in lsc:
                lsc.new_labelset(label_id)
            lsc.update_labelset([entity], label_id)
        return lsc
