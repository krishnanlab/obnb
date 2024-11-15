from typing import Iterable, Iterator, List, Tuple

import numpy

from obnb.label.collection import LabelsetCollection
from obnb.label.split.base import BaseSplit
from numpy import ndarray


class ByTermSplit(BaseSplit):
    """
    Produces splits based on an explicit list of terms. Genes
    which match each term will be placed in the split corresponding
    to that term.

    A split with a single term '*' will act as a catch-all for any
    genes that weren't matched by any of the other splits. This would
    allow you to, e.g., only retain a specific set of genes in the
    training set, and place all others in the test set.

    Note that if the '*' split is not provided, any genes that don't
    match any of the other splits will not be present in the returned
    splits at all.
    """

    def __init__(
            self, labelset: LabelsetCollection,
            split_terms: Iterable[Iterable[str]],
            exclusive: bool = False
        ) -> None:
        """
        Initialize ByTermSplit object with reference labels and terms into
        which to create splits.

        Args:
            labelset: LabelsetCollection object containing terms for each
                gene ID.
            split_terms: a nested collection. The first level of nesting
                indicates the splits; the second level within each identifies
                terms that should be matched to place a gene in that split.
            exclusive: if True, a gene can occur only once across all the
                splits; it will belong to the first split in which it occurs.
        """

        self.labelset = labelset
        self.split_terms = [set(x) for x in split_terms]
        self.exclusive = exclusive

        # verify that there's only one catch-all split
        if sum(1 for x in self.split_terms if x == {"*"}) > 1:
            raise ValueError("Only one catch-all '*' split is allowed")

        # convert labelset into a dataframe where one can search for
        # the terms associated with each gene ID like so:
        # self.long_df[self.long_df["Value"] == str(gene_id)]["Name"]
        df = self.labelset.to_df()
        self.long_df = df.melt(
            id_vars=["Name"],
            value_vars=df.columns.difference(["Info", "Size", "Name"]),
            value_name="Value"
        ).dropna(subset=["Value"])

        # group gene id and aggregate terms into a set, which makes
        # it faster to retrieve all the terms for a given gene ID
        self.gene_id_to_terms = (
            self.long_df.groupby("Value")["Name"]
            .apply(set)
            .reset_index()
            .rename(columns={"Value": "GeneID", "Name": "Terms"})
        )

        super().__init__()

    def __call__(self, ids: List[str], y: ndarray) -> Iterator[Tuple[ndarray, ...]]:
        """
        For each gene ID, look up the term it's associated with
        in the labelset, and place it in the corresponding split.

        Returns as many splits as there are elements in the split_terms
        tuple.
        """

        # alias field to shorten the code below
        gdf = self.gene_id_to_terms

        # for each split, filter to the gene IDs that have at least one
        # term in the split
        result = [
            (
                set([
                    id for id in ids
                    if gdf[gdf["GeneID"] == str(id)]["Terms"].values[0] & terms
                ]) if terms != {"*"} else None
            )
            for terms in self.split_terms
        ]

        # if one of the resulting splits ended up as 'None', we need to
        # fill in that split with any gene that wasn't matched by any of
        # the other splits
        for idx in range(len(result)):
            if result[idx] is None:
                result[idx] = set([
                    id for id in ids
                    if not any(
                        gdf[gdf["GeneID"] == str(id)]["Terms"].values[0] & terms
                        for terms in self.split_terms
                    )
                ])

        if self.exclusive:
            # if exclusive, remove genes in the current split that occurred
            # in any previous split
            # (we skip the first split since there's nothing with which to
            # compare it)
            for idx in range(1, len(result)):
                result[idx] = result[idx] - set.union(*result[:idx])

        # yield it in the format returned by other splitters, e.g. a tuple of
        # numpy arrays. we cast to list because leaving it as a set would cause
        # numpy.asarray() to create an array with a single element, the set,
        # rather than an array with the elements of the list
        yield tuple([ numpy.asarray(list(x)) for x in result ])
