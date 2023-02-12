import pandas as pd

from nleval.data.annotation.base import BaseAnnotationData
from nleval.typing import Optional


class DISEASESAnnotation(BaseAnnotationData):
    """DISEASES disease gene annotations from the JensonLab.

    Disease gene associations are retrieved from diseases.jensenlab.org

    This is the ``integrated`` disease annotation channel from the Jensen Lab
    DISEASES annotation database, which combines evidences from *text-mining*,
    *knowledge*, and *experiment* channels. See  the
    `DISEASES <https://diseases.jensenlab.org/About>`_ webpage for more
    information

    """

    annotation_url = "https://download.jensenlab.org/human_disease_integrated_full.tsv"
    annotation_file_name = "human_disease_integrated_full.tsv"
    annotation_file_zip_type = "none"

    def __init__(
        self,
        root: str,
        *,
        score_min: Optional[float] = 3,
        score_max: Optional[float] = None,
        **kwargs,
    ):
        """Initialize DisGeNET annotation data object."""
        self.score_min = score_min
        self.score_max = score_max
        super().__init__(root, **kwargs)

    def load_processed_data(self):
        path = self.raw_file_path(0)
        self.plogger.info(f"Loading raw annotation from {path}")
        rename_map = {0: "gene_id", 2: "term_id", 4: "score"}
        annot_df = pd.read_csv(path, sep="\t", header=None).rename(columns=rename_map)

        # Filter by disease-gene association score
        if self.score_max is not None:
            self.plogger.info(f"Removing annotations above score: {self.score_max}")
            annot_df = annot_df[annot_df["score"] <= self.score_max]
        if self.score_min is not None:
            self.plogger.info(f"Removing annotations below score: {self.score_min}")
            annot_df = annot_df[annot_df["score"] >= self.score_min]

        # XXX: currently only use DOID terms, need to find a way to map
        # ICD10CM terms in the future
        ind = annot_df["term_id"].str.startswith("DOID:")
        annot_df = annot_df[ind]
        self.plogger.info(f"Using DOID terms: {ind.sum():,} out of {ind.shape[0]:,}")

        # Convert gene ids
        gene_id_converter = self.get_gene_id_converter()
        gene_id_converter.map_df(annot_df, "gene_id")

        # Save attributes
        self.data = annot_df[["gene_id", "term_id"]].copy()
