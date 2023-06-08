from urllib.parse import urljoin

import pandas as pd
from obnb.data.annotation.base import BaseAnnotationData
from obnb.typing import List, Optional


class DISEASESAnnotation(BaseAnnotationData):
    """DISEASES disease gene annotations from the JensonLab.

    Disease gene associations are retrieved from diseases.jensenlab.org

    This is the ``integrated`` disease annotation channel from the Jensen Lab
    DISEASES annotation database, which combines evidences from *text-mining*,
    *knowledge*, and *experiment* channels. See  the
    `DISEASES <https://diseases.jensenlab.org/About>`_ webpage for more
    information

    """

    base_url = "https://download.jensenlab.org"
    annotation_channels = {
        "integrated_full",
        "textmining_full",
        "textmining_filtered",
        "knowledge_full",
        "knowledge_filtered",
        "experiments_full",
        "experiments_filtered",
    }
    annotation_file_zip_type = "none"

    def __init__(
        self,
        root: str,
        *,
        score_min: Optional[float] = 3,
        score_max: Optional[float] = None,
        channel: str = "integrated_full",
        **kwargs,
    ):
        """Initialize DisGeNET annotation data object."""
        if channel not in self.annotation_channels:
            raise KeyError(
                f"Unknown channel {channel!r}, available options are "
                f"{self.annotation_channels!r}",
            )

        self.channel = channel
        self.annotation_file_name = f"human_disease_{channel}.tsv"
        self.annotation_url = urljoin(self.base_url, self.annotation_file_name)

        self.score_min = score_min
        self.score_max = score_max
        super().__init__(root, **kwargs)

    def get_column_names(self) -> List[str]:
        """Get annotation table column names for the selected channel.

        All channels start with four columns: ``gene_id``, ``gene_name``,
        ``term_id``, and ``term_name``. The extended columns are specific to
        each type of channels:

        - ``integrated`` contains an additional confidence score column.
        - ``textmining`` contains z-score, the confidence score, and the URL to
          the abstracts view.
        - ``knowledge`` contains the source database, evidence type, and the
          confidence score.
        - ``experiments`` continas the source data base, source score, and the
          confidence score.

        See the `download page <https://diseases.jensenlab.org/Downloads>`_
        for more information.

        NOTE:
            The *confidence scores* (``score``) are normalized across the
            DISEASES database. In brief, it is a "5-star" system, where the
            disease-gene associations with highest confidence are assigned with
            full socre of five. The confidence scores from the ``textmining``
            channel are computed as half of the z-score and are capped at the
            value of four. On the other hand, the confidence scores from the
            ``experiments`` channel were calibrated using the gold-standard
            benchmarking scheme, where the gold-standards are derived from
            curated annotations, i.e., the ``knowledge`` channel. All
            annotations from the ``knowledge`` channel are scored 4-5 stars.

        """
        names = ["gene_id", "gene_name", "term_id", "term_name"]

        if self.channel.startswith("integrated"):
            names += ["score"]

        elif self.channel.startswith("textmining"):
            names += ["z_score", "score", "abstracts_url"]

        elif self.channel.startswith("knowledge"):
            names += ["source_db", "evidence_type", "score"]

        elif self.channel.startswith("experiments"):
            names += ["source_db", "source_score", "score"]

        else:
            raise ValueError(
                f"Unknown channel {self.channel}, should've been caught at __init__",
            )

        self.plogger.info(f"Column names for {self.channel}: {names}")

        return names

    def load_processed_data(self):
        path = self.raw_file_path(0)
        self.plogger.info(f"Loading raw annotation from {path}")
        annot_df = pd.read_csv(path, sep="\t", header=0, names=self.get_column_names())

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
