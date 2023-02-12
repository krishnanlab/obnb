import pprint

import pandas as pd

from nleval.data.annotation.base import BaseAnnotationData
from nleval.typing import List, Optional


class HumanPhenotypeOntologyAnnotation(BaseAnnotationData):
    """The Human Phenotype Ontology gene annotations.

    Annotations are retrieved from https://hpo.jax.org/app/

    The disease/trait annotations mainly come from two sources, namely, *OMIM*
    and *Orphanet*. By default, both sources are used. For more information,
    please refer to the HPO annotation
    `download <https://hpo.jax.org/app/data/annotations>`_ page

    **[Last updated: 2023-02-12]**

    Args:
        root: Root directory of the data.
        data_sources: List of evidene types to be considered. If not set,
            then use the default channels (OMIM and Orphanet).

    """

    annotation_url = "http://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt"
    annotation_file_name = "genes_to_phenotype.txt"
    annotation_file_zip_type = "none"

    def __init__(
        self,
        root: str,
        *,
        data_sources: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize GeneOntology annotation data object."""
        self._data_sources = data_sources
        super().__init__(root, **kwargs)

    @property
    def data_sources(self) -> List[str]:
        if self._data_sources is None:
            return [
                "mim2gene",  # OMIM
                "orphadata",  # Orphanet
            ]
        else:
            return self._data_sources

    def load_processed_data(self):
        path = self.raw_file_path(0)
        self.plogger.info(f"Loading raw annotation from {path}")

        # Load hpo gene annotation data
        # https://hpo.jax.org/app/data/annotations
        annot_df = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            header=0,
            names=[
                "gene_id",
                "gene_symbol",
                "term_id",
                "term_name",
                "freq_row",
                "freq_hpo",
                "info",
                "source",
                "disease_id",
            ],
        )
        annot_df["gene_id"] = annot_df["gene_id"].astype(str)

        # Select specified channels
        evidence_str = pprint.pformat(self.data_sources)
        self.plogger.info(f"Subsetting annotations to evidences:\n{evidence_str}")
        ind = annot_df["source"].isin(self.data_sources)
        self.plogger.info(f"{ind.sum():,} (out of {ind.shape[0]:,}) entries selected")
        annot_df = annot_df[ind]

        # Save attributes
        self.data = annot_df[["gene_id", "term_id"]].copy()
