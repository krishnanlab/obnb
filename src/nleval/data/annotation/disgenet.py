import pprint

import pandas as pd
from obnb.data.annotation.base import BaseAnnotationData
from obnb.typing import List, Optional


class DisGeNETAnnotation(BaseAnnotationData):
    """DisGeNET disease gene annotations.

    Disease gene associations are retrieved from disgenet.org.

    There are four different categories of annotation sources from DisGeNET (
    see below). By default, we only use the *Curated* and the *Inferred* data
    sources. User can change the sources by passing the list of sources to the
    `data_sources` argument. (Note: ~70% of the disease-gene annotations in
    DisGeNET are only available in the *literature* data source). See the
    DisGeNET `data sources <https://www.disgenet.org/dbinfo>`_ documentation
    page for more information.

    - Curated (``CURATED``):
        - ``CGI`` Caner Genome Interpreter
        - ``CLINGEN`` Clinical Genome Resource
        - ``CTD_human`` Comparative Toxicogenomics Database (Human)
        - ``GENOMICS_ENGLAND`` Genomics England PanelApp
        - ``ORPHANET`` Orphan drugs and rare diseases
        - ``PSYGENET`` Psychiatric disorders gene association network
        - ``CLINVAR`` ClinVar disease-gene information with supported evidences

    - Inferred (``INFERRED``):
        - ``HPO`` Human Phenotype Ontology
        - ``UNIPROT`` UniProt/SwissProt database
        - ``GWASCAT`` GWAS Catalog curated SNPs (p-val < 1e-6)
        - ``GWASDB`` GWASdb (p-val < 1e-6)

    - Animal models (``ANIMAL``):
        - ``CTD_mouse`` Comparative Toxicogenomics Database (Mouse)
        - ``CTD_rat`` Comparative Toxicogenomics Database (Rat)
        - ``MGD`` Mouse Genome Database
        - ``RGD`` Rat Genome Database

    - Literature (``LITERATURE``):
        - ``BEFREE`` Disease-gene association extracted from MEDLINE using BeFree
        - ``LHGDN`` Literature derived human disease network

    **[Last updated: 2023-01-14]**

    Args:
        root: Root directory of the data.
        data_sources: List of evidence types to be considered. If not set,
            then use the default channels (curated and inferred evidences).
        dsi_min: Minimum value of ``DSI`` below which the annotations are removed.
        dsi_max: Maximum value of ``DSI`` above which the annotations are removed.
        dpi_min: Minimum value of ``DPI`` below which the annotations are removed.
        dpi_max: Maximum value of ``DPI`` above which the annotations are removed.

    Notes:
        ``DSI`` and ``DPI`` stands for *Disease Specificity Index* and
        *Disease Pleiotropy Index*. The two metrics measure how specific a gene
        is associated to a particular disease (vs. being associated to many
        diseases) and how pleiotropic a gene is (i.e., does the gene contribute
        to a wide variety of disease types, according to MeSH disease classes).
        The exact definitions of ``DSI`` and ``DPI`` can be found on in the
        DisGeNET `documentation <https://www.disgenet.org/dbinfo>`_ webpage.

    """

    annotation_file_name = "all_gene_disease_associations.tsv"
    annotation_url = "https://www.disgenet.org/static/disgenet_ap1/files/downloads/all_gene_disease_associations.tsv.gz"

    def __init__(
        self,
        root: str,
        *,
        data_sources: Optional[List[str]] = None,
        dsi_min: Optional[float] = None,
        dsi_max: Optional[float] = None,
        dpi_min: Optional[float] = None,
        dpi_max: Optional[float] = None,
        **kwargs,
    ):
        """Initialize DisGeNET annotation data object."""
        self._data_sources = data_sources
        self.dsi_min = dsi_min
        self.dsi_max = dsi_max
        self.dpi_min = dpi_min
        self.dpi_max = dpi_max
        super().__init__(root, **kwargs)

    @property
    def data_sources(self) -> List[str]:
        if self._data_sources is None:
            return [
                # Curated
                "CGI",
                "CLINGEN",
                "CTD_human",
                "GENOMICS_ENGLAND",
                "ORPHANET",
                "PSYGENET",
                "UNIPROT",
                # Inferred
                "CLINVAR",
                "GWASCAT",
                "GWASDB",
                "HPO",
            ]
        else:
            return self._data_sources

    def load_processed_data(self):
        path = self.raw_file_path(0)
        self.plogger.info(f"Loading raw annotation from {path}")
        annot_df = pd.read_csv(path, sep="\t")

        # Select specified channels
        evidence_str = pprint.pformat(self.data_sources)
        self.plogger.info(f"Subsetting annotations to evidences:\n{evidence_str}")
        annot_df = annot_df[
            (
                annot_df.source.str.split(";", expand=True)
                .isin(self.data_sources)
                .any(axis=1)
            )
        ]

        # Filter by DSI and DPI scores
        if self.dsi_max is not None:
            self.plogger.info(f"Removing annotations above DSI: {self.dsi_max}")
            annot_df = annot_df[annot_df["DSI"] <= self.dsi_max]
        if self.dsi_min is not None:
            self.plogger.info(f"Removing annotations below DSI: {self.dsi_min}")
            annot_df = annot_df[annot_df["DSI"] >= self.dsi_min]
        if self.dpi_max is not None:
            self.plogger.info(f"Removing annotations above DPI: {self.dpi_max}")
            annot_df = annot_df[annot_df["DPI"] <= self.dpi_max]
        if self.dpi_min is not None:
            self.plogger.info(f"Removing annotations below DPI: {self.dpi_min}")
            annot_df = annot_df[annot_df["DPI"] >= self.dpi_min]

        # Select relevant columns and rename to standardized column names
        annot_df = annot_df[["geneId", "diseaseId"]].reset_index(drop=True)
        annot_df.columns = ["gene_id", "term_id"]
        # Specify id prefixes
        annot_df["gene_id"] = annot_df["gene_id"].astype(str)
        annot_df["term_id"] = "UMLS:" + annot_df["term_id"].astype(str).values

        # Save attributes
        self.data = annot_df.copy()
