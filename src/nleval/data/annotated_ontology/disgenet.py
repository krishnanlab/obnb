import pandas as pd
from tqdm import tqdm

from nleval.data.annotated_ontology.base import BaseAnnotatedOntologyData
from nleval.exception import IDNotExistError
from nleval.graph import OntologyGraph
from nleval.label import LabelsetCollection
from nleval.label.filters import Compose, LabelsetNonRedFilter, LabelsetRangeFilterSize
from nleval.typing import List, Mapping, Optional, Union
from nleval.util.logger import display_pbar


class DisGeNET(BaseAnnotatedOntologyData):
    """DisGeNET disease gene annotations.

    Disease gene associations are retrieved from disgenet.org and then mapped
    to the disease ontology from obofoundry.org. The annotations are propagated
    upwards the ontology. Then, several filters are applied to reduce the
    redundancies between labelsets (disease genes):

    - Disease specificity index (DSI) filter
    - Max size filter
    - Min size filter
    - Jaccard index filter
    - Overlap coefficient filter

    """

    CONFIG_KEYS: List[str] = BaseAnnotatedOntologyData.CONFIG_KEYS + [
        "dsi_threshold",
        "data_sources",
    ]
    ontology_url = "http://purl.obolibrary.org/obo/mondo.obo"
    annotation_url = "https://www.disgenet.org/static/disgenet_ap1/files/downloads/all_gene_disease_associations.tsv.gz"
    ontology_file_name = "mondo.obo"
    annotation_file_name = "all_gene_disease_associations.tsv"

    def __init__(
        self,
        root: str,
        dsi_threshold: float = 0.5,
        min_size: int = 10,
        max_size: int = 600,
        overlap: float = 0.7,
        jaccard: float = 0.5,
        data_sources: Union[List[str], str] = "default",
        gene_id_converter: Optional[Union[Mapping[str, str], str]] = None,
        **kwargs,
    ):
        """Initialize the DisGeNET data object."""
        self.dsi_threshold = dsi_threshold
        self.min_size = min_size
        self.max_size = max_size
        self.jaccard = jaccard
        self.overlap = overlap
        self._data_sources = data_sources
        super().__init__(root, gene_id_converter=gene_id_converter, **kwargs)

    @property
    def data_sources(self) -> List[str]:
        if self._data_sources == "default":
            return [
                # Curated
                "CGI",  # Cancer Genome Interpreter
                "CLINGEN",  # Clinical Genome Resource
                "CTD_human",  # Comparative Toxicogenomics Database (Human)
                "GENOMICS_ENGLAND",  # Genomics England PanelApp
                "ORPHANET",  # Orphan drugs and rare diseases
                "PSYGENET",  # Psychiatric disorders gene association network
                "UNIPROT",  # UniProt/SwissProt data base
                # Inferred
                "CLINVAR",  # ClinVar disease-gene info with supported evidences
                "GWASCAT",  # GWAS Catalog curated SNPs (p-val < 1e-6)
                "GWASDB",  # GWASdb (p-val < 1e-6)
                "HPO",  # Human Phenotype Ontology
            ]
        else:
            return self._data_sources  # type: ignore

    @property
    def _default_pre_transform(self):
        return Compose(
            LabelsetRangeFilterSize(max_val=self.max_size),
            LabelsetNonRedFilter(self.jaccard, self.overlap),
            LabelsetRangeFilterSize(min_val=self.min_size),
            log_level=self.log_level,
        )

    def process(self):
        g = OntologyGraph()
        umls_to_mondo = g.read_obo(self.ontology_file_path, xref_prefix="UMLS")

        annot_df = pd.read_csv(self.annotation_file_path, sep="\t")
        annot_df = annot_df[
            (
                annot_df.source.str.split(";", expand=True)
                .isin(self.data_sources)
                .any(axis=1)
            )
            & (annot_df["DSI"] >= self.dsi_threshold)
        ]

        enable_pbar = display_pbar(self.log_level)
        pbar = tqdm(annot_df[["geneId", "diseaseId"]].values, disable=not enable_pbar)
        pbar.set_description("Annotating MONDOs")
        for gene_id, disease_id in pbar:
            for mondo in umls_to_mondo[disease_id]:
                try:
                    g._update_node_attr_partial(mondo, str(gene_id))
                except IDNotExistError:
                    self.plogger.debug(
                        f"Skipping {disease_id}({mondo})-{gene_id} because "
                        f"{mondo} is not available in the DO graph.",
                    )
        g._update_node_attr_finalize()

        # Propagate annotations and show progress
        g.propagate_node_attrs(pbar=enable_pbar)

        lsc = LabelsetCollection.from_ontology_graph(g, min_size=self.min_size)
        lsc.export_gmt(self.processed_file_path(0))
