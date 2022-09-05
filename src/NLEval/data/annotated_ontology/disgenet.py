import gzip
import os.path as osp

import pandas as pd
import requests
from tqdm import tqdm

from NLEval.data.annotated_ontology.base import BaseAnnotatedOntologyData
from NLEval.exception import IDNotExistError
from NLEval.graph import OntologyGraph
from NLEval.label import LabelsetCollection
from NLEval.label.filters import (
    Compose,
    LabelsetNonRedFilterJaccard,
    LabelsetNonRedFilterOverlap,
    LabelsetRangeFilterSize,
)
from NLEval.typing import List
from NLEval.util.logger import display_pbar


class DisGeNet(BaseAnnotatedOntologyData):
    """DisGeNet disease gene annotations.

    Disease gene associations are retreived from disgenet.org and then mapped
    to the disease ontology from obofoundry.org. The annotations are propagated
    upwards the ontology. Then, several filters are applied to reduce the
    redundancies between labelsets (disease genes):
    - Disease specificity index (DSI) filter
    - Max size filter
    - Min size filter
    - Jaccard index filter
    - Overlap coefficient filter

    """

    CONFIG_KEYS: List[str] = BaseAnnotatedOntologyData.CONFIG_KEYS + ["dsi_threshold"]
    ontology_url = "http://purl.obolibrary.org/obo/doid.obo"
    annotation_url = "https://www.disgenet.org/static/disgenet_ap1/files/downloads/all_gene_disease_associations.tsv.gz"
    ontology_file_name = "doid.obo"
    annotation_file_name = "all_gene_disease_associations.tsv"

    def __init__(
        self,
        root: str,
        dsi_threshold: float = 0.5,
        min_size: int = 10,
        max_size: int = 600,
        overlap: float = 0.7,
        jaccard: float = 0.5,
        **kwargs,
    ):
        """Initialize the DisGeNet data object."""
        self.dsi_threshold = dsi_threshold
        self.min_size = min_size
        self.max_size = max_size
        self.jaccard = jaccard
        self.overlap = overlap
        super().__init__(root, **kwargs)

    @property
    def _default_pre_transform(self):
        return Compose(
            LabelsetRangeFilterSize(max_val=self.max_size),
            LabelsetNonRedFilterOverlap(self.overlap),
            LabelsetNonRedFilterJaccard(self.jaccard),
            LabelsetRangeFilterSize(min_val=self.min_size),
            log_level=self.log_level,
        )

    def download_annotations(self):
        self.plogger.info(f"Download annotation from: {self.annotation_url}")
        resp = requests.get(self.annotation_url)
        annotation_file_name = self.annotation_file_name
        with open(osp.join(self.raw_dir, annotation_file_name), "wb") as f:
            f.write(gzip.decompress(resp.content))

    def process(self):
        g = OntologyGraph()
        umls_to_doid = g.read_obo(
            self.ontology_file_path,
            xref_prefix="UMLS_CUI",
        )
        annot_df = pd.read_csv(self.annotation_file_path, sep="\t")

        sub_df = annot_df[annot_df["DSI"] >= self.dsi_threshold]
        enable_pbar = display_pbar(self.log_level)
        pbar = tqdm(sub_df[["geneId", "diseaseId"]].values, disable=not enable_pbar)
        pbar.set_description("Annotating DOIDs")
        for gene_id, disease_id in pbar:
            for doid in umls_to_doid[disease_id]:
                try:
                    g._update_node_attr_partial(doid, str(gene_id))
                except IDNotExistError:
                    continue
        g._update_node_attr_finalize()

        # Propagate annotations and show progress
        g.complete_node_attrs(pbar=enable_pbar)

        lsc = LabelsetCollection.from_ontology_graph(g, min_size=self.min_size)
        lsc.export_gmt(self.processed_file_path(0))
