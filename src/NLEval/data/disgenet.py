import gzip
import os.path as osp
from typing import Dict

import pandas as pd
import requests
from tqdm import tqdm

from ..graph import OntologyGraph
from ..label.filters import LabelsetPairwiseFilterJaccard
from ..label.filters import LabelsetPairwiseFilterOverlap
from ..label.filters import LabelsetRangeFilterSize
from ..util.exceptions import IDNotExistError
from .base import BaseAnnotatedOntologyData


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

    ontology_url = "http://purl.obolibrary.org/obo/doid.obo"
    annotation_url = "https://www.disgenet.org/static/disgenet_ap1/files/downloads/all_gene_disease_associations.tsv.gz"

    def __init__(
        self,
        root: str,
        dsi_threshold: float = 0.6,
        min_size: int = 10,
        max_size: int = 600,
        jaccard: float = 0.5,
        overlap: float = 0.7,
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
    def data_name_dict(self) -> Dict[str, str]:
        return {
            "ontology": "doid.obo",
            "annotation": "all_gene_disease_associations.tsv",
        }

    @property
    def filters(self):
        return [
            LabelsetRangeFilterSize(max_val=self.max_size),
            LabelsetPairwiseFilterJaccard(
                self.jaccard,
                size_constraint="smaller",
                inclusive=True,
            ),
            LabelsetPairwiseFilterOverlap(
                self.overlap,
                size_constraint="smaller",
                inclusive=True,
            ),
            LabelsetRangeFilterSize(min_val=self.min_size),
        ]

    def download_annotations(self):
        resp = requests.get(self.annotation_url)
        annotation_file_name = self.data_name_dict["annotation"]
        with open(osp.join(self.raw_dir, annotation_file_name), "wb") as f:
            f.write(gzip.decompress(resp.content))

    def process(self):
        g = OntologyGraph()
        umls_to_doid = g.read_obo(
            self.ontology_data_path,
            xref_prefix="UMLS_CUI",
        )
        annot_df = pd.read_csv(self.annotation_data_path, sep="\t")

        sub_df = annot_df[annot_df["DSI"] >= self.dsi_threshold]
        pbar = tqdm(sub_df[["geneId", "diseaseId"]].values)
        pbar.set_description("Annotating DOIDs")
        for gene_id, disease_id in pbar:
            for doid in umls_to_doid[disease_id]:
                try:
                    g._update_node_attr_partial(doid, str(gene_id))
                except IDNotExistError:
                    continue
        g._update_node_attr_finalize()

        # Propagate annotations and show progress
        g.complete_node_attrs(pbar=True)

        self.read_ontology_graph(g, min_size=self.min_size)
        print(self.stats())

        for filter_ in self.filters:
            self.iapply(filter_, progress_bar=True)
            print(self.stats())

        print("Saving processed gmt...")
        self.export_gmt(self.processed_data_path)
