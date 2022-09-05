import pandas as pd
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
from NLEval.typing import List, Mapping, Optional, Union
from NLEval.util.logger import display_pbar


class GeneOntology(BaseAnnotatedOntologyData):
    """Gene Ontology gene annotations."""

    CONFIG_KEYS: List[str] = BaseAnnotatedOntologyData.CONFIG_KEYS + [
        "data_sources",
        "qualifiers",
    ]
    ontology_url = "http://purl.obolibrary.org/obo/go.obo"
    annotation_url = "http://geneontology.org/gene-associations/goa_human.gaf.gz"
    ontology_file_name = "go.obo"
    annotation_file_name = "goa_human.gaf"

    def __init__(
        self,
        root: str,
        min_size: int = 10,
        max_size: int = 500,
        overlap: float = 0.7,
        jaccard: float = 0.5,
        data_sources: Union[List[str], str] = "default",
        qualifiers: Union[List[str], str] = "default",
        gene_id_converter: Optional[Union[Mapping[str, str], str]] = "HumanEntrez",
        **kwargs,
    ):
        """Initialize the GOBP data object."""
        self.min_size = min_size
        self.max_size = max_size
        self.jaccard = jaccard
        self.overlap = overlap
        self._data_sources = data_sources
        self._qualifiers = qualifiers
        super().__init__(root, gene_id_converter=gene_id_converter, **kwargs)

    @property
    def data_sources(self) -> List[str]:
        if self._data_sources == "default":
            return [
                "EXP",  # Experiment
                "IDA",  # Direct Assay
                "IPI",  # Physical Interaction
                "IMP",  # Mutant Phenotype
                "IGI",  # Genetic Interaction
                "IEP",  # Expression Pattern
                "TAS",  # Traceable Author Statement
                "NAS",  # Non-traceable Author Statement
                "IC",  # Inferred by Curator
            ]
        else:
            return self._data_sources  # type: ignore

    @property
    def qualifiers(self) -> List[str]:
        if self._data_sources == "default":
            return [
                "acts_upstream_of",
                "acts_upstream_of_negative_effect",
                "acts_upstream_of_or_within",
                "acts_upstream_of_or_within_negative_effect",
                "acts_upstream_of_or_within_positive_effect",
                "acts_upstream_of_positive_effect",
                "colocalizes_with",
                "contributes_to",
                "enables",
                "involved_in",
                "is_active_in",
                "located_in",
                "part_of",
            ]
        else:
            return self._data_sources  # type: ignore

    @property
    def _default_pre_transform(self):
        return Compose(
            LabelsetRangeFilterSize(max_val=self.max_size),
            LabelsetNonRedFilterOverlap(self.overlap),
            LabelsetNonRedFilterJaccard(self.jaccard),
            LabelsetRangeFilterSize(min_val=self.min_size),
            log_level=self.log_level,
        )

    def process(self):
        g = OntologyGraph.from_obo(self.ontology_file_path)

        # Get GO terms in specific namespace (BP, CC, MF)
        with g.cache_on_static():
            goids = set(
                filter(
                    lambda goid: self.namespace in g.ancestors(goid),
                    g.node_ids,
                ),
            )

        # Load gene annotation data (gaf-version: 2.2)
        annot_df = pd.read_csv(
            self.annotation_file_path,
            sep="\t",
            comment="!",
            header=0,
            names=[
                "db",
                "db_id",
                "db_symbol",
                "qual",
                "go_id",
                "db_ref",
                "ec",
                "wof",
                "aspect",
                "eb_name",
                "db_syn",
                "db_type",
                "taxon",
                "date",
                "assigned_by",
                "annot_ext",
                "gene_prod_id",
            ],
            low_memory=False,
        )

        # Bulk query gene symbol to entrez conversion and get converted genes
        gene_symbols_to_query = annot_df["db_symbol"].unique().tolist()
        gene_id_converter = self.get_gene_id_converter()
        gene_id_converter.query_bulk(gene_symbols_to_query)
        converted_symbols = {
            i for i in gene_symbols_to_query if gene_id_converter[i] is not None
        }

        self.plogger.info(f"Number of go-gene annotations: {annot_df.shape[0]:,}")
        annot_df = annot_df[
            annot_df["go_id"].isin(goids)
            & annot_df["db_symbol"].isin(converted_symbols)
            & annot_df["ec"].isin(self.data_sources)
            & annot_df["qual"].isin(self.qualifiers)
        ]
        self.plogger.info(
            f"Number of go-gene annotations after filtering: {annot_df.shape[0]:,}",
        )

        enable_pbar = display_pbar(self.log_level)
        pbar = tqdm(annot_df[["db_symbol", "go_id"]].values, disable=not enable_pbar)
        pbar.set_description("Annotating GO terms")
        for gene_symbol, go_id in pbar:
            try:
                gene_id = gene_id_converter[gene_symbol]
                g._update_node_attr_partial(go_id, gene_id)
            except IDNotExistError:
                self.plogger.error(
                    f"Skipping {go_id}-{gene_symbol}({gene_id}) because "
                    f"{go_id} is not available in the GO graph.",
                )
        g._update_node_attr_finalize()

        # Propagate annotations and show progress
        g.complete_node_attrs(pbar=enable_pbar)

        lsc = LabelsetCollection.from_ontology_graph(
            g,
            min_size=self.min_size,
            namespace=self.namespace,
        )
        lsc.export_gmt(self.processed_file_path(0))


class GOBP(GeneOntology):
    """Gene Ontology Biological Process gene annotations."""

    namespace = "GO:0008150"  # biological_process


class GOCC(GeneOntology):
    """Gene Ontology Cellular Component gene annotations."""

    namespace = "GO:0005575"  # cellular_component


class GOMF(GeneOntology):
    """Gene Ontology Molecular Function gene annotations."""

    namespace = "GO:0003674"  # molecular_function
