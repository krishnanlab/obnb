import mygene
from tqdm import tqdm

from NLEval.data.annotated_ontology.base import BaseAnnotatedOntologyData
from NLEval.graph import OntologyGraph
from NLEval.label import LabelsetCollection
from NLEval.label.filters import (
    Compose,
    LabelsetNonRedFilterJaccard,
    LabelsetNonRedFilterOverlap,
    LabelsetRangeFilterSize,
)
from NLEval.util.logger import display_pbar


class GeneOntology(BaseAnnotatedOntologyData):
    """Gene Ontology gene annotations."""

    ontology_url = "http://purl.obolibrary.org/obo/go.obo"
    ontology_file_name = "go.obo"

    def __init__(
        self,
        root: str,
        min_size: int = 10,
        max_size: int = 500,
        jaccard: float = 0.5,
        overlap: float = 0.7,
        **kwargs,
    ):
        """Initialize the GOBP data object."""
        self.min_size = min_size
        self.max_size = max_size
        self.jaccard = jaccard
        self.overlap = overlap
        super().__init__(root, **kwargs)

    @property
    def _default_pre_transform(self):
        return Compose(
            LabelsetRangeFilterSize(max_val=self.max_size),
            LabelsetNonRedFilterJaccard(self.jaccard),
            LabelsetNonRedFilterOverlap(self.overlap),
            LabelsetRangeFilterSize(min_val=self.min_size),
            log_level=self.log_level,
        )

    def download_annotations(self):
        pass

    def process(self):
        g = OntologyGraph.from_obo(self.ontology_file_path)

        # Query of GO terms in specific namespace (BP, CC, MF)
        mg = mygene.MyGeneInfo()
        with g.cache_on_static():
            goids = filter(
                lambda goid: self.namespace in g.ancestors(goid),
                g.node_ids,
            )
            queries = mg.querymany(
                goids,
                scopes="go",
                species="human",
                fields="entrezgene",
                entrezonly=True,
            )

        enable_pbar = display_pbar(self.log_level)
        pbar = tqdm(queries, disable=not enable_pbar)
        pbar.set_description("Annotating GO terms")
        for query in pbar:
            try:
                goid = query["query"]
                gene_id = query["entrezgene"]
                g._update_node_attr_partial(goid, gene_id)
            except KeyError:
                continue
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
