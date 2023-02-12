from nleval.data.annotated_ontology.base import BaseAnnotatedOntologyData
from nleval.data.annotation import GeneOntologyAnnotation
from nleval.data.ontology import GeneOntology
from nleval.label.filters import Compose, LabelsetNonRedFilter, LabelsetRangeFilterSize
from nleval.typing import List, LogLevel, Mapping, Optional, Union


class GO(BaseAnnotatedOntologyData):
    """The Gene Ontology gene set collection."""

    namespace: Optional[str] = None

    def __init__(
        self,
        root: str,
        min_size: int = 10,
        max_size: int = 200,
        overlap: float = 0.7,
        jaccard: float = 0.5,
        data_sources: Optional[List[str]] = None,
        gene_id_converter: Optional[Union[Mapping[str, str], str]] = "HumanEntrez",
        redownload: bool = False,
        version: str = "latest",
        log_level: LogLevel = "INFO",
        **kwargs,
    ):
        """Initialize the GO data object."""
        self.min_size = min_size
        self.max_size = max_size
        self.jaccard = jaccard
        self.overlap = overlap

        annotation = GeneOntologyAnnotation(
            root,
            data_sources=data_sources,
            gene_id_converter=gene_id_converter,
            redownload=redownload,
            version=version,
            log_level=log_level,
        )
        ontology = GeneOntology(
            root,
            redownload=redownload,
            version=version,
            log_level=log_level,
        )
        if self.namespace is not None:
            ontology.data = ontology.data.restrict_to_branch(self.namespace)

        super().__init__(
            root,
            annotation=annotation,
            ontology=ontology,
            **kwargs,
        )

    @property
    def default_pre_transform(self):
        return Compose(
            LabelsetRangeFilterSize(max_val=self.max_size),
            LabelsetNonRedFilter(self.jaccard, self.overlap),
            LabelsetRangeFilterSize(min_val=self.min_size),
            log_level=self.log_level,
        )


class GOBP(GO):
    """The Gene Ontology Biological Process gene set collection."""

    namespace = "GO:0008150"  # biological_process


class GOCC(GO):
    """The Gene Ontology Cellular Component gene set collection."""

    namespace = "GO:0005575"  # cellular_component


class GOMF(GO):
    """The Gene Ontology Molecular Function gene set collection."""

    namespace = "GO:0003674"  # molecular_function
