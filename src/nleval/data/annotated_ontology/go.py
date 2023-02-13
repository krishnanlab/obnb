from nleval.data.annotated_ontology.base import BaseAnnotatedOntologyData
from nleval.data.annotation import GeneOntologyAnnotation
from nleval.data.ontology import GeneOntology
from nleval.label.filters import Compose, LabelsetNonRedFilter, LabelsetRangeFilterSize
from nleval.typing import List, Mapping, Optional, Union
from nleval.util.registers import overload_class


class GO(BaseAnnotatedOntologyData):
    """The Gene Ontology gene set collection."""

    def __init__(
        self,
        root: str,
        min_size: int = 10,
        max_size: int = 200,
        overlap: float = 0.7,
        jaccard: float = 0.5,
        branch: Optional[str] = None,
        data_sources: Optional[List[str]] = None,
        gene_id_converter: Optional[Union[Mapping[str, str], str]] = "HumanEntrez",
        **kwargs,
    ):
        """Initialize the GO data object."""
        self.min_size = min_size
        self.max_size = max_size
        self.jaccard = jaccard
        self.overlap = overlap

        super().__init__(
            root,
            annotation_factory=GeneOntologyAnnotation,
            ontology_factory=GeneOntology,
            annotation_kwargs={
                "data_sources": data_sources,
                "gene_id_converter": gene_id_converter,
            },
            ontology_kwargs={"branch": branch},
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


GOBP = overload_class(
    GO,
    "BP",
    sep="",
    docstring="The Gene Ontology Biological Process gene set collection.",
    branch="GO:0008150",  # biological_process
)
GOCC = overload_class(
    GO,
    "CC",
    sep="",
    docstring="The Gene Ontology Cellular Component gene set collection.",
    branch="GO:0005575",  # cellular_component
)
GOMF = overload_class(
    GO,
    "MF",
    sep="",
    docstring="The Gene Ontology Molecular Function gene set collection.",
    branch="GO:0003674",  # molecular_function
)
