from nleval.data.annotated_ontology.base import BaseAnnotatedOntologyData
from nleval.data.annotation import HumanPhenotypeOntologyAnnotation
from nleval.data.ontology import MondoDiseaseOntology
from nleval.label.filters import Compose, LabelsetNonRedFilter, LabelsetRangeFilterSize
from nleval.typing import List, LogLevel, Mapping, Optional, Union


class HPO(BaseAnnotatedOntologyData):
    """The HPO disease and triat gene set collection."""

    def __init__(
        self,
        root: str,
        min_size: int = 10,
        max_size: int = 600,
        overlap: float = 0.7,
        jaccard: float = 0.5,
        data_sources: Optional[List[str]] = None,
        gene_id_converter: Optional[Union[Mapping[str, str], str]] = None,
        redownload: bool = False,
        version: str = "latest",
        log_level: LogLevel = "INFO",
        **kwargs,
    ):
        """Initialize the HPO data object."""
        self.min_size = min_size
        self.max_size = max_size
        self.jaccard = jaccard
        self.overlap = overlap

        annotation = HumanPhenotypeOntologyAnnotation(
            root,
            data_sources=data_sources,
            gene_id_converter=gene_id_converter,
            redownload=redownload,
            version=version,
            log_level=log_level,
        )
        ontology = MondoDiseaseOntology(
            root,
            xref_prefix="HP",
            redownload=redownload,
            version=version,
            log_level=log_level,
        )

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
