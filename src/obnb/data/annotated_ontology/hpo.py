from obnb.data.annotated_ontology.base import BaseAnnotatedOntologyData
from obnb.data.annotation import HumanPhenotypeOntologyAnnotation
from obnb.data.ontology import MondoDiseaseOntology
from obnb.label.filters import Compose, LabelsetNonRedFilter, LabelsetRangeFilterSize
from obnb.typing import List, Mapping, Optional, Union


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
        **kwargs,
    ):
        """Initialize the HPO data object."""
        self.min_size = min_size
        self.max_size = max_size
        self.jaccard = jaccard
        self.overlap = overlap

        super().__init__(
            root,
            annotation_factory=HumanPhenotypeOntologyAnnotation,
            ontology_factory=MondoDiseaseOntology,
            annotation_kwargs={
                "data_sources": data_sources,
                "gene_id_converter": gene_id_converter,
            },
            ontology_kwargs={"xref_prefix": "HP"},
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
