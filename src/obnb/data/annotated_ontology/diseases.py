from obnb.data.annotated_ontology.base import BaseAnnotatedOntologyData
from obnb.data.annotation import DISEASESAnnotation
from obnb.data.ontology import MondoDiseaseOntology
from obnb.label.filters import Compose, LabelsetNonRedFilter, LabelsetRangeFilterSize
from obnb.typing import Mapping, Optional, Union
from obnb.util.registers import overload_class


class DISEASES(BaseAnnotatedOntologyData):
    """The DISEASES disease gene set collection."""

    def __init__(
        self,
        root: str,
        score_min: Optional[float] = 3,
        score_max: Optional[float] = None,
        channel: str = "integrated_full",
        min_size: int = 10,
        max_size: int = 600,
        overlap: float = 0.7,
        jaccard: float = 0.5,
        gene_id_converter: Optional[Union[Mapping[str, str], str]] = "HumanEntrez",
        **kwargs,
    ):
        """Initialize the DisGeNET data object."""
        self.min_size = min_size
        self.max_size = max_size
        self.jaccard = jaccard
        self.overlap = overlap

        super().__init__(
            root,
            annotation_factory=DISEASESAnnotation,
            ontology_factory=MondoDiseaseOntology,
            annotation_kwargs={
                "score_min": score_min,
                "score_max": score_max,
                "channel": channel,
                "gene_id_converter": gene_id_converter,
            },
            ontology_kwargs={"xref_prefix": "DOID"},
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


DISEASES_IntegratedFull = overload_class(
    DISEASES,
    "IntegratedFull",
    channel="integrated_full",
    score_min=None,
)
DISEASES_TextminingFull = overload_class(
    DISEASES,
    "TextminingFull",
    channel="textmining_full",
    score_min=None,
)
DISEASES_TextminingFiltered = overload_class(
    DISEASES,
    "TextminingFiltered",
    channel="textmining_filtered",
    score_min=None,
)
DISEASES_KnowledgeFull = overload_class(
    DISEASES,
    "KnowledgeFull",
    channel="knowledge_full",
    score_min=None,
)
DISEASES_KnowledgeFiltered = overload_class(
    DISEASES,
    "KnowledgeFiltered",
    channel="knowledge_filtered",
    score_min=None,
)
DISEASES_ExperimentsFull = overload_class(
    DISEASES,
    "ExperimentsFull",
    channel="experiments_full",
    score_min=None,
)
DISEASES_ExperimentsFiltered = overload_class(
    DISEASES,
    "ExperimentsFiltered",
    channel="experiments_filtered",
    score_min=None,
)
