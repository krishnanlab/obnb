from obnb.data.annotated_ontology.base import BaseAnnotatedOntologyData
from obnb.data.annotation import DisGeNETAnnotation
from obnb.data.ontology import MondoDiseaseOntology
from obnb.label.filters import Compose, LabelsetNonRedFilter, LabelsetRangeFilterSize
from obnb.typing import List, Mapping, Optional, Union
from obnb.util.registers import overload_class


class DisGeNET(BaseAnnotatedOntologyData):
    """The DisGeNET disease gene set collection."""

    def __init__(
        self,
        root: str,
        dsi_min: Optional[float] = None,
        dsi_max: Optional[float] = None,
        dpi_min: Optional[float] = None,
        dpi_max: Optional[float] = None,
        min_size: int = 10,
        max_size: int = 600,
        overlap: float = 0.7,
        jaccard: float = 0.5,
        data_sources: Optional[List[str]] = None,
        gene_id_converter: Optional[Union[Mapping[str, str], str]] = None,
        **kwargs,
    ):
        """Initialize the DisGeNET data object."""
        self.min_size = min_size
        self.max_size = max_size
        self.jaccard = jaccard
        self.overlap = overlap

        super().__init__(
            root,
            annotation_factory=DisGeNETAnnotation,
            ontology_factory=MondoDiseaseOntology,
            annotation_kwargs={
                "data_sources": data_sources,
                "dsi_min": dsi_min,
                "dsi_max": dsi_max,
                "dpi_min": dpi_min,
                "dpi_max": dpi_max,
                "gene_id_converter": gene_id_converter,
            },
            ontology_kwargs={"xref_prefix": "UMLS"},
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


DisGeNET_Curated = overload_class(
    DisGeNET,
    "Curated",
    data_sources=[
        "CGI",
        "CLINGEN",
        "CTD_human",
        "GENOMICS_ENGLAND",
        "ORPHANET",
        "PSYGENET",
        "CLINVAR",
    ],
)
DisGeNET_Animal = overload_class(
    DisGeNET,
    "Animal",
    data_sources=[
        "CTD_mouse",
        "CTD_rat",
        "MGD",
        "RGD",
    ],
)
DisGeNET_GWAS = overload_class(DisGeNET, "GWAS", data_sources=["GWASCAT", "GWASDB"])
DisGeNET_BEFREE = overload_class(DisGeNET, "BEFREE", data_sources=["BEFREE"])
