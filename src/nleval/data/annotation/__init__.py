"""Annotation data."""
from nleval.data.annotation.diseases import DISEASESAnnotation
from nleval.data.annotation.disgenet import DisGeNETAnnotation
from nleval.data.annotation.gene_ontology import GeneOntologyAnnotation

__all__ = [
    "DISEASESAnnotation",
    "DisGeNETAnnotation",
    "GeneOntologyAnnotation",
]
