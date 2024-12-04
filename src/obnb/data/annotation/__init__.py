"""Annotation data."""

from obnb.data.annotation.diseases import DISEASESAnnotation
from obnb.data.annotation.disgenet import DisGeNETAnnotation
from obnb.data.annotation.gene_ontology import GeneOntologyAnnotation
from obnb.data.annotation.human_phenotype_ontology import (
    HumanPhenotypeOntologyAnnotation,
)

__all__ = [
    "DISEASESAnnotation",
    "DisGeNETAnnotation",
    "GeneOntologyAnnotation",
    "HumanPhenotypeOntologyAnnotation",
]
