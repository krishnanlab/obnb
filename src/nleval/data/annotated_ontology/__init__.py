"""Annotated ontology data."""
from nleval.data.annotated_ontology.diseases import DISEASES
from nleval.data.annotated_ontology.disgenet import DisGeNET
from nleval.data.annotated_ontology.go import GOBP, GOCC, GOMF
from nleval.data.annotated_ontology.hpo import HPO

__all__ = [
    "DISEASES",
    "DisGeNET",
    "GOBP",
    "GOCC",
    "GOMF",
    "HPO",
]
