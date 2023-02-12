"""Annotated ontology data."""
from nleval.data.annotated_ontology.diseases import (
    DISEASES,
    DISEASES_ExperimentsFiltered,
    DISEASES_ExperimentsFull,
    DISEASES_IntegratedFull,
    DISEASES_KnowledgeFiltered,
    DISEASES_KnowledgeFull,
    DISEASES_TextminingFiltered,
    DISEASES_TextminingFull,
)
from nleval.data.annotated_ontology.disgenet import (
    DisGeNET,
    DisGeNET_Animal,
    DisGeNET_BEFREE,
    DisGeNET_Curated,
    DisGeNET_GWAS,
)
from nleval.data.annotated_ontology.go import GOBP, GOCC, GOMF
from nleval.data.annotated_ontology.hpo import HPO

__all__ = [
    "DISEASES",
    "DISEASES_ExperimentsFiltered",
    "DISEASES_ExperimentsFull",
    "DISEASES_IntegratedFull",
    "DISEASES_KnowledgeFiltered",
    "DISEASES_KnowledgeFull",
    "DISEASES_TextminingFiltered",
    "DISEASES_TextminingFull",
    "DisGeNET",
    "DisGeNET_Animal",
    "DisGeNET_BEFREE",
    "DisGeNET_Curated",
    "DisGeNET_GWAS",
    "GOBP",
    "GOCC",
    "GOMF",
    "HPO",
]
