"""Annotated ontology data."""
from obnb.data.annotated_ontology.diseases import (
    DISEASES,
    DISEASES_ExperimentsFiltered,
    DISEASES_ExperimentsFull,
    DISEASES_IntegratedFull,
    DISEASES_KnowledgeFiltered,
    DISEASES_KnowledgeFull,
    DISEASES_TextminingFiltered,
    DISEASES_TextminingFull,
)
from obnb.data.annotated_ontology.disgenet import (
    DisGeNET,
    DisGeNET_Animal,
    DisGeNET_BEFREE,
    DisGeNET_Curated,
    DisGeNET_GWAS,
)
from obnb.data.annotated_ontology.go import GO, GOBP, GOCC, GOMF
from obnb.data.annotated_ontology.hpo import HPO

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
    "GO",
    "GOBP",
    "GOCC",
    "GOMF",
    "HPO",
]
