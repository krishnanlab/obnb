"""Interface with various databases to retrieve data."""
from nleval.data.annotated_ontology import GOBP, GOCC, GOMF, DisGeNET
from nleval.data.network import (
    HIPPIE,
    STRING,
    BioGRID,
    BioPlex,
    FunCoup,
    HumanNet,
    PCNet,
)

__all__ = classes = [
    "BioGRID",
    "BioPlex",
    "DisGeNET",
    "FunCoup",
    "GOBP",
    "GOCC",
    "GOMF",
    "HIPPIE",
    "HumanNet",
    "PCNet",
    "STRING",
]
