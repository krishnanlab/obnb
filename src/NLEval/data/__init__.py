"""Interface with various databases to retrieve data."""
from NLEval.data.annotated_ontology import (
    DisGeNet,
    GOBP,
    GOCC,
    GOMF,
)
from NLEval.data.network import (
    BioGRID,
    BioPlex,
    FunCoup,
    HIPPIE,
    HumanNet,
    PCNet,
    STRING,
)

__all__ = [
    "BioGRID",
    "BioPlex",
    "DisGeNet",
    "FunCoup",
    "GOBP",
    "GOCC",
    "GOMF",
    "HIPPIE",
    "HumanNet",
    "PCNet",
    "STRING",
]
