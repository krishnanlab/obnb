"""Interface with various databases to retrieve data."""
from .annotated_ontology import GOBP, GOCC, GOMF, DisGeNet
from .network import HIPPIE, STRING, BioGRID, BioPlex, FunCoup, HumanNet, PCNet

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
