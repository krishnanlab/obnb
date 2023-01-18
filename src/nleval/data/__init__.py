"""Interface with various databases to retrieve data."""
from nleval.data.annotated_ontology import GOBP, GOCC, GOMF, DisGeNET
from nleval.data.network import (
    HIPPIE,
    STRING,
    BioGRID,
    BioPlex,
    ComPPIHumanInt,
    FunCoup,
    HumanBaseTopGlobal,
    HumanNet,
    HuRI,
    OmniPath,
    PCNet,
    ProteomeHD,
)

__all__ = classes = [
    "BioGRID",
    "BioPlex",
    "ComPPIHumanInt",
    "DisGeNET",
    "FunCoup",
    "GOBP",
    "GOCC",
    "GOMF",
    "HIPPIE",
    "HuRI",
    "HumanBaseTopGlobal",
    "HumanNet",
    "OmniPath",
    "PCNet",
    "ProteomeHD",
    "STRING",
]
