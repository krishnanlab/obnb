"""Interface with various databases to retrieve data."""
from .biogrid import BioGRID
from .bioplex import BioPlex
from .disgenet import DisGeNet
from .funcoup import FunCoup
from .go import GOBP
from .go import GOCC
from .go import GOMF
from .hippie import HIPPIE
from .humannet import HumanNet
from .string import STRING

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
    "STRING",
]
