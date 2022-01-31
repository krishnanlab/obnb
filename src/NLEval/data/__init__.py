"""Interface with various databases to retrieve data."""
from .biogrid import BioGRID
from .bioplex import BioPlex
from .funcoup import FunCoup
from .hippie import HIPPIE
from .humannet import HumanNet
from .string import STRING

__all__ = [
    "BioGRID",
    "BioPlex",
    "FunCoup",
    "HIPPIE",
    "HumanNet",
    "STRING",
]
