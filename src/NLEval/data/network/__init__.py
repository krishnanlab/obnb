"""Network data."""
from .biogrid import BioGRID
from .bioplex import BioPlex
from .funcoup import FunCoup
from .hippie import HIPPIE
from .humannet import HumanNet
from .pcnet import PCNet
from .string import STRING

__all__ = [
    "BioGRID",
    "BioPlex",
    "FunCoup",
    "HIPPIE",
    "HumanNet",
    "PCNet",
    "STRING",
]
