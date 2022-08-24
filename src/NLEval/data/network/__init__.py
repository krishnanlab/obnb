"""Network data."""
from NLEval.data.network.biogrid import BioGRID
from NLEval.data.network.bioplex import BioPlex
from NLEval.data.network.funcoup import FunCoup
from NLEval.data.network.hippie import HIPPIE
from NLEval.data.network.humannet import HumanNet
from NLEval.data.network.pcnet import PCNet
from NLEval.data.network.string import STRING

__all__ = [
    "BioGRID",
    "BioPlex",
    "FunCoup",
    "HIPPIE",
    "HumanNet",
    "PCNet",
    "STRING",
]
