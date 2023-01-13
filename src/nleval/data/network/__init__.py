"""Network data."""
from nleval.data.network.biogrid import BioGRID
from nleval.data.network.bioplex import BioPlex
from nleval.data.network.funcoup import FunCoup
from nleval.data.network.hippie import HIPPIE
from nleval.data.network.humannet import HumanNet
from nleval.data.network.pcnet import PCNet
from nleval.data.network.proteomehd import ProteomeHD
from nleval.data.network.string import STRING

__all__ = [
    "BioGRID",
    "BioPlex",
    "FunCoup",
    "HIPPIE",
    "HumanNet",
    "PCNet",
    "ProteomeHD",
    "STRING",
]
