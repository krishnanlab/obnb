"""Network data."""
from nleval.data.network.biogrid import BioGRID
from nleval.data.network.bioplex import BioPlex
from nleval.data.network.comppi import ComPPIHumanInt
from nleval.data.network.consensuspathdb import ConsensusPathDB
from nleval.data.network.funcoup import FunCoup
from nleval.data.network.hippie import HIPPIE
from nleval.data.network.humanbase import HumanBaseTopGlobal
from nleval.data.network.humannet import HumanNet, HumanNet_CC, HumanNet_FN
from nleval.data.network.humap import HuMAP
from nleval.data.network.huri import HuRI
from nleval.data.network.omnipath import OmniPath
from nleval.data.network.pcnet import PCNet
from nleval.data.network.proteomehd import ProteomeHD
from nleval.data.network.signor import SIGNOR
from nleval.data.network.string import STRING

__all__ = [
    "BioGRID",
    "BioPlex",
    "ComPPIHumanInt",
    "ConsensusPathDB",
    "FunCoup",
    "HIPPIE",
    "HuRI",
    "HuMAP",
    "HumanBaseTopGlobal",
    "HumanNet",
    "HumanNet_CC",
    "HumanNet_FN",
    "OmniPath",
    "PCNet",
    "ProteomeHD",
    "SIGNOR",
    "STRING",
]
