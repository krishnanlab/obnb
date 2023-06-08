"""Network data."""
from obnb.data.network.biogrid import BioGRID
from obnb.data.network.bioplex import BioPlex
from obnb.data.network.comppi import ComPPIHumanInt
from obnb.data.network.consensuspathdb import ConsensusPathDB
from obnb.data.network.funcoup import FunCoup
from obnb.data.network.hippie import HIPPIE
from obnb.data.network.humanbase import HumanBaseTopGlobal
from obnb.data.network.humannet import HumanNet, HumanNet_CC, HumanNet_FN
from obnb.data.network.humap import HuMAP
from obnb.data.network.huri import HuRI
from obnb.data.network.omnipath import OmniPath
from obnb.data.network.pcnet import PCNet
from obnb.data.network.proteomehd import ProteomeHD
from obnb.data.network.signor import SIGNOR
from obnb.data.network.string import STRING

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
