"""Interface with various databases to retrieve data."""
from .annotated_ontology import DisGeNet
from .annotated_ontology import GOBP
from .annotated_ontology import GOCC
from .annotated_ontology import GOMF
from .network import BioGRID
from .network import BioPlex
from .network import FunCoup
from .network import HIPPIE
from .network import HumanNet
from .network import PCNet
from .network import STRING

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
