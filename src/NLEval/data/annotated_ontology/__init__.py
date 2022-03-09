"""Annotated ontology data."""
from .disgenet import DisGeNet
from .go import GOBP
from .go import GOCC
from .go import GOMF

__all__ = [
    "DisGeNet",
    "GOBP",
    "GOCC",
    "GOMF",
]
