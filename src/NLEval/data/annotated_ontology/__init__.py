"""Annotated ontology data."""
from .disgenet import DisGeNet
from .go import GOBP, GOCC, GOMF

__all__ = [
    "DisGeNet",
    "GOBP",
    "GOCC",
    "GOMF",
]
