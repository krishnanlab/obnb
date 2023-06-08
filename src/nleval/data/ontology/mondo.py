from obnb.data.ontology.base import BaseOntologyData


class MondoDiseaseOntology(BaseOntologyData):
    """The Mondo Disease Ontology.

    https://mondo.monarchinitiative.org/

    """

    ontology_file_name = "mondo.obo"
    ontology_url = "http://purl.obolibrary.org/obo/mondo.obo"

    def __init__(self, root, xref_prefix=None, **kwargs):
        """Initialize MondoDiseaseOntology data object."""
        super().__init__(root, xref_prefix=xref_prefix, **kwargs)
