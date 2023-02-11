from nleval.data.ontology.base import BaseOntologyData


class MondoDiseaseOntology(BaseOntologyData):
    """The Mondo Disease Ontology.

    https://mondo.monarchinitiative.org/

    """

    ontology_file_name = "mondo.obo"
    ontology_url = "http://purl.obolibrary.org/obo/mondo.obo"
    xref_prefix = "UMLS"

    def __init__(self, root, **kwargs):
        """Initialize MondoDiseaseOntology data object."""
        super().__init__(root, **kwargs)
