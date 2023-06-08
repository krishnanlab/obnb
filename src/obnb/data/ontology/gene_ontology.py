from obnb.data.ontology.base import BaseOntologyData


class GeneOntology(BaseOntologyData):
    """The Gene Ontology.

    http://geneontology.org/

    """

    ontology_file_name = "go.obo"
    ontology_url = "http://purl.obolibrary.org/obo/go.obo"
    xref_prefix = None

    def __init__(self, root, **kwargs):
        """Initialize GeneOntology data object."""
        super().__init__(root, **kwargs)
