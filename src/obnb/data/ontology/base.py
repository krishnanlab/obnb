from obnb.alltypes import List, Optional
from obnb.data.base import BaseData
from obnb.graph import OntologyGraph
from obnb.util.download import stream_download


class BaseOntologyData(BaseData):
    """BaseOntologyData object.

    Attributes:
        xref_prefix: Cross reference prefix to filter (see
            :meth:`obnb.graph.ontology.OntologyGraph.read_obo`)

    """

    CONFIG_KEYS: List[str] = BaseData.CONFIG_KEYS + ["ontology_url", "xref_prefix"]
    ontology_url: Optional[str] = None
    ontology_file_name: Optional[str] = None

    def __init__(
        self,
        root: str,
        *,
        xref_prefix: Optional[str] = None,
        branch: Optional[str] = None,
        **kwargs,
    ):
        """Initialize BaseOntologyData."""
        self.xref_prefix = xref_prefix
        self.branch = branch
        super().__init__(root, **kwargs)

    @property
    def raw_files(self) -> List[str]:
        if not isinstance(self.ontology_file_name, str):
            raise ValueError("Ontology file name not specified.")
        return [self.ontology_file_name]

    def download(self):
        """Download ontology from the OBO Foundry."""
        self.plogger.info(f"Downloading obo file from: {self.ontology_url}")
        content = stream_download(self.ontology_url, log_level=self.log_level)[1]
        with open(self.raw_file_path(0), "wb") as f:
            f.write(content)

    def process(self):
        # NOTE: we process the ontology graph from raw file directly, so we
        # do not need to pre-process and save the processed file.
        pass

    def process_completed(self) -> bool:
        return True

    def load_processed_data(self, path: Optional[str] = None):
        """Load ontology graph."""
        path = self.raw_file_path(0)
        self.plogger.info(f"Load processed annodataion {path}")
        ont = OntologyGraph(logger=self.plogger)
        self.xref_to_onto_ids = ont.read_obo(path, xref_prefix=self.xref_prefix)
        self.data = ont if self.branch is None else ont.restrict_to_branch(self.branch)
