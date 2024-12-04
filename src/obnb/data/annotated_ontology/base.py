from tqdm import tqdm

from obnb.alltypes import Any, Dict, List, Optional, Type
from obnb.data.annotation.base import BaseAnnotationData
from obnb.data.base import BaseData
from obnb.data.ontology.base import BaseOntologyData
from obnb.exception import IDNotExistError
from obnb.label import LabelsetCollection
from obnb.util.logger import display_pbar


class BaseAnnotatedOntologyData(BaseData, LabelsetCollection):
    """General object for labelset collection from annotated ontology."""

    def __init__(
        self,
        root: str,
        *,
        annotation_factory: Type[BaseAnnotationData],
        ontology_factory: Type[BaseOntologyData],
        annotation_kwargs: Dict[str, Any] = {},
        ontology_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        """Initialize the BaseAnnotatedOntologyData object."""
        self.annotation_factory = annotation_factory
        self.ontology_factory = ontology_factory
        self.annotation_kwargs = annotation_kwargs
        self.ontology_kwargs = ontology_kwargs
        super().__init__(root, **kwargs)

    @property
    def processed_files(self) -> List[str]:
        return ["data.gmt"]

    def download_completed(self) -> bool:
        # NOTE: data are doownloaded separately by the annotation and ontology objects
        return True

    def download(self):
        pass

    def process(self):
        """Process raw data and save as gmt for future usage."""
        # NOTE: Reprocess is not a valid option for annotation and ontology
        # data objects as we do not save the processed data. Similarly,
        # retransform is invalid as there is not transformation for them yet.
        opts = {
            "redownload": self.redownload,
            "version": self.version,
            "log_level": self.log_level,
        }
        ann = self.annotation_factory(self.root, **self.annotation_kwargs, **opts)
        ont = self.ontology_factory(self.root, **self.ontology_kwargs, **opts)

        annot = ann.data
        g = ont.data
        xref_to_onto_ids = ont.xref_to_onto_ids
        enable_pbar = display_pbar(self.log_level)

        # Attach annotations to the corresponding ontology terms
        # TODO: extract the following block as a method of OntologyGraph?
        pbar = tqdm(annot.values, disable=not enable_pbar)
        hits, nohits = set(), set()
        for gene_id, disease_id in pbar:
            if disease_id not in xref_to_onto_ids:
                nohits.add(disease_id)
                continue

            hits.add(disease_id)
            for onto_id in xref_to_onto_ids[disease_id]:
                try:
                    g._update_node_attr_partial(onto_id, str(gene_id))
                except IDNotExistError:
                    self.plogger.debug(
                        f"Skipping {disease_id}({onto_id})-{gene_id} because "
                        f"{onto_id} is not available in the DO graph.",
                    )
        g._update_node_attr_finalize()

        self.plogger.info(f"{len(hits):,} annotation terms mapped to the ontology")
        self.plogger.info(f"{len(nohits):,} annotation terms missing in the ontology")
        k = annot["term_id"].isin(hits).sum()
        self.plogger.info(f"{k:,} out of {annot.shape[0]:,} annotations mapped")

        # Propagate annotations and show progress
        g.propagate_node_attrs(pbar=enable_pbar)

        lsc = LabelsetCollection.from_ontology_graph(g, min_size=self.min_size)
        lsc.export_gmt(self.processed_file_path(0))

    def apply_transform(self, transform: Any):
        """Apply a (pre-)transformation to the loaded data."""
        self.iapply(transform, progress_bar=display_pbar(self.log_level))

    def save(self, path):
        """Save the labelset collection as gmt."""
        self.export_gmt(path)

    def load_processed_data(self, path: Optional[str] = None):
        """Load processed labels from GMT."""
        path = path or self.processed_file_path(0)
        self.plogger.info(f"Load processed file {path}")
        self.read_gmt(path, reload=True)
