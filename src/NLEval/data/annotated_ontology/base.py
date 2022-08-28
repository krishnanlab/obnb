import os.path as osp

import requests

from NLEval.data.base import BaseData
from NLEval.label import LabelsetCollection
from NLEval.typing import Any, List, Optional
from NLEval.util.logger import display_pbar


class BaseAnnotatedOntologyData(BaseData, LabelsetCollection):
    """General object for labelset collection from annotated ontology."""

    ontology_url: Optional[str] = None
    annotation_url: Optional[str] = None
    ontology_file_name: Optional[str] = None
    annotation_file_name: Optional[str] = None

    def __init__(
        self,
        root: str,
        **kwargs,
    ):
        """Initialize the BaseAnnotatedOntologyData object."""
        super().__init__(root, **kwargs)

    @property
    def raw_files(self) -> List[str]:
        """List of available raw files."""
        files = [self.ontology_file_name, self.annotation_file_name]
        return list(filter(None, files))

    @property
    def processed_files(self) -> List[str]:
        return ["data.gmt"]

    @property
    def ontology_file_path(self) -> str:
        """Path to onlogy file."""
        if self.ontology_file_name is not None:
            return osp.join(self.raw_dir, self.ontology_file_name)
        else:
            raise ValueError(
                f"Ontology file name not available for {self.classname!r}",
            )

    @property
    def annotation_file_path(self) -> str:
        """Path to annotation fil."""
        if self.annotation_file_name is not None:
            return osp.join(self.raw_dir, self.annotation_file_name)
        else:
            raise ValueError(
                f"Annotation file name not available for {self.classname!r}",
            )

    def download_ontology(self):
        """Download ontology from obo foundary."""
        self.plogger.info(f"Download obo from: {self.ontology_url}")
        resp = requests.get(self.ontology_url)
        with open(self.ontology_file_path, "wb") as f:
            f.write(resp.content)

    def download_annotations(self):
        """Download annotations."""
        raise NotImplementedError

    def download(self):
        """Download the ontology and annotations."""
        self.download_ontology()
        self.download_annotations()

    def process(self):
        """Process raw data and save as gmt for future usage."""
        raise NotImplementedError

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
