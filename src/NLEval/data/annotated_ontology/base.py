import os.path as osp

import requests

from NLEval.data.base import BaseData
from NLEval.label import LabelsetCollection
from NLEval.label.filters import Compose
from NLEval.typing import Any, List, Optional


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

    @property
    def filters(self):
        """Labelset collection processing filters."""
        return Compose()

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

    def filter_and_save(self, lsc):
        self.plogger.info(f"Raw stats:\n{lsc.stats()}")

        self.plogger.info(f"Apply {self.filters}\n")
        lsc.iapply(self.filters, progress_bar=True)

        out_path = self.processed_file_path(0)
        lsc.export_gmt(out_path)
        self.plogger.info(f"Saved processed file {out_path}")

    def transform(self, transform: Any):
        """Apply a (pre-)transformation to the loaded data."""
        # TODO: Option to disabble progress bar?
        self.iapply(transform, progress_bar=True)

    def load_processed_data(self, path: Optional[str] = None):
        """Load processed labels from GMT."""
        path = path or self.processed_file_path(0)
        self.plogger.info(f"Load processed file {path}")
        self.read_gmt(path, reload=True)
