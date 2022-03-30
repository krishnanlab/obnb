import os.path as osp

import requests

from ...label import LabelsetCollection
from ...typing import Any
from ...typing import List
from ...typing import Optional
from ..base import BaseData


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

    def transform(self, transformation: Any, cache_dir: str):
        """Apply a transformation to the loaded data."""
        self.plogger.info(f"Before transformation:\n{self.stats()}")
        self.plogger.info(f"Applying transformation:\n{transformation}")
        self.iapply(transformation, progress_bar=True)
        self.plogger.info(f"After transformation:\n{self.stats()}")

        out_path = osp.join(cache_dir, "data.gmt")
        self.export_gmt(out_path)
        self.plogger.info(f"Saved cache transformation to {out_path}")

    def load_processed_data(self, path: Optional[str] = None):
        """Load processed labels from GMT."""
        path = path or self.processed_file_path(0)
        self.plogger.info(f"Load processed file {path}")
        self.read_gmt(path, reload=True)
