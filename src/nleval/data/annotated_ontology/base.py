import gzip
import os.path as osp

from nleval.data.base import BaseData
from nleval.label import LabelsetCollection
from nleval.typing import Any, List, Optional
from nleval.util.download import stream_download
from nleval.util.logger import display_pbar


class BaseAnnotatedOntologyData(BaseData, LabelsetCollection):
    """General object for labelset collection from annotated ontology."""

    CONFIG_KEYS: List[str] = BaseData.CONFIG_KEYS + ["ontology_url", "annotation_url"]
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
        """Download ontology from the OBO Foundry."""
        self.plogger.info(f"Download obo from: {self.ontology_url}")
        content = stream_download(self.ontology_url, log_level=self.log_level)[1]
        with open(self.ontology_file_path, "wb") as f:
            f.write(content)

    def download_annotations(self):
        """Download annotations."""
        self.plogger.info(f"Download annotation from: {self.annotation_url}")
        content = stream_download(self.annotation_url, log_level=self.log_level)[1]
        with open(osp.join(self.raw_dir, self.annotation_file_name), "wb") as f:
            f.write(gzip.decompress(content))

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
