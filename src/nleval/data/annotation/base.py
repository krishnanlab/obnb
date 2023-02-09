import gzip
import os.path as osp

import pandas as pd

from nleval.data.base import BaseData
from nleval.typing import List, Optional
from nleval.util.download import stream_download


class BaseAnnotationData(BaseData):
    CONFIG_KEYS: List[str] = BaseData.CONFIG_KEYS + ["annotation_url"]
    annotation_url: Optional[str] = None
    annotation_file_name: Optional[str] = None

    def __init__(self, root: str, **kwargs):
        """Initialize BaseAnnotationData."""
        super().__init__(root, **kwargs)

    @property
    def raw_files(self) -> List[str]:
        if not isinstance(self.annotation_file_name, str):
            raise ValueError("Annotation file name not specified.")
        return [self.annotation_file_name]

    @property
    def processed_files(self) -> List[str]:
        return ["annotation.csv"]

    def download(self):
        """Download raw annotation table.

        Note:
            The raw file is assumed to be gzipped.

        """
        self.plogger.info(f"Download annotation from: {self.annotation_url}")
        content = stream_download(self.annotation_url, log_level=self.log_level)[1]
        with open(osp.join(self.raw_dir, self.annotation_file_name), "wb") as f:
            f.write(gzip.decompress(content))

    def load_processed_data(self, path: Optional[str] = None):
        """Load processed annotation table.

        The annotation table is a csv file with two columns: ``gene_id`` and
        ``term_id``.

        """
        path = path or self.processed_file_path(0)
        self.plogger.info(f"Load processed annodataion {path}")
        self.data = pd.read_csv(path)
