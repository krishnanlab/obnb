import gzip

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

    def download(self):
        """Download raw annotation table.

        Note:
            The raw file is assumed to be gzipped.

        """
        self.plogger.info(f"Download annotation from: {self.annotation_url}")
        content = stream_download(self.annotation_url, log_level=self.log_level)[1]
        with open(self.raw_file_path(0), "wb") as f:
            f.write(gzip.decompress(content))

    def process(self):
        # NOTE: we process the ontology graph from raw file directly, so we
        # do not need to pre-process and save the processed file.
        pass

    def process_completed(self) -> bool:
        return True

    def load_processed_data(self, path: Optional[str] = None):
        # To be implemented in the child class.
        raise NotImplementedError
