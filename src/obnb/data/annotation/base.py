from obnb.data.base import BaseData
from obnb.alltypes import List, Optional, ZipType
from obnb.util.download import download_unzip


class BaseAnnotationData(BaseData):
    CONFIG_KEYS: List[str] = BaseData.CONFIG_KEYS + ["annotation_url"]
    annotation_url: Optional[str] = None
    annotation_file_name: Optional[str] = None
    annotation_file_zip_type: ZipType = "gzip"

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
        download_unzip(
            self.annotation_url,
            self.raw_dir,
            zip_type=self.annotation_file_zip_type,
            rename=self.raw_files[0],
            logger=self.plogger,
        )

    def process(self):
        # NOTE: we process the ontology graph from raw file directly, so we
        # do not need to pre-process and save the processed file.
        pass

    def process_completed(self) -> bool:
        return True

    def load_processed_data(self, path: Optional[str] = None):
        # To be implemented in the child class.
        raise NotImplementedError
