import pyroe

from NLEval.data.base import BaseData
from NLEval.graph import FeatureVec
from NLEval.typing import Dict, List, Optional


class AlevinFry(BaseData, FeatureVec):
    """The AlevinFry scRNA-seq datasets.

    https://github.com/COMBINE-lab/alevin-fry

    """

    METADATA_KEYWORDS: List[str] = [
        "check_validity",
        "chemistry",
        "dataset_id",
        "dataset_name",
        "dataset_url",
        "decompress_quant",
        "delete_fastq",
        "fastq_MD5sum",
        "fastq_url",
        "feature_barcode_csv_url",
        "fetch_quant",
        "get_available_dataset_df",
        "load_quant",
        "multiplexing_library_csv_url",
        "print_available_datasets",
        "quant_path",
        "quant_tar_url",
        "reference",
        "tar_path",
    ]

    def __init__(
        self,
        root: str,
        dataset_id: int,  # TODO: add option to view data id -> name?
        quiet: bool = False,  # TODO: after captured to log, replace this w loglvl
        delete_tar: bool = False,
        **kwargs,
    ):
        """Initialize the AlevinFry data object.

        Args:
            root: The root directory of the data.
            dataset_id: The ID of the Alevin-Fry dataset (see more at
                https://github.com/COMBINE-lab/pyroe).
            quiet: If set to True, do not print any information to the screen
                about data downloading and processing.
            delete_art: If set to True, delete the tar ball file after the
                data has been extracted.

        """
        self.dataset_id = dataset_id
        self.quiet = quiet
        self.delete_tar = delete_tar
        self._metadata: Dict[str, str] = {}
        super().__init__(root, **kwargs)

    @property
    def metadata(self):
        return self._metadata

    def download_completed(self) -> bool:
        # Download completion check left to pyroe (fetch_processed_quant)
        return False

    def process_completed(self) -> bool:
        # Process completion check left to pyroe (load_processed_quant)
        return True

    def download(self):
        # TODO: capture prints and redirect to logger?
        pyroe.fetch_processed_quant(
            dataset_ids=[self.dataset_id],
            fetch_dir=self.processed_dir,
            force=self.redownload,
            delete_tar=self.delete_tar,
            quiet=self.quiet,
        )

    def _load_metadata(self, data):
        for key in self.METADATA_KEYWORDS:
            self._metadata[key] = getattr(data, key)

    def load_processed_data(self, path: Optional[str] = None):
        # TODO: capture prints and redirect to logger?
        dts_id = self.dataset_id
        data = pyroe.load_processed_quant(
            dataset_ids=[dts_id],
            fetch_dir=self.processed_dir,
            quiet=self.quiet,
        )[dts_id]

        self._load_metadata(data)
        # FIX: map to entrez genes
        # FIX: keep track of feature IDs (i.e., the gene IDs)
        self.read_anndata(data.anndata)
