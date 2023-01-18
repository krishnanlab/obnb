import io

import pandas as pd
import requests

from nleval.data.network.base import BaseURLSparseGraphData
from nleval.typing import List, Mapping, Optional, Union
from nleval.util.download import stream_download


class OmniPath(BaseURLSparseGraphData):
    """The OmniPath intra- dand inter-cellular signaling knowledge base.

    https://omnipathdb.org/

    - ``dorothea`` Interactions obtained from the DoRothEA database, which
      contains comprehensive resource of TF-promoter interactions curated from
      over 18 sources. Only the interactions with confidence from A-D are
      included in the OmniPath database.
    - ``kinaseextra`` Addition kinase-substrate interactions from prior
      knowledge.
    - ``ligrecextra`` Ligand-receptor interactions from prior knowledge.
    - ``lncrna_mrna`` Interactions between long non-coding RNAs and mRNAs,
      curated from three literatures.
    - ``mirnatarget`` Micro RNA target interactions.
    - ``omnipath`` Interaction information from literature curation, high
      throughput experiments, and prior knowledge.
    - ``pathwayextra`` Pathway information from prior konwledge.
    - ``small_molecule`` Small molecul protein interactions.
    - ``tf_mirna`` Transcription factor micro RNA interaction curated from
      two literature sources.
    - ``tf_target`` Transcription factor target curated from six literatures.
    - ``tfregulons`` Transcription factor regulon interacions.

    Note:
        ``Prior knolwedge`` means annotations done by the aurhors without any
        literature references.

    """

    url: str = "https://omnipathdb.org/interactions"
    omnipath_datasets: List[str] = [
        "dorothea",
        "kinaseextra",
        "ligrecextra",
        "lncrna_mrna",
        "mirnatarget",
        "omnipath",
        "pathwayextra",
        "small_molecule",
        "tf_mirna",
        "tf_target",
        "tfregulons",
    ]
    omnipath_fields: List[str] = [
        "curation_effort",
        "references",
        "sources",
        "type",
    ]
    selected_columns: List[str] = ["source", "target"]

    def __init__(
        self,
        root,
        weighted: bool = False,
        directed: bool = False,  # FIX: should be True, but need to fix LCC first
        largest_comp: bool = True,
        gene_id_converter: Optional[Union[Mapping[str, str], str]] = "HumanEntrez",
        **kwargs,
    ):
        """Initialize the OmniPath object."""
        super().__init__(
            root,
            weighted=weighted,
            directed=directed,
            largest_comp=largest_comp,
            gene_id_converter=gene_id_converter,
            **kwargs,
        )

    @property
    def raw_files(self) -> List[str]:
        return ["data_clean.txt", "data.txt"]

    def download(self):
        """Download data from URL."""
        # Prepare URL and parameters
        datasets_str = ",".join(self.omnipath_datasets)
        fields_str = ",".join(self.omnipath_fields)
        params = {"datasets": datasets_str, "fields": fields_str, "format": "tsv"}
        self.plogger.info(f"Base url: {self.url}")
        self.plogger.info(f"URL parameters: {params}")

        # Construct URL to obtain raw data
        req = requests.Request("GET", self.url, params=params)
        s = requests.Session()
        prepped_url = s.prepare_request(req).url
        self.plogger.info(f"Start download data from {prepped_url}")
        _, content = stream_download(prepped_url)

        # Download data from URL
        self.plogger.info("Finished downloading, start unpacking...")
        df = pd.read_csv(io.BytesIO(content), sep="\t")
        df_clean = df[self.selected_columns]

        # Save raw data
        clean_path, raw_path = self.raw_file_path(0), self.raw_file_path(1)
        df.to_csv(raw_path, sep="\t", index=False)
        self.plogger.info(f"Raw file saved to {raw_path}")
        df_clean.to_csv(clean_path, sep="\t", index=False, header=None)
        self.plogger.info(f"Cleaned raw file saved to {clean_path}")
