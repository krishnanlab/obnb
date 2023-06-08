import io

import pandas as pd
import requests

from obnb.data.network.base import BaseURLSparseGraphData
from obnb.typing import Any, Dict, List, Mapping, Optional, Union


class ComPPI(BaseURLSparseGraphData):
    r"""The Compartmentalized Protein-Protein Interaction Database.

    The ComPPI database comes with interactomes with different contexts,
    including compartmentalization and species. To request download from the
    webserver, a `POST` request is send with the following options.

    - ``fDlSet``: What type of data to download, available options are

        - ``int``: Integrated protein-protein interactions across compartments.
        - ``comp``: Compartmentalized interactions.
        - ``protnloc``: Subcellular localization information of proteins (this
          is not interaction data).

    - ``fDlSpec``: What species to use, available options are

        - ``0``: H. sapiens (human).
        - ``1``: D. melanogaster (fruit fly).
        - ``2``: C.elegans (worm).
        - ``3``: S. cerevisiae (yeast).
        - ``all``: use all the above, the default option.

    - ``fDlMLoc``: What subcellular localization to use (do not specify when
      ``fDlSet`` is set to ``int``), available options are

        - ``0``: Cytosol.
        - ``1``: Mitochondrion.
        - ``2``: Nucleus.
        - ``3``: Extracellular.
        - ``4``: Secretory pathway.
        - ``5``: Membrane.
        - ``all``: Use all the above, the default option.

    Example:
        Request the file for integrated human interactom file and load into
        a pandas dataframe ``df`` via

        >>> r = requests.post("https://comppi.linkgroup.hu/downloads",
        ...                   data={"fDlSet": "int", "fDlSpec": "0"})
        >>> df = pd.read_csv(io.BytesIO(r.content), sep="\t",
        ...                  compression="gzip")

    **[Last updated: 2023-11-17]**

    """

    CONFIG_KEYS: List[str] = BaseURLSparseGraphData.CONFIG_KEYS + [
        "selected_columns",
    ]
    url: str = "https://comppi.linkgroup.hu/downloads"
    # TODO: parase args and setup at init
    url_kwargs: Dict[str, Any] = {}
    selected_columns: List[str] = ["Protein A", "Protein B", "Interaction Score"]

    def __init__(
        self,
        root: str,
        weighted: bool = True,
        directed: bool = False,
        largest_comp: bool = True,
        gene_id_converter: Optional[Union[Mapping[str, str], str]] = "HumanEntrez",
        **kwargs,
    ):
        """Initialize the CompPPI object."""
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
        self.plogger.info(
            f"Downloading data via POST from {self.url} with params: {self.url_kwargs}",
        )
        r = requests.post(self.url, **self.url_kwargs)

        self.plogger.info("Finished downloading, start unpacking...")
        df = pd.read_csv(io.BytesIO(r.content), sep="\t", compression="gzip")
        df_clean = df[self.selected_columns]

        clean_path, raw_path = self.raw_file_path(0), self.raw_file_path(1)
        df.to_csv(raw_path, sep="\t", index=False)
        self.plogger.info(f"Raw file saved to {raw_path}")
        df_clean.to_csv(clean_path, sep="\t", index=False, header=None)
        self.plogger.info(f"Cleaned raw file saved to {clean_path}")


class ComPPIHumanInt(ComPPI):
    """The ComPPI human integrated interaction network."""

    url_kwargs: Dict[str, Any] = {"data": {"fDlSet": "int", "fDlSpec": "0"}}
