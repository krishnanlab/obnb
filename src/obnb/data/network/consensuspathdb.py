import itertools
import re

import numpy as np
import pandas as pd

from obnb.data.network.base import BaseURLSparseGraphData
from obnb.alltypes import List, Literal, Mapping, Optional, Union
from obnb.util.download import download_unzip


class ConsensusPathDB(BaseURLSparseGraphData):
    """The ConsensusPathDB interaction network.

    The `ConsensusPathDB <http://cpdb.molgen.mpg.de/>`_ integrates gene
    interaction evidences from many databases:

        - ``BIND``
        - ``BioCarta``
        - ``Biogrid``
        - ``CORUM``
        - ``DIP``
        - ``HPRD``
        - ``HumanCyc``
        - ``INOH``
        - ``InnateDB``
        - ``IntAct``
        - ``MINT``
        - ``MIPS-MPPI``
        - ``Manual upload``
        - ``MatrixDB``
        - ``NetPath``
        - ``PDB``
        - ``PDZBase``
        - ``PID``
        - ``PINdb``
        - ``PhosphoPOINT``
        - ``Reactome``
        - ``Spike``

    These sources cover a wide range of interaction tyeps:

        - Protein interactions
        - Signaling reactions
        - Metabolic reactions
        - Gene regulations
        - Genetic interactions
        - Drug-target interactions
        - Biochemical pathways

    Check out the `ConsensusPathDB <http://cpdb.molgen.mpg.de/>`_ webpage for
    more information about the specific types of interactions provided by each
    source databases.

    **[Last updated: 2023-02-13]**

    """

    url = "http://cpdb.molgen.mpg.de/download/ConsensusPathDB_human_PPI.gz"
    selected_sources: List[str] = [
        "BIND",
        "BioCarta",
        "Biogrid",
        "CORUM",
        "DIP",
        "HPRD",
        "HumanCyc",
        "INOH",
        "InnateDB",
        "IntAct",
        "MINT",
        "MIPS-MPPI",
        "Manual upload",
        "MatrixDB",
        "NetPath",
        "PDB",
        "PDZBase",
        "PID",
        "PINdb",
        "PhosphoPOINT",
        "Reactome",
        "Spike",
    ]

    def __init__(
        self,
        root: str,
        weighted: bool = True,
        directed: bool = False,
        largest_comp: bool = True,
        gene_id_converter: Optional[Union[Mapping[str, str], str]] = None,
        fill_value: Literal["mean", "max"] = "max",
        **kwargs,
    ):
        """Initialize the ConsensusPathDB object."""
        self.fill_value = fill_value
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
        download_unzip(
            self.url,
            self.raw_dir,
            zip_type=self.download_zip_type,
            rename=self.raw_files[1],
            logger=self.plogger,
        )

        # Load interaction table
        df = pd.read_csv(
            self.raw_file_path(1),
            sep="\t",
            comment="#",
            header=0,
            names=[
                "source_db",
                "publications",
                "uniprot_entry",
                "uniprot_id",
                "gene_name",
                "hgnc_id",
                "entrez",
                "ensg",
                "score",
            ],
        )

        # Filter by sources
        df = df[df["source_db"].str.contains("|".join(self.selected_sources))]

        # Fill in missing interaction weights
        if self.fill_value == "mean":
            fill_value = np.nanmean(df["score"].values)
        elif self.fill_value == "max":
            fill_value = np.nanmax(df["score"].values)
        else:
            raise ValueError(
                f"Unknown fill value option {self.fill_value}, "
                "supported options are 'mean' and 'max'",
            )
        df["score"].fillna(fill_value, inplace=True)

        # Construct interactions to undirected edges
        df = df[~pd.isna(df["entrez"])]
        edges = []
        for genes, score in df[["entrez", "score"]].values:
            genes = re.split(r",|\.", genes)
            genes = list(filter(None, genes))  # remove empty string
            if len(genes) < 2:  # discard self-loops
                continue

            # Prepare edge list: [(gene1, gene2, score), ...]
            edges.extend(i + (score,) for i in itertools.combinations(genes, 2))
        edge_df = pd.DataFrame(edges)

        # Make undirected by filling in the connections from reversed direction
        edge_df = pd.concat((edge_df, edge_df.rename(columns={0: 1, 1: 0})))
        self.plogger.info(f"Converted interactions to edge list:\n{edge_df}")

        # Drop duplicated edges and keep the largest weight
        edge_df = (
            edge_df.sort_values(2, ascending=False)
            .drop_duplicates([0, 1])
            .sort_values([0, 1])
            .reset_index(drop=True)
        )
        self.plogger.info(f"Dropped duplicates:\n{edge_df}")

        out_path = self.raw_file_path(0)
        edge_df.to_csv(out_path, sep="\t", index=False, header=None)
        self.plogger.info(f"Cleaned raw file saved to {out_path}")
