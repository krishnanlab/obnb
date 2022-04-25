import mygene

from ..typing import Dict
from ..typing import List
from ..typing import LogLevel
from ..typing import Optional
from ..util.logger import get_logger


class MyGeneInfoConverter:
    """Gene ID conversion via MyGeneInfo."""

    def __init__(
        self,
        log_level: LogLevel = "INFO",
        remove_multimap: bool = True,
        species: str = "human",
        **query_kwargs,
    ):
        """Initialize the converter.

        Args:
            log_level: Logging level.
            remove_multimap (bool): If set to True, then ignore any gene
                that is mapped to multiple entrez gene id.
            species (str): comma-separated species name of interest
            query_kwargs: Other keyword arguments for
                mygene.MyGeneInfo.querymany

        """
        self.client = mygene.MyGeneInfo()
        self.convert_map: Dict[str, Optional[str]] = {}

        self.logger = get_logger(self.__class__.__name__, log_level=log_level)

        self.remove_multimap = remove_multimap
        self.species = species
        self.query_kwargs = query_kwargs

    def __call__(self, old_id: str) -> Optional[str]:
        """Convert an ID to entrez gene ID.

        Args:
            old_id (str): gene/protein ID to be converted.

        Returns:
            str: Entrez gene ID, or None if not available.

        """
        try:
            new_id = self.convert_map[old_id]
        except KeyError:
            info = self.client.getgene(old_id, fields="entrezgene")
            new_id = (
                info["entrezgene"]
                if info is not None and "entrezgene" in info
                else None
            )
            self.convert_map[old_id] = new_id
        return new_id

    def query_bulk(self, ids: List[str]):
        """Query gene IDs in bulk for performnace."""
        queries = self.client.querymany(
            ids,
            entrezonly=True,
            fields="entrezgene",
            species=self.species,
            **self.query_kwargs,
        )
        for query in queries:
            gene = query["query"]
            gene_id = query.get("entrezgene")
            if gene in self.convert_map:
                if self.remove_multimap:
                    self.convert_map[gene] = None
                    self.logger.info(
                        f"Removing {gene} due to multiple entrez mapping.",
                    )
                    continue

                old_gene_id = self.convert_map[gene]
                self.logger.info(
                    f"Overwriting {gene} -> {old_gene_id} to "
                    f"{gene} -> {gene_id}",
                )
            self.convert_map[gene] = gene_id
