import mygene

from ..typing import Dict
from ..typing import List


class MyGeneInfoConverter:
    """Gene ID conversion via MyGeneInfo."""

    def __init__(self, species: str = "human", **query_kwargs):
        """Initialize the converter.

        Args:
            species (str): comma-separated species name of interest
            query_kwargs: Other keyword arguments for
                mygene.MyGeneInfo.querymany

        """
        self.client = mygene.MyGeneInfo()
        self.convert_map: Dict[str, str] = {}
        self.species = species
        self.query_kwargs = query_kwargs

    def __call__(self, old_id: str) -> str:
        """Convert an ID to entrez gene ID.

        Args:
            old_id (str): gene/protein ID to be converted.

        Returns:
            str: Entrez gene ID.

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
            self.convert_map[gene] = query.get("entrezgene")
