import mygene

from ..typing import List


class MyGeneInfoConverter:
    """Gene ID conversion via MyGeneInfo."""

    def __init__(self):
        """Initialize the converter."""
        self.client = mygene.MyGeneInfo()
        self.convert_map = {}

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
        queries = self.client.getgenes(ids, fields="entrezgene")
        for query in queries:
            if "entrezgene" in query:
                self.convert_map[query["query"]] = query["entrezgene"]
