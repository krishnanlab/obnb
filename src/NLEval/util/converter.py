import json
import os

import mygene

from NLEval.typing import Dict, Iterator, List, LogLevel, Optional
from NLEval.util.checkers import checkType
from NLEval.util.logger import get_logger


class MyGeneInfoConverter:
    """Gene ID conversion via MyGeneInfo."""

    def __init__(
        self,
        *,
        root: Optional[str] = None,
        use_cache: bool = True,
        save_cache: bool = True,
        log_level: LogLevel = "INFO",
        remove_multimap: bool = True,
        species: str = "human",
        **query_kwargs,
    ):
        """Initialize the converter.

        Args:
            root (str, optional): Root data directory for caching gene ID
                conversion mapping. If None, do not save cache.
            use_cache (bool): If set to True, then use cached gene conversion,
                which is found under <root>/.cache/mygene_convert.json
            save_cache (bool): If set to True, then save cache after query_bulk
                is called. The conversion mappings are merged with existing
                cache if available.
            log_level: Logging level.
            remove_multimap (bool): If set to True, then ignore any gene
                that is mapped to multiple entrez gene id.
            species (str): comma-separated species name of interest
            query_kwargs: Other keyword arguments for
                mygene.MyGeneInfo.querymany

        """
        self.client = mygene.MyGeneInfo()
        self.convert_map: Dict[str, Optional[str]] = {}

        self.root = root
        self.use_cache = use_cache
        self.save_cache = save_cache
        self.logger = get_logger(self.__class__.__name__, log_level=log_level)

        self.remove_multimap = remove_multimap
        self.species = species
        self.query_kwargs = query_kwargs

    @property
    def root(self) -> Optional[str]:
        """Cache data directory."""
        return self._root

    @root.setter
    def root(self, root: Optional[str]):
        if root is not None:
            cache_dir = os.path.join(root, ".cache")
            os.makedirs(cache_dir, exist_ok=True)
            root = cache_dir
        self._root = root

    @property
    def cache_path(self) -> Optional[str]:
        """Cached gene conversion file path."""
        return (
            None
            if self.root is None
            else os.path.join(self.root, "mygene_convert.json")
        )

    def __getitem__(self, old_id: str) -> Optional[str]:
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

    def __len__(self) -> int:
        """Number of genes converted."""
        return len(self.convert_map)

    def __iter__(self) -> Iterator[Optional[str]]:
        """Iterate over genes converted."""
        yield from self.convert_map

    def _load_cache(self):
        if not self.use_cache:
            return

        if self.root is None:
            self.logger.warning(
                "load_cache option set but root directory not defined, "
                "skipping cache loading.",
            )
            return

        try:
            with open(self.cache_path, "r") as f:
                self.convert_map = json.load(f)
            self.logger.info(f"Loaded gene conversion cache {self.cache_path}")
        except FileNotFoundError:
            self.logger.info(f"Cache file not yet available {self.cache_path}")

    def _save_cache(self):
        if not self.save_cache:
            return

        if self.root is None:
            self.logger.warning(
                "save_cache option set but root directory not defined, "
                "skipping cache saving.",
            )
            return

        full_convert_map = {}
        if os.path.isfile(self.cache_path):
            with open(self.cache_path, "r") as f:
                full_convert_map = json.load(f)

        for i, j in self.convert_map.items():
            if i in full_convert_map and j != full_convert_map[i]:
                self.logger.error(
                    f"Conflicting mapping for {i!r}, previously mapped to"
                    f"{full_convert_map[i]!r}, overwritting to {j!r})",
                )
            full_convert_map[i] = j

        with open(self.cache_path, "w") as f:
            json.dump(full_convert_map, f, indent=4, sort_keys=True)
        self.logger.info(f"Gene conversion cache saved {self.cache_path}")

    def query_bulk(
        self,
        ids: List[str],
    ):
        """Query gene IDs in bulk for performnace."""
        self._load_cache()

        ids_set = set(ids)
        ids_to_query = ids_set.difference(self.convert_map)
        self.logger.info(
            f"Total number of genes: {len(ids):,} ({len(ids_set):,} unique)",
        )

        if ids_to_query:
            self.logger.info(
                f"Number of genes to be queried: {len(ids_to_query)}",
            )
            queries = self.client.querymany(
                ids_to_query,
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
                    self.logger.warning(
                        f"Overwriting {gene} -> {old_gene_id} to "
                        f"{gene} -> {gene_id}",
                    )
                self.convert_map[gene] = gene_id

            self._save_cache()

        else:
            self.logger.info("No query needed.")

    @classmethod
    def construct(cls, name: str, **kwargs):
        """Construct default converter based on name.

        Currently available options:

            - HumanEntrez

        """
        checkType("Name of converter", str, name)
        if name == "HumanEntrez":
            converter = cls(
                scopes="entrezgene,ensemblgene,symbol",
                species="human",
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown converter {name!r}.")

        return converter
