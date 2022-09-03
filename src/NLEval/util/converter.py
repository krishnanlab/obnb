import json
import os

import mygene

from NLEval.typing import Any, Dict, Iterator, List, LogLevel, Optional
from NLEval.util.checkers import checkType
from NLEval.util.logger import get_logger


class BaseConverter:
    """BaseConverter object."""

    def __init__(
        self,
        root: Optional[str] = None,
        *,
        use_cache: bool = True,
        save_cache: bool = True,
        log_level: LogLevel = "INFO",
    ):
        """Initialize BaseConverter."""
        self.root = root
        self.logger = get_logger(self.__class__.__name__, log_level=log_level)

        self.use_cache = use_cache
        self.save_cache = save_cache

        self._convert_map: Dict[str, Optional[str]] = {}

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
            None if self.root is None else os.path.join(self.root, self.cache_file_name)
        )

    @property
    def cache_file_name(self) -> str:
        """Cache file name."""
        raise NotImplementedError

    def __getitem__(self, query_id: str) -> Any:
        """Return converted value given query ID."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of genes converted."""
        return len(self._convert_map)

    def __iter__(self) -> Iterator[Optional[str]]:
        """Iterate over genes converted."""
        yield from self._convert_map

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
                self._convert_map = json.load(f)
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

        for i, j in self._convert_map.items():
            if i in full_convert_map and j != full_convert_map[i]:
                self.logger.error(
                    f"Conflicting mapping for {i!r}, previously mapped to"
                    f"{full_convert_map[i]!r}, overwritting to {j!r})",
                )
            full_convert_map[i] = j

        with open(self.cache_path, "w") as f:
            json.dump(full_convert_map, f, indent=4, sort_keys=True)
        self.logger.info(f"Gene conversion cache saved {self.cache_path}")


class MyGeneInfoConverter(BaseConverter):
    """Gene ID conversion via MyGeneInfo."""

    def __init__(
        self,
        root: Optional[str] = None,
        *,
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
        super().__init__(
            root,
            use_cache=use_cache,
            save_cache=save_cache,
            log_level=log_level,
        )

        self.client = mygene.MyGeneInfo()

        self.remove_multimap = remove_multimap
        self.species = species
        self.query_kwargs = query_kwargs

    @property
    def cache_file_name(self) -> str:
        return "mygene_convert.json"

    def __getitem__(self, query_id: str) -> Optional[str]:
        """Convert an ID to entrez gene ID.

        Args:
            query_id (str): gene/protein ID to be converted.

        Returns:
            str: Entrez gene ID, or None if not available.

        """
        try:
            new_id = self._convert_map[query_id]
        except KeyError:
            info = self.client.getgene(query_id, fields="entrezgene")
            new_id = (
                info["entrezgene"]
                if info is not None and "entrezgene" in info
                else None
            )
            self._convert_map[query_id] = new_id
        return new_id

    def query_bulk(
        self,
        ids: List[str],
    ):
        """Query gene IDs in bulk for performnace."""
        self._load_cache()

        ids_set = set(ids)
        ids_to_query = ids_set.difference(self._convert_map)
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
                if gene in self._convert_map:
                    if self.remove_multimap:
                        self._convert_map[gene] = None
                        self.logger.info(
                            f"Removing {gene} due to multiple entrez mapping.",
                        )
                        continue

                    old_gene_id = self._convert_map[gene]
                    self.logger.warning(
                        f"Overwriting {gene} -> {old_gene_id} to "
                        f"{gene} -> {gene_id}",
                    )
                self._convert_map[gene] = gene_id

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
