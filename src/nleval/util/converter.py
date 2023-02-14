import gzip
import json
import os
import os.path as osp
from ftplib import FTP
from io import BytesIO

import mygene
import pandas as pd

from nleval.typing import Any, Dict, Iterator, List, LogLevel, Optional
from nleval.util.checkers import checkType
from nleval.util.logger import get_logger


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
        self._load_cache()

    @property
    def root(self) -> Optional[str]:
        """Cache data directory."""
        return self._root

    @root.setter
    def root(self, root: Optional[str]):
        if root is not None:
            cache_dir = osp.join(root, ".cache")
            os.makedirs(cache_dir, exist_ok=True)
            root = cache_dir
        self._root = root

    @property
    def cache_path(self) -> Optional[str]:
        """Cached gene conversion file path."""
        return None if self.root is None else osp.join(self.root, self.cache_file_name)

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

    def _load_cache(self) -> bool:
        """Load cache file and return True if successfully loaded cache."""
        if not self.use_cache:
            return False

        if self.cache_path is None:
            self.logger.warning(
                "load_cache option set but root directory not defined, "
                "skipping cache loading.",
            )
            return False

        try:
            with open(self.cache_path) as f:
                self._convert_map = json.load(f)
            self.logger.info(f"Loaded gene conversion cache {self.cache_path}")
            return True
        except FileNotFoundError:
            self.logger.debug(f"Cache file not yet available {self.cache_path}")
            return False

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
        if osp.isfile(self.cache_path):
            with open(self.cache_path) as f:
                full_convert_map = json.load(f)

        for i, j in self._convert_map.items():
            if i in full_convert_map and j != full_convert_map[i]:
                self.logger.error(
                    f"Conflicting mapping for {i!r}, previously mapped to"
                    f"{full_convert_map[i]!r}, overwriting to {j!r})",
                )
            full_convert_map[i] = j

        with open(self.cache_path, "w") as f:
            json.dump(full_convert_map, f, indent=4, sort_keys=True)
        self.logger.info(f"Gene conversion cache saved {self.cache_path}")

    def query_bulk(self, ids: List[str]):
        raise NotImplementedError

    def map_df(
        self,
        df: pd.DataFrame,
        input_column: str,
        output_column: Optional[str] = None,
        copy: bool = False,
    ) -> pd.DataFrame:
        """Map the id conversion to a column of a dataframe.

        Args:
            df: Input dataframe.
            input_column: Column to use as conversion keys.
            output_column: Column to save the converted values.
            copy: If set to ``True``, then create a copy of the dataframe.
                Otherwise, modify the dataframe inplace.

        """
        if copy:
            df = df.copy()

        if output_column is None:
            output_column = input_column

        try:
            to_query = df[input_column].unique().tolist()
            self.query_bulk(to_query)
        except NotImplementedError:
            self.logger.warning(f"{type(self)} do not support bulk query yet")

        converted = {i for i in to_query if self[i] is not None}
        num_succ = len(converted)
        num_failed = len(to_query) - len(converted)
        self.logger.info(f"Successfully mapped {num_succ:,} ({num_failed:,} failed)")

        ind = df[input_column].isin(converted)
        num_removed = ind.shape[0] - ind.sum()
        self.logger.info(f"{num_removed:,} entries removed by due to conversion.")
        df.drop(ind[~ind].index, inplace=True)

        # XXX: Assumes all mappings are one-to-one. Need to adapt to one-to-many.
        df[output_column] = df[input_column].map(self.__getitem__)
        df.reset_index(inplace=True)

        return df


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
        """Query gene IDs in bulk for performance."""
        ids_set = set(ids)
        ids_to_query = ids_set.difference(self._convert_map)
        self.logger.info(
            f"Total number of genes: {len(ids):,} ({len(ids_set):,} unique)",
        )

        if ids_to_query:
            self.logger.info(
                f"Number of genes to be queried: {len(ids_to_query):,}",
            )
            queries = self.client.querymany(
                ids_to_query,
                entrezonly=True,
                fields="entrezgene",
                species=self.species,
                **self.query_kwargs,
            )

            num_mod = 0
            for query in queries:
                gene = query["query"]
                gene_id = query.get("entrezgene")
                if gene in self._convert_map:
                    if self.remove_multimap:
                        self._convert_map[gene] = None
                        self.logger.debug(
                            f"Removing {gene} due to multiple entrez mapping.",
                        )
                    else:
                        old_gene_id = self._convert_map[gene]
                        self.logger.warning(
                            f"Overwriting {gene} -> {old_gene_id} to "
                            f"{gene} -> {gene_id}",
                        )
                    num_mod += 1
                self._convert_map[gene] = gene_id

            self._save_cache()

            if num_mod > 0:
                mod_name = "removed" if self.remove_multimap else "overwritten"
                self.logger.info(f"{num_mod} mappings {mod_name} during this update.")

        else:
            self.logger.info("No query needed.")

    @classmethod
    def construct(cls, name: str, **kwargs):
        """Construct default converter based on name.

        Currently available options:

            - HumanEntrez

        """
        checkType("Name of converter", str, name)
        scopes = [
            "ensembl.protein",
            "ensembl.gene",
            "ensembl.transcript",
            "entrezgene",
            "uniprot",
            "accession",
            "symbol",
            "alias",
        ]
        if name == "HumanEntrez":
            converter = cls(
                scopes=",".join(scopes),
                species="human",
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown converter {name!r}.")

        return converter


class GenePropertyConverter(BaseConverter):
    """Gene property data obtained from NCBI."""

    host: str = "ftp.ncbi.nlm.nih.gov"

    def __init__(
        self,
        root: Optional[str] = None,
        name: str = "PubMedCount",
        *,
        use_cache: bool = True,
        save_cache: bool = True,
        default_value: Any = "default",
        log_level: LogLevel = "INFO",
    ):
        """Initialize GenePropertyConverter.

        Args:
            root: Root data directory for caching gene ID conversion mapping.
                If None, do not save cache.
            name: Name of the property to use.
            use_cache: If set to True, then use cached gene conversion, which
                is found under <root>/.cache/mygene_convert.json
            save_cache: If set to True, then save cache after query_bulk is
                called. The conversion mappings are merged with existing cache
                if available.
            default_value: Default value to return if the property is
                unavailable for a particular entity. If set to 'default', will
                use the default value determined beforehand.
            log_level: Logging level.

        """
        self._default = default_value
        self.name = name
        super().__init__(
            root,
            use_cache=use_cache,
            save_cache=save_cache,
            log_level=log_level,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    @property
    def cache_file_name(self) -> str:
        return f"geneprop_convert-{self.name}.json"

    def _load_cache(self):
        if not super()._load_cache():
            self._get_data()
            self._save_cache()

    @property
    def name(self) -> str:
        """Name of the property."""
        return self._name

    @name.setter
    def name(self, name: str):
        if name == "PubMedCount":
            self._proc = lambda df: df["GeneID"].astype(str).value_counts().to_dict()
            self._raw_data_name = "gene2pubmed"
            self._default = 0 if self._default == "default" else self._default
        else:
            raise NotImplementedError(f"{name} gene property unavailable yet.")
        self._name = name

    def _get_data(self):
        with FTP(self.host) as ftp:
            self.logger.info(f"Retrieving {self._raw_data_name} from {self.host}")
            ftp.login()
            buf = BytesIO()
            ftp.retrbinary(f"RETR gene/DATA/{self._raw_data_name}.gz", buf.write)

        self.logger.info(f"Decompressing and loading {self._raw_data_name}")
        buf.seek(0)
        decomp = BytesIO(gzip.decompress(buf.read()))
        df = pd.read_csv(decomp, sep="\t")

        self._convert_map = self._proc(df)
        self.logger.info(f"Finished processing {self.name}")

    def __getitem__(self, query_id: str):
        return self._convert_map.get(query_id) or self._default
