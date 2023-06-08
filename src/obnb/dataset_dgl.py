"""DGL dataset object."""
import os.path as osp

from dgl import load_graphs, save_graphs
from dgl.data import DGLDataset
from dgl.data.utils import load_info, save_info

from obnb import __data_version__
from obnb.typing import Callable, LogLevel, Optional
from obnb.util.dataset_constructors import default_constructor
from obnb.util.logger import verbose


class OpenBiomedNetBenchDGL(DGLDataset):
    """DGL default dataset construct.

    Args:
        root: Root directory of the dataset to be saved.
        network: Name of the network to use.
        label: Name of the gene annotation label to use.
        version: Version of the OpenBiomedNetBench data to use. By default,
            "current" means using current (archived) release. If specified as
            "latest", then download data from source and process them from
            scratch.
        log_level: Data downloading and processing verbosity.
        transform: DGL transforms to be applied to the DGL graph.

    """

    def __init__(
        self,
        root: str,
        network: str,
        label: str,
        *,
        version: str = "current",
        log_level: LogLevel = "INFO",
        transform: Optional[Callable] = None,
    ):
        self.root = root
        self.network = network
        self.label = label
        self.version = __data_version__ if version == "current" else version
        self.log_level = log_level
        super().__init__(
            name=f"{network}-{label}",
            save_dir=osp.join(root, self.__class__.__name__),
            verbose=verbose(log_level),
            transform=transform,
        )

    def process(self):
        dataset = default_constructor(
            self.root,
            version=self.version,
            graph_name=self.network,
            label_name=self.label,
            log_level=self.log_level,
        )
        self._graph = dataset.to_dgl_data()

    @property
    def processed_graph_path(self) -> str:
        return osp.join(self.save_path, "dgl_graph.bin")

    @property
    def processed_info_path(self) -> str:
        return osp.join(self.save_path, "info.pkl")

    def save(self):
        save_graphs(self.processed_graph_path, self._graph)
        save_info(
            self.processed_info_path,
            {"node_ids": self._graph.node_ids, "task_ids": self._graph.task_ids},
        )

    def load(self):
        graphs, _ = load_graphs(self.processed_graph_path)
        self._graph = graphs[0]
        info = load_info(self.processed_info_path)
        self._graph.node_ids = info["node_ids"]
        self._graph.task_ids = info["task_ids"]

    def has_cache(self) -> bool:
        has_graph = osp.exists(self.processed_graph_path)
        has_info = osp.exists(self.processed_info_path)
        return has_graph and has_info

    def __getitem__(self, idx: int = 0):
        return self._graph

    def __len__(self):
        return 1
