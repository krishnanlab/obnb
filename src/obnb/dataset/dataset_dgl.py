"""DGL dataset object."""
import os.path as osp

try:
    import torch
    from dgl import load_graphs, save_graphs
    from dgl.data import DGLDataset
    from dgl.data.utils import load_info, save_info
except (ModuleNotFoundError, OSError):
    DGLDataset = object

import obnb
from obnb.dataset import OpenBiomedNetBench
from obnb.typing import Callable, LogLevel, Optional
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
        if DGLDataset is object:
            raise ImportError(
                "OpenBiomedNetBenchDGL requires the DGL libary, which is currently "
                "missing.\nPlease follow the installation instructions on "
                "https://www.dgl.ai/pages/start.html to install the DGL library first",
            )

        self.root = root
        self.network = network
        self.label = label
        self.version = obnb.__data_version__ if version == "current" else version
        self.log_level = log_level
        super().__init__(
            name=f"{network}-{label}",
            save_dir=osp.join(root, self.__class__.__name__),
            verbose=verbose(log_level),
            transform=transform,
        )

    def process(self):
        dataset = OpenBiomedNetBench(
            root=self.root,
            graph_name=self.network,
            label_name=self.label,
            version=self.version,
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

        for key, val in self._graph.ndata.items():
            if key.endswith("_mask"):
                self._graph.ndata[key] = self._graph.ndata[key].to(dtype=torch.bool)

    def has_cache(self) -> bool:
        has_graph = osp.exists(self.processed_graph_path)
        has_info = osp.exists(self.processed_info_path)
        return has_graph and has_info

    def __getitem__(self, idx: int = 0):  # noqa: D105
        return self._graph

    def __len__(self):  # noqa: D105
        return 1
