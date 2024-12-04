"""PyTorch Geometric dataset object."""

import os.path as osp

try:
    import torch
    from torch_geometric.data import InMemoryDataset
except (ModuleNotFoundError, OSError):
    InMemoryDataset = object

from obnb.alltypes import Callable, LogLevel, Optional
from obnb.dataset import OpenBiomedNetBench
from obnb.util.logger import verbose
from obnb.util.version import parse_data_version


class OpenBiomedNetBenchPyG(InMemoryDataset):
    """PyTorch Geometric default dataset construct.

    Args:
        root: Root directory of the dataset to be saved.
        network: Name of the network to use.
        label: Name of the gene annotation label to use.
        version: Version of the OpenBiomedNetBench data to use. By default,
            "current" means using current (archived) release. If specified as
            "latest", then download data from source and process them from
            scratch.
        log_level: Data downloading and processing verbosity.
        transform: PyG transforms to be applied.
        pre_transform: PyG transforms to be applied before saving.

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
        pre_transform: Optional[Callable] = None,
    ):
        if InMemoryDataset is object:
            raise ImportError(
                "OpenBiomedNetBenchPyG requires PyTorch and PyG, at least one of "
                "is currently missing.\nPlease follow the installation instructions "
                "on https://pytorch-geometric.readthedocs.io to install first.",
            )

        self.network = network
        self.label = label
        self.name = f"{network}-{label}"
        self.version = parse_data_version(version)
        self.log_level = log_level

        super().__init__(root, transform, pre_transform, log=verbose(log_level))
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __repr__(self) -> str:
        """Return the representation containing data size and names."""
        paramstr = f"{len(self):,}, network={self.network}, label={self.label}"
        return f"{self.__class__.__name__}({paramstr})"

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.__class__.__name__, self.name, "processed")

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def process(self):
        dataset = OpenBiomedNetBench(
            root=self.root,
            graph_name=self.network,
            label_name=self.label,
            version=self.version,
            log_level=self.log_level,
        )
        data = dataset.to_pyg_data()

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
