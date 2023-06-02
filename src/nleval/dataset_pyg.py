"""PyTorch Geometric dataset object."""
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset

from nleval import __data_version__
from nleval.typing import Callable, List, LogLevel, Optional
from nleval.util.dataset_constructors import default_constructor


class OpenBiomedNetBench(InMemoryDataset):
    """PyTorch Geometric default dataset construct.

    Args:
        root: Root directory of the dataset to be saved.
        network: Name of the network to use.
        label: Name of the gene annotation label to use.
        selected_genes: An optional list of genes. When supplied, will be used
            to filter out genes in the label in addition to the filtering based
            on network genes.
        data_version: Version of the OpenBiomedNetBench data to use. If not
            specified, will use the current (archived) release. If specified as
            "latest", then download data from source and process them from
            scratch.
        log_level: Data downloading and processing verbosity.
        transform: PyG transformation to be applied.
        pre_transform: PyG transformation to be applied before saving.

    """

    def __init__(
        self,
        root: str,
        network: str,
        label: str,
        *,
        selected_genes: Optional[List[str]] = None,
        data_version: Optional[str] = None,
        log_level: LogLevel = "INFO",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.network = network
        self.label = label
        self.name = f"{network}-{label}"
        self.selected_genes = selected_genes
        self.data_version = data_version or __data_version__
        self.log_level = log_level

        super().__init__(root, transform, pre_transform)
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

    def get_raw_dataset(self, log_level: Optional[LogLevel] = None):
        return default_constructor(
            self.root,
            version=self.data_version,
            graph_name=self.network,
            label_name=self.label,
            log_level=log_level or self.log_level,
        )

    def process(self):
        dataset = self.get_raw_dataset()
        data = dataset.to_pyg_data()

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
