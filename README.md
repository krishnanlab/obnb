[![PyPI version](https://badge.fury.io/py/obnb.svg)](https://badge.fury.io/py/obnb)
[![Documentation Status](https://readthedocs.org/projects/obnb/badge/?version=latest)](https://obnb.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

[![Tests](https://github.com/krishnanlab/obnb/actions/workflows/tests.yml/badge.svg)](https://github.com/krishnanlab/obnb/actions/workflows/tests.yml)
[![Test Examples](https://github.com/krishnanlab/obnb/actions/workflows/examples.yml/badge.svg)](https://github.com/krishnanlab/obnb/actions/workflows/examples.yml)
[![Test Data](https://github.com/krishnanlab/obnb/actions/workflows/test_data.yml/badge.svg)](https://github.com/krishnanlab/obnb/actions/workflows/test_data.yml)

# Open Biomedical Network Benchmark

The Open Biomedical Network Benchmark (OBNB) is a comprehensive resource for setting up benchmarking graph datasets using _biomedical networks_ and _gene annotations_.
Our goal is to leverage advanced graph machine learning techniques, such as graph neural networks and graph embeddings, to accelerate the development of network biology for gaining insights into genes' function, trait, and disease associations using biological networks.
OBNB additionally provides dataset objects compatible with popular graph deep learning frameworks, including [PyTorch Geometric (PyG)](https://github.com/pyg-team/pytorch_geometric) and [Deep Graph Library (DGL)](https://github.com/dmlc/dgl).

A comprehensive benchmarking study with a wide-range of graph neural networks and graph embedding methods on OBNB datasets can be found in our benchmarking repository `[obnbench](https://github.com/krishnanlab/obnbench)`.

## Package usage

### Construct default datasets

We provide a high-level dataset constructor to help users easily set up benchmarking graph datasets
for a combination of network and label. In particular, the dataset will be set up with study-bias
holdout split (6/2/2), where 60% of the most well-studied genes according to the number of
associated PubMed publications are used for training, 20% of the least studied genes are used for
testing, and the rest of the 20% genes are used for validation. For more customizable data loading
and processing options, see the [customized dataset construction](#customized-dataset-construction)
section below.

```python
from obnb import __data_version__
from obnb.dataset import OpenBiomedNetBench

root = "datasets"  # save dataset and cache under the datasets/ directory
version = __data_version__  # use the last archived version (same as setting to "current")
# Optionally, set version to the specific data version number
# Or, set version to "latest" to download the latest data from source and process it from scratch

# Download and process network/label data. Use the adjacency matrix as the ML feature
dataset = OpenBiomedNetBench(root=root, graph_name="BioGRID", label_name="DisGeNET",
                             version=version, graph_as_feature=True, use_dense_graph=True)
```

Users can also load the dataset objects into ones that are compatible with PyG or DGL (see below).

#### PyG dataset

```python
from obnb.dataset import OpenBiomedNetBenchPyG
dataset = OpenBiomedNetBenchPyG(root, "BioGRID", "DisGeNET")
```

**Note**: requires installing PyG first (see [installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html))

#### DGL dataset

```python
from obnb.dataset import OpenBiomedNetBenchDGL
dataset = OpenBiomedNetBenchDGL(root, "BioGRID", "DisGeNET")
```

**Note**: requires installing DGL first (see [installation instructions](https://www.dgl.ai/pages/start.html))

### Evaluating standard models

Evaluation of simple machine learning methods such as logistic regression and label propagation
can be done easily using the trainer objects.

```python
from obnb.model_trainer import SupervisedLearningTrainer, LabelPropagationTrainer

sl_trainer = SupervisedLearningTrainer()
lp_trainer = LabelPropagationTrainer()
```

Then, use the `fit_and_eval` method of the trainer to evaluate a given ML model over all tasks
in a one-vs-rest setting.

```python
from sklearn.linear_model import LogisticRegression
from obnb.model.label_propagation import OneHopPropagation

# Initialize models
sl_mdl = LogisticRegression(penalty="l2", solver="lbfgs")
lp_mdl = OneHopPropagation()

# Evaluate the models over all tasks
sl_results = sl_trainer.fit_and_eval(sl_mdl, dataset)
lp_results = lp_trainer.fit_and_eval(lp_mdl, dataset)
```

### Evaluating GNN models

Training and evaluation of Graph Neural Network (GNN) models can be done in a very similar fashion.

```python
from torch_geometric.nn import GCN
from obnb.model_trainer.gnn import SimpleGNNTrainer

# Use 1-dimensional trivial node feature by default
dataset = OpenBiomedNetBench(root=root, graph_name="BioGRID", label_name="DisGeNET", version=version)

# Train and evaluate a GCN
gcn_mdl = GCN(in_channels=1, hidden_channels=64, num_layers=5, out_channels=n_tasks)
gcn_trainer = SimpleGNNTrainer(device="cuda", metric_best="apop")
gcn_results = gcn_trainer.train(gcn_mdl, dataset)
```

### Customized dataset construction

#### Load network and labels

```python
from obnb import data

root = "datasets"  # save dataset and cache under the datasets/ directory

# Load processed BioGRID data from archive.
g = data.BioGRID(root, version=version)

# Load DisGeNET gene set collections.
lsc = data.DisGeNET(root, version=version)
```

#### Setting up data and splits

```python
from obnb.util.converter import GenePropertyConverter
from obnb.label.split import RatioHoldout

# Load PubMed count gene property converter and use it to set up study-bias holdout split
pubmedcnt_converter = GenePropertyConverter(root, name="PubMedCount")
splitter = RatioHoldout(0.6, 0.4, ascending=False, property_converter=pubmedcnt_converter)
```

#### Filter labeled data based on network genes and splits

```python
# Apply in-place filters to the labelset collection
lsc.iapply(
    filters.Compose(
        # Only use genes that are present in the network
        filters.EntityExistenceFilter(list(g.node_ids)),
        # Remove any labelsets with less than 50 network genes
        filters.LabelsetRangeFilterSize(min_val=50),
        # Make sure each split has at least 10 positive examples
        filters.LabelsetRangeFilterSplit(min_val=10, splitter=splitter),
    ),
)
```

#### Combine into dataset

```python
from obnb import Dataset
dataset = Dataset(graph=g, feature=g.to_dense_graph().to_feature(), label=lsc, splitter=splitter)
```

## Installation

OBNB can be installed easily via pip from [PyPI](https://pypi.org/project/obnb/):

```bash
pip install obnb
```

### Install with extension modules (optional)

OBNB provides interfaces with several other packages for network feature extractions, such as
[PecanPy](https://github.com/krishnanlab/PecanPy) and [GraPE](https://github.com/AnacletoLAB/grape).
To enable those extensions, install `obnb` with the `ext` extra option enabled:

```bash
pip install obnb[ext]
```

### Install graph deep learning libraries (optional)

Follow installation instructions for [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) or [DGL](https://www.dgl.ai/pages/start.html) to set up the graph deep learning library of your choice.

Alternatively, we also provide an [installation script](install.sh) that helps you installthe graph deep-learning dependencies in a new conda environment `obnb`:

```bash
git clone https://github.com/krishnanlab/obnb && cd obnb
source install.sh cu117  # other options are [cpu,cu118]
```
