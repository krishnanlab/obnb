[![PyPI version](https://badge.fury.io/py/obnb.svg)](https://badge.fury.io/py/obnb)
[![Documentation Status](https://readthedocs.org/projects/obnb/badge/?version=latest)](https://obnb.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

[![Tests](https://github.com/krishnanlab/obnb/actions/workflows/tests.yml/badge.svg)](https://github.com/krishnanlab/obnb/actions/workflows/tests.yml)
[![Test Examples](https://github.com/krishnanlab/obnb/actions/workflows/examples.yml/badge.svg)](https://github.com/krishnanlab/obnb/actions/workflows/examples.yml)
[![Test Data](https://github.com/krishnanlab/obnb/actions/workflows/test_data.yml/badge.svg)](https://github.com/krishnanlab/obnb/actions/workflows/test_data.yml)

# Open Biomedical Network Benchmark

## Installation

Clone the repository first and then install via `pip`

```bash
git clone https://github.com/krishnanlab/obnb && cd obnb
pip install -e .
```

The `-e` option means 'editable', i.e. no need to reinstall the library if you make changes to the source code.
Feel free to not use the `-e` option and simply do `pip install .` if you do not plan on modifying the source code.

### Optional Pytorch Geometric installation

User need to install [Pytorch Geomtric](https://github.com/pyg-team/pytorch_geometric) to enable some GNN related features.
To install PyG, first need to install [PyTorch](https://pytorch.org).
For full details about installation instructions, visit the links above.
Assuming the system has Python3.8 or above installed, with CUDA10.2, use the following to install both PyTorch and PyG.

```bash
conda install pytorch=1.12.1 torchvision cudatoolkit=10.2 -c pytorch
pip install torch-geometric==2.0.4 torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
```

### Quick install using the installation script

```bash
source install.sh cu102  # other options are [cpu,cu113]
```

## Quick Demonstration

### Construct default datasets

We provide a high-level dataset constructor to help user effortlessly set up a ML-ready dataset
for a combination of network and label. In particular, the dataset will be set up with study-bias
holdout split (6/2/2), where 60% of the most well studied genes according to the number of
associated PubMed publications are used for training, 20% of the least studied genes are used for
testing, and rest of the 20% genes are used for validation. For more customizable data loading
and processing options, see the [customized dataset construction](#customized-dataset-construction)
section below.

```python
from obnb.dataset import OpenBiomedNetBench

root = "datasets"  # save dataset and cache under the datasets/ directory
version = "current"  # use the last archived version
# Optionally, set version to the specific data version number
# Or, set version to "latest" to download the latest data from source and process it from scratch

# Download and process network/label data. Use the adjacency matrix as the ML feature
dataset = OpenBiomedNetBench(root=root, graph_name="BioGRID", label_name="DisGeNET",
                             version=version, graph_as_feature=True, use_dense_graph=True)
```

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
g = data.BioGRID(root, version="current")

# Load DisGeNET gene set collections.
lsc = data.DisGeNET(root, version="current")
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

## Data preparation and releasing notes

First, bump data version in `__init__.py` to the next data release version, e.g., `obnbdata-v0.1.0 -> obnbdata-v0.1.1-dev`.
Then, download and process all latest data by running

```bash
python script/release_data.py
```

By default, the data ready to be uploaded (e.g., to [Zenodo](zenodo.org)) is saved under `data_release/archived`.
After some necessary inspection and checking, if everything looks good, upload and publish the new archived data.

**Note:** `dev` data should be uploaded to the [sandbox](https://sandbox.zenodo.org/record/1097545#.YxYrqezMJzV) instead.

Check items:

- [ ] Update `__data_version__`
- [ ] Run [`release_data.py`](script/release_data.py)
- [ ] Upload archived data to Zenodo (be sure to edit the data version there also)
- [ ] Update url dict in config (will improve in the future to get info from Zenodo directly)
- [ ] Update network stats in data [test](test/test_data.py)

Finally, commit and push the bumped version.
