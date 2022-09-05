[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/krishnanlab/NetworkLearningEval/actions/workflows/tests.yml/badge.svg)](https://github.com/krishnanlab/NetworkLearningEval/actions/workflows/tests.yml)
[![Test Examples](https://github.com/krishnanlab/NetworkLearningEval/actions/workflows/examples.yml/badge.svg)](https://github.com/krishnanlab/NetworkLearningEval/actions/workflows/examples.yml)
[![Test Data](https://github.com/krishnanlab/NetworkLearningEval/actions/workflows/test_data.yml/badge.svg)](https://github.com/krishnanlab/NetworkLearningEval/actions/workflows/test_data.yml)

# NetworkLearningEval

## Installation

Clone the repository first and then install via ``pip``

```bash
git clone https://github.com/krishnanlab/NetworkLearningEval && cd NetworkLearningEval
pip install -e .
```

The ``-e`` option means 'editable', i.e. no need to reinstall the library if you make changes to the source code.
Feel free to not use the ``-e`` option and simply do ``pip install .`` if you do not plan on modifying the source code.

### Optional Pytorch Geometric installation

User need to install [Pytorch Geomtric](https://github.com/pyg-team/pytorch_geometric) to enable some GNN related features.
To install PyG, first need to install [PyTorch](https://pytorch.org).
For full details about installation instructions, visit the links above.
Assuming the system has Python3.8 or above installed, with CUDA10.2, use the following to install both PyTorch and PyG.

```bash
conda install pytorch=1.12.1 torchvision cudatoolkit=10.2 -c pytorch
pip install torch-geometric==2.0.4 torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
```

### Quick install using the installatino script

```bash
source install.sh cu102  # other options are [cpu,cu113]
```

## Quick Demonstration

### Load network and labels

```python
from NLEval import data

root = "datasets"  # save dataset and cache under the datasets/ directory

# Load processed BioGRID data from archive.
# Alternatively, set version="latest" to get and process the newest data from scratch.
g = data.BioGRID(root, version="nledata-v0.1.0-dev1")

# Load DisGeNet gene set collections.
lsc = data.DisGeNet(root, version="latest")
```

### Setting up data and splits

```python
from NLEval import Dataset
from NLEval.util.converter import GenePropertyConverter
from NLEval.label.split import RatioHoldout

# Load PubMed count gene propery converter and use it to set up study-bias holdout split
pubmedcnt_converter = GenePropertyConverter(root, name="PubMedCount")
splitter = RatioHoldout(0.6, 0.4, ascending=False, property_converter=pubmedcnt_converter)
y, masks = lsc.split(splitter, target_ids=g.node_ids, labelset_name=label_id, consider_negative=True)

# Combine everything into a dataset object
dataset = Dataset(graph=g, feature=g.to_dense_graph().to_feature(), y=y, masks=masks)
```

### Evaluating models on the processed dataset

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from NLEval.model.label_propagation import OneHopPropagation
from NLEval.model_trainer import SupervisedLearningTrainer, LabelPropagationTrainer

# Specify model(s) and metrics
sl_mdl = LogisticRegression(penalty="l2", solver="lbfgs")
lp_mdl = OneHopPropagation()
metrics = {"auroc": roc_auc_score}

sl_results = SupervisedLearningTrainer(metrics).train(sl_mdl, dataset)
lp_results = LabelPropagationTrainer(metrics).train(lp_mdl, dataset)
```

### Evaluating GNN models on the processed dataset

```python
from torch_geometric.nn import GCN
from NLEval.model_trainer.gnn import SimpleGNNTrainer

# Prepare study-bias holdout split on the whole geneset collection, do not consider defined negatives
y, masks = lsc.split(splitter, target_ids=g.node_ids, consider_negative=False)
dataset = Dataset(graph=g, feature=g.to_dense_graph().to_feature(), y=y, masks=masks)

# Evaluate GCN on the whole geneset collection
gcn_mdl = GCN(in_channels=1, hidden_channels=64, num_layers=5, out_channels=n_tasks)
gcn_results = SimpleGNNTrainer(metrics, device="cuda", metric_best="auroc").train(mdl, dataset)
```

## Dev notes

### Dev installation

```bash
pip install -r requirements.txt  # install dependencies with pinned version
pip install -e ".[dev]"  # install extra dependencies for dev
```

### Testing

Run ``pytest`` to run all tests

```bash
pytest
```

Run type checks and coding style checks using mypy and flake8 via tox:

```bash
$ tox -e mypy,flake8
```

### Data preparation and releasing

First, bump data version in ``__init__.py`` to the next data release version, e.g., ``nledata-v0.1.0 -> nledata-v0.1.1-dev``.
Then, download and process all latest data by running

```bash
python script/release_data.py
```

By default, the data ready to be uploaded (e.g., to [Zenodo](zenodo.org)) is saved under ``data_release/archived``.
After some necessary inspection and checking, if everything looks good, upload and publish the new archived data.

**Note:** ``dev`` data should be uploaded to the [sandbox](https://sandbox.zenodo.org/record/1097545#.YxYrqezMJzV) instead.

Finally, commit and push the bumped version.
