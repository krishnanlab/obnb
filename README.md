[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/krishnanlab/NetworkLearningEval/actions/workflows/tests.yml/badge.svg)](https://github.com/krishnanlab/NetworkLearningEval/actions/workflows/tests.yml)
[![Example Executions](https://github.com/krishnanlab/NetworkLearningEval/actions/workflows/run_examples.yml/badge.svg)](https://github.com/krishnanlab/NetworkLearningEval/actions/workflows/run_examples.yml)
[![Test Data](https://github.com/krishnanlab/NetworkLearningEval/actions/workflows/test_data.yml/badge.svg)](https://github.com/krishnanlab/NetworkLearningEval/actions/workflows/test_data.yml)

# NetworkLearningEval

## Installation

Clone the repository first and then install via ``pip``

```bash
git clone https://github.com/krishnanlab/NetworkLearningEval
cd NetworkLearningEval
pip install -e .
```

The ``-e`` option means 'editable', i.e. no need to reinstall the library if you make changes to the source code.
Feel free to not use the ``-e`` option and simply do ``pip install .`` if you do not plan on modifying the source code.

### Optional Pytorch Geometric installation

One can install [Pytorch Geomtric](https://github.com/pyg-team/pytorch_geometric) to enable some GNN related features.
To install PyG, first need to install [PyTorch](https://pytorch.org).
For full details about installation instructions, visit the links above.
Assuming the system has Python3.8 or above installed, with CUDA10.2, use the following to install both PyTorch and PyG.

```bash
conda install pytorch=1.9 torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install pyg -c pyg -c conda-forge
```

Note: To support some new features in PyG that are not yet released, need to install from the repo directly:
```bash
pip install git+https://github.com/pyg-team/pytorch_geometric.git#egg=torch-geometric[full]
```

### Full dev installation

```bash
conda create -n nle-dev-pyg python=3.9 -y && conda activate nle-dev-pyg
pip install -e .[dev]
conda install pytorch=1.11.0 torchvision=0.12.0 cudatoolkit=10.2 -c pytorch -y
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
pip install git+https://github.com/pyg-team/pytorch_geometric.git#egg=torch-geometric[full]
conda clean --all -y
```

## Quick Demonstration

### Load network and labels

Directly load the network and labels from online resouces.
Once it is processed and saved to the specified root directory, next execution would use the saved processed files directly.

```python
from NLEval import data

root = "datasets/"

# Load BioGRID and save to datasets/
g = data.BioGRID(root)

# Load DisGeNet and save to datasets/
lsc = data.DisGeNet(root)
```

Alternatively, could load from local files as shown in the following examplse.

### Setting up data and splits

```python
from NLEval import graph, label, model_trainer

# Load a weighted undirected graph from an edge list file
g = graph.DenseGraph.from_edgelist("data/networks/STRING-EXP.edg", weighted=True, directed=False)

# Load geneset collection from a GMT file
lsc = label.LabelsetCollection.from_gmt("data/labels/KEGGBP.gmt")

# Remove genes not present in the network from the geneset collection
lsc.iapply(label.filters.EntityExistenceFilter(g.node_ids))

# Remove small genesets
lsc.iapply(label.filters.LabelsetRangeFilterSize(min_val=50))

# Generate negatives via hypergeometric test
lsc.iapply(label.filters.NegativeGeneratorHypergeom(p_thresh=0.05))

# Load PubMed count and use it to setup study-bias holdout split
lsc.load_entity_properties("data/properties/PubMedCount.txt", "PubMed Count", 0, int)
splitter = RatioHoldout(0.6, 0.4, ascending=False)
```

### Evaluating models on the processed dataset
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from NLEval.model.label_propagation import OneHopPropagation
from NLEval.model_trainer import SupervisedLearningTrainer, LabelPropagationTrainer

# Prepare study-bias holdout split on a specific geneset, taking into account of defined negatives
y, masks = lsc.split(splitter, target_ids=g.node_ids, labelset_name=label_id, property_name="PubMedCount", consider_negative=True)

# Specify model(s) and metrics
sl_mdl = LogisticRegression(penalty="l2", solver="lbfgs")
lp_mdl = OneHopPropagation()
metrics = {"auroc": roc_auc_score}

sl_results = SupervisedLearningTrainer(metrics, g).train(sl_mdl, y, masks)
lp_results = LabelPropagationTrainer(metrics, g).train(lp_mdl, y, masks)
```

### Evaluating GNN models on the processed dataset
```python
from torch_geometric.nn import GCN
from NLEval.model_trainer.gnn import SimpleGNNTrainer

# Prepare study-bias holdout split on the whole geneset collection, do not consider defined negatives
y, masks = lsc.split(splitter, target_ids=g.node_ids, property_name="PubMedCount")

# Evaluate GCN on the whole geneset collection
gcn_mdl = GCN(in_channels=1, hidden_channels=64, num_layers=5, out_channels=n_tasks)
gcn_results = SimpleGNNTrainer(metrics, g, device="cuda", metric_best="auroc").train(mdl, y, masks)
```

## Contributing

### Additional packages used for dev

* [tox](https://tox.wiki/en/latest/index.html)
* [pytest](https://docs.pytest.org/en/6.2.x/)
* [pytest-cov](https://pypi.org/project/pytest-cov/)
* [pytest-subtest](https://pypi.org/project/pytest-subtests/)
* [pre-commit](https://github.com/pre-commit/pre-commit)

See ``requirements-dev.txt``. Run the following to install all dev dependencies

```bash
$ pip install -r requirements-dev.txt
```

### Testing

Simply run ``pytest`` to run all tests

```bash
$ pytest
```

Alternatively, can also show the coverage report
```bash
$ pytest --cov src/NLEval
```

Run type checks and coding style checks using mypy and flake8 via tox:
```bash
$ tox -e mypy,flake8
```
