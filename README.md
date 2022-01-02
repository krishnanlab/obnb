![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Tests](https://github.com/krishnanlab/NetworkLearningEval/actions/workflows/tests.yml/badge.svg)

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

## Optional Pytorch Geometric installation

One can install [Pytorch Geomtric](https://github.com/pyg-team/pytorch_geometric) to enable some GNN related features.
To install PyG, first need to install [PyTorch](https://pytorch.org).
For full details about installation instructions, visit the links above.
Assuming the system has Python3.8 or above installed, with CUDA10.2, use the following to install both PyTorch and PyG.

```bash
conda install pytorch=1.9 torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install pyg -c pyg -c conda-forge
```

## Full dev installation

With Python3.8 or above and CUDA10.2:

```bash
pip install -e .
pip install -r requirements-dev.txt
conda install pytorch=1.9 torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install pyg -c pyg -c conda-forge
```

## Quick Demonstration
```python
from NLEval import graph, valsplit, label, model

# specify data paths
network_fp = '/path/to/network.edg' # edg for edgelist representation of sparse network
labelset_fp = '/path/to/label.gmt' # label in the format of Gene Matrix Transpose

# load data (network and labelset collection)
g = graph.DenseGraph.DenseGraph.from_edgelst(network_fp, weighted=True, directed=False)
lsc = label.LabelsetCollection.SplitLSC.from_gmt(labelset_fp)

# initialize models (note that specific network is required to initialize model)
SL_A = model.SupervisedLearning.LogReg(g, penalty='l2', solver='lbfgs')
LP_A = model.LabelPropagation.LP(g)

# train model and get genomewide prediction scores
positive_set = lsc.get_labelset(some_label_id)
negative_set = lsc.get_negative(some_label_id)
SLA_score_dict = SL_A.predict(positive_set, negative_set)
LPA_score_dict = LP_A.predict(positive_set, negative_set)
```

## Contributing

### Additional packages used for dev

* [tox](https://tox.wiki/en/latest/index.html)
* [pytest](https://docs.pytest.org/en/6.2.x/)
* [pytest-cov](https://pypi.org/project/pytest-cov/)
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
