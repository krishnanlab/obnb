[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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
positive_set = lsc.getLabelset(some_labelID)
negative_set = lsc.getNegative(some_labelID)
SLA_score_dict = SL_A.predict(positive_set, negative_set)
LPA_score_dict = LP_A.predict(positive_set, negative_set)
```

## Contributing

### Coding style

Use [``black``](https://github.com/psf/black) to format code, also enforce max 80 line length.

```bash
$ black src/ -l 80
```

The ``-l`` option specifies the maixmum line-length, which is set to be 88 originally.
