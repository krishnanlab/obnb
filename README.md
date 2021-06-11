# NetworkLearningEval

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
