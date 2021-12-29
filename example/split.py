from sys import path

path.append("../")
from NLEval import graph, valsplit, label, model
from sklearn.metrics import roc_auc_score as auroc
import numpy as np

data_path = "../../data/"
network = "STRING"
dataset = "KEGGBP"
network_fp = data_path + f"networks/{network}.edg"
label_fp = data_path + f"labels/{dataset}.gmt"

train_ratio = 0.6
test_ratio = 0.2
min_pos = 10

print(f"Run test using network = {repr(network)} and dataset = {repr(dataset)}")

g = graph.SparseGraph.SparseGraph.from_edglst(network_fp, True, False)
lsc = label.labelset_collection.SplitLSC.from_gmt(label_fp)

print(f"Number of labelsets in original file: {len(lsc.label_ids)}")

lsc.apply(
    label.labelset_filter.EntityExistanceFilter(g.idmap.lst),
    inplace=True,
)
lsc.apply(
    label.labelset_filter.LabelsetRangeFilterSize(min_val=50),
    inplace=True,
)
lsc.load_entity_properties(
    data_path + "/properties/pubcnt.txt",
    "Pubmed Count",
    0,
    int,
)
lsc.valsplit = valsplit.Holdout.TrainValTest(
    train_ratio=train_ratio,
    test_ratio=test_ratio,
)

print(f"Number of labelsets before filtering: {len(lsc.label_ids)}")
lsc.train_test_setup(g, prop_name="Pubmed Count", min_pos=min_pos)

print(
    f"Number of labelsets after filtering "
    f"(train_ratio={train_ratio}, test_ratio={test_ratio}, "
    f"min_pos={min_pos}): {len(lsc.label_ids)}\n"
    f"Number of training = {lsc.valsplit.train_index.size}, "
    f"validation = {lsc.valsplit.valid_index.size}, "
    f"testing = {lsc.valsplit.test_index.size}",
)

"""
Run test using network = 'STRING' and dataset = 'KEGGBP'
Number of labelsets in original file: 139
Number of labelsets before filtering: 58
Number of labelsets after filtering (train_ratio=0.6, test_ratio=0.2, min_pos=10): 27
Number of training = 1273, validation = 426, testing = 424
"""
