import os.path as osp

import numpy as np
from NLEval import model
from NLEval.graph.SparseGraph import SparseGraph
from NLEval.label import labelset_collection
from NLEval.label import labelset_filter
from NLEval.valsplit.Holdout import TrainValTest
from sklearn.metrics import roc_auc_score as auroc

NETWORK = "STRING-EXP"
DATASET = "KEGGBP"
DATA_DIR = osp.join(osp.pardir, "data")
GRAPH_FP = osp.join(DATA_DIR, "networks", f"{NETWORK}.edg")
LABEL_FP = osp.join(DATA_DIR, "labels", f"{DATASET}.gmt")
PROPERTY_FP = osp.join(DATA_DIR, "properties", "pubcnt.txt")

train_ratio = 0.6
test_ratio = 0.2
min_pos = 10

print(f"Run test using network = {NETWORK!r} and dataset = {DATASET!r}")

g = SparseGraph.from_edglst(GRAPH_FP, True, False)
lsc = labelset_collection.SplitLSC.from_gmt(LABEL_FP)

print(f"Number of labelsets in original file: {len(lsc.label_ids)}")

lsc.apply(labelset_filter.EntityExistanceFilter(g.idmap.lst), inplace=True)
lsc.apply(labelset_filter.LabelsetRangeFilterSize(min_val=50), inplace=True)
lsc.load_entity_properties(PROPERTY_FP, "Pubmed Count", 0, int)
lsc.valsplit = TrainValTest(train_ratio=train_ratio, test_ratio=test_ratio)

print(f"Number of labelsets before filtering: {len(lsc.label_ids)}")
lsc.train_test_setup(g, prop_name="Pubmed Count", min_pos=min_pos)

print(
    f"Number of labelsets after filtering "
    f"(train={train_ratio}, test={test_ratio}, "
    f"min_pos={min_pos}): {len(lsc.label_ids)}\n"
    f"Number of training = {lsc.valsplit.train_index.size}, "
    f"validation = {lsc.valsplit.valid_index.size}, "
    f"testing = {lsc.valsplit.test_index.size}",
)

print(
    """
Expected outcome
--------------------------------------------------------------------------------
Run test using network = 'STRING-EXP' and dataset = 'KEGGBP'
Number of labelsets in original file: 139
Number of labelsets before filtering: 54
Number of labelsets after filtering (train=0.6, test=0.2, min_pos=10): 21
Number of training = 1005, validation = 336, testing = 335
--------------------------------------------------------------------------------
""",
)
