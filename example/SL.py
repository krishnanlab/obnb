import os.path as osp

import numpy as np
from NLEval import model
from NLEval import valsplit
from NLEval.graph import DenseGraph
from NLEval.label import filters
from NLEval.label.collection import SplitLSC
from sklearn.metrics import roc_auc_score as auroc

DATA_DIR = osp.join(osp.pardir, "data")
GRAPH_FP = osp.join(DATA_DIR, "networks", "STRING-EXP.edg")
LABEL_FP = osp.join(DATA_DIR, "labels", "KEGGBP.gmt")
PROPERTY_FP = osp.join(DATA_DIR, "properties", "PubMedCount.txt")

g = DenseGraph.from_edglst(GRAPH_FP, weighted=True, directed=False)
lsc = SplitLSC.from_gmt(LABEL_FP)

lsc.iapply(filters.EntityExistenceFilter(g.idmap.lst))
lsc.iapply(filters.LabelsetRangeFilterSize(min_val=50))

lsc.load_entity_properties(PROPERTY_FP, "Pubmed Count", 0, int)
# lsc.valsplit = valsplit.Holdout.BinHold(3, shuffle=True)
lsc.valsplit = valsplit.Holdout.TrainValTest(train_ratio=0.33, test_ratio=0.33)

print(f"Number of labelsets before filtering: {len(lsc.label_ids)}")
lsc.train_test_setup(g, prop_name="Pubmed Count", min_pos=10)
print(f"Number of labelsets after filtering: {len(lsc.label_ids)}")

mdl = model.SupervisedLearning.LogReg(g, penalty="l2", solver="lbfgs")

scores = []
for label_id in lsc.label_ids:
    score = auroc(*(mdl.test(lsc.split_labelset(label_id))))
    scores.append(score)
    print(f"{score:.4f}\t{label_id}")
print(f"Average score = {np.mean(scores):.4f}, std = {np.std(scores):.4f}")

"""
# Print the average properties in train/test sets
for label_id in lsc.label_ids:
    tr, _, ts, _ = next(lsc.split_labelset(label_id))
    train_props = [lsc.entity.get_property(i, 'Pubmed Count') for i in tr]
    test_props = [lsc.entity.get_property(i, 'Pubmed Count') for i in ts]
    print(
        f"Avg train prop: {np.mean(train_props):.6f}\t"
        f"Avg test prop: {np.mean(test_props):.6f}",
    )
"""

print(
    """
Expected outcome
--------------------------------------------------------------------------------
Average score = 0.7527, std = 0.1128
--------------------------------------------------------------------------------
""",
)
