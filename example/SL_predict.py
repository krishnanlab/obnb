import os.path as osp

import numpy as np
from NLEval import model
from NLEval import valsplit
from NLEval.graph import DenseGraph
from NLEval.label import labelset_collection
from sklearn.metrics import roc_auc_score as auroc

DATA_DIR = osp.join(osp.pardir, "data")
GRAPH_FP = osp.join(DATA_DIR, "networks", "STRING-EXP.edg")
LABEL_FP = osp.join(DATA_DIR, "labels", "KEGGBP.gmt")

i = 54  # index of labelset
k = 50  # numbers of top genes to display

# load graph and labelset collection
g = DenseGraph.from_edglst(GRAPH_FP, weighted=True, directed=False)
lsc = labelset_collection.SplitLSC.from_gmt(LABEL_FP)

# initialize model
mdl = model.SupervisedLearning.LogReg(g, penalty="l2", solver="liblinear")

# diplay choice of labelsets
for l, m in enumerate(lsc.label_ids):
    # index, labelset size, labelset ID
    print(f"{l:>4d} {len(lsc.get_labelset(m)):>4d} {m}")
print("")

# get label_id
label_id = lsc.label_ids[i]
print(label_id)

# get positive and negative samples
pos = lsc.get_labelset(label_id)
neg = lsc.get_negative(label_id)

# train and get genome wide prediction
score_dict = mdl.predict(pos, neg)

# print top ranked genes and its intersection with known ones
top_list = sorted(score_dict, key=score_dict.get, reverse=True)[:k]
intersection = list(set(top_list) & pos)

print(f"Top {k} genes: {repr(top_list)}")
print(f"Known genes in top {k}: {repr(intersection)}")
print(f"\nWARNING: This script fails to reproduce consistent results.")
