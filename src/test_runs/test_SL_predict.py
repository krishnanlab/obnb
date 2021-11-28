from sys import path

path.append("../")
from NLEval import graph, valsplit, label, model
from sklearn.metrics import roc_auc_score as auroc
import numpy as np

# define connstatns
data_path = "../../data/"  # path to data
i = 54  # index of labelset
k = 50  # numbers of top genes to display

# load graph and labelset collection
g = graph.DenseGraph.DenseGraph.from_edglst(
    data_path + "networks/STRING-EXP.edg", weighted=True, directed=False
)
lsc = label.LabelsetCollection.SplitLSC.from_gmt(
    data_path + "labels/KEGGBP.gmt"
)

# initialize model
mdl = model.SupervisedLearning.LogReg(g, penalty="l2", solver="lbfgs")

# diplay choice of labelsets
for l, m in enumerate(lsc.label_ids):
    # index, labelset size, labelset ID
    print(f"{l:>4d} {len(lsc.getLabelset(m)):>4d} {m}")
print("")

# get labelID
labelID = lsc.label_ids[i]
print(labelID)

# get positive and negative samples
pos = lsc.getLabelset(labelID)
neg = lsc.getNegative(labelID)

# train and get genome wide prediction
score_dict = mdl.predict(pos, neg)

# print top ranked genes and its intersection with known ones
top_list = sorted(score_dict, key=score_dict.get, reverse=True)[:k]
intersection = list(set(top_list) & pos)

print(f"Top {k} genes: {repr(top_list)}")
print(f"Known genes in top {k}: {repr(intersection)}")
