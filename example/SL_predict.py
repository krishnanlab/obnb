import os.path as osp

from NLEval.graph import DenseGraph
from NLEval.label import filters
from NLEval.label.collection import LabelsetCollection
from NLEval.label.split import AllHoldout
from NLEval.model_trainer import SupervisedLearningTrainer
from sklearn.linear_model import LogisticRegression
from sklearn.metric import roc_auc_score as auroc

DATA_DIR = osp.join(osp.pardir, "data")
GRAPH_FP = osp.join(DATA_DIR, "networks", "STRING-EXP.edg")
LABEL_FP = osp.join(DATA_DIR, "labels", "KEGGBP.gmt")

i = 24  # index of labelset
k = 50  # numbers of top genes to display

# load graph and labelset collection
g = DenseGraph.from_edglst(GRAPH_FP, weighted=True, directed=False)
lsc = LabelsetCollection.from_gmt(LABEL_FP)
lsc.iapply(filters.EntityExistenceFilter(g.idmap.lst))
lsc.iapply(filters.LabelsetRangeFilterSize(min_val=50))

# initialize model
mdl = LogisticRegression(penalty="l2", solver="liblinear")

# diplay choice of labelsets
for j, m in enumerate(lsc.label_ids):
    print(f"Index: {j:>4d}, Labelset size: {len(lsc.get_labelset(m)):>4d}, {m}")
print("")

# get label_id
label_id = lsc.label_ids[i]
print(label_id)

# train and get genome wide prediction
y, masks = lsc.split(
    AllHoldout(),
    target_ids=g.node_ids,
    labelset_name=label_id,
    mask_names=("train",),
)

metrics = {"auroc": auroc}
trainer = SupervisedLearningTrainer(metrics, g)
trainer.train(mdl, y, masks)
score_dict = {i: j for i, j in zip(g.node_ids, mdl.decision_function(g.mat))}

# print top ranked genes and its intersection with known ones
top_list = sorted(score_dict, key=score_dict.get, reverse=True)[:k]
intersection = sorted(set(top_list) & lsc.get_labelset(label_id))

print(f"Top {k} genes: {repr(top_list)}")
print(f"Known genes in top {k}: {repr(intersection)}")
