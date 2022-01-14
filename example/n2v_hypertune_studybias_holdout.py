import os.path as osp

import numpy as np
from NLEval.graph import DenseGraph
from NLEval.graph import MultiFeatureVec
from NLEval.label import filters
from NLEval.label import LabelsetCollection
from NLEval.label.split import RatioPartition
from NLEval.model_trainer import MultiSupervisedLearningTrainer
from numba import set_num_threads
from pecanpy.pecanpy import SparseOTF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auroc

NETWORK = "STRING-EXP"
LABEL = "KEGGBP"
DATA_DIR = osp.join(osp.pardir, "data")
GRAPH_FP = osp.join(DATA_DIR, "networks", f"{NETWORK}.edg")
LABEL_FP = osp.join(DATA_DIR, "labels", f"{LABEL}.gmt")
PROPERTY_FP = osp.join(DATA_DIR, "properties", "PubMedCount.txt")

print(f"{NETWORK=}\n{LABEL=}")

# Load data
g = DenseGraph.from_edglst(GRAPH_FP, weighted=True, directed=False)
lsc = LabelsetCollection.from_gmt(LABEL_FP)

# Filter labels
print(f"Number of labelsets before filtering: {len(lsc.label_ids)}")
lsc.iapply(filters.EntityExistenceFilter(g.idmap.lst))
lsc.iapply(filters.LabelsetRangeFilterSize(min_val=50))
lsc.iapply(filters.NegativeGeneratorHypergeom(p_thresh=0.05))
print(f"Number of labelsets after filtering: {len(lsc.label_ids)}")

# Load gene properties for study-bias holdout
# Note: wait after filtering is done to reduce time for filtering
lsc.load_entity_properties(PROPERTY_FP, "PubMed Count", 0, int)

# 3/2 train/test split using genes with higher PubMed Count for training
splitter = RatioPartition(0.6, 0.2, 0.2, ascending=False)

# Select model
mdl = LogisticRegression(penalty="l2", solver="lbfgs", n_jobs=1)

set_num_threads(32)
mats = []
qs = [0.01, 0.1, 1, 10, 100]
for q in qs:
    # TODO: Fix and use PreComp here
    g_n2v = SparseOTF.from_mat(
        g.mat,
        g.idmap.lst,
        p=1,
        q=q,
        workers=32,
        verbose=True,
    )
    mats.append(g_n2v.embed())
mfvec = MultiFeatureVec.from_mats(mats, g.idmap, [f"{q=}" for q in qs])

# Setup trainer, use auroc as the evaluation metric
metrics = {"auroc": auroc}
trainer = MultiSupervisedLearningTrainer(metrics, mfvec, log=True)

scores = []
for label_id in lsc.label_ids:
    y, masks = lsc.split(
        splitter,
        target_ids=g.node_ids,
        labelset_name=label_id,
        property_name="PubMed Count",
        consider_negative=True,
    )
    results = trainer.train(mdl, y, masks)
    scores.append(results["test_auroc"])
    train_score, test_score = results["train_auroc"], results["test_auroc"]
    print(f"Train: {train_score:.4f}\tTest: {test_score:.4f}\t{label_id}")

print(f"Average test score = {np.mean(scores):.4f}, std = {np.std(scores):.4f}")
