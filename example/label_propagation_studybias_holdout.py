import os.path as osp

import numpy as np
from NLEval import model
from NLEval.graph import DenseGraph
from NLEval.label import filters
from NLEval.label import LabelsetCollection
from NLEval.label.split import RatioHoldout
from NLEval.model.label_propagation import OneHopPropagation
from NLEval.model_trainer.label_propagation import LabelPropagationTrainer
from sklearn.metrics import roc_auc_score as auroc

NETWORK = "STRING"
LABEL = "KEGGBP"
DATA_DIR = osp.join(osp.pardir, "data")
GRAPH_FP = osp.join(DATA_DIR, "networks", f"{NETWORK}.edg")
LABEL_FP = osp.join(DATA_DIR, "labels", f"{LABEL}.gmt")
PROPERTY_FP = osp.join(DATA_DIR, "properties", "pubcnt.txt")

print(f"{NETWORK=}\n{LABEL=}")

# Load data
g = DenseGraph.from_edglst(GRAPH_FP, weighted=True, directed=False)
lsc = LabelsetCollection.from_gmt(LABEL_FP)

# Filter labels
print(f"Number of labelsets before filtering: {len(lsc.label_ids)}")
lsc.apply(filters.EntityExistenceFilter(g.idmap.lst), inplace=True)
lsc.apply(filters.LabelsetRangeFilterSize(min_val=50), inplace=True)
print(f"Number of labelsets after filtering: {len(lsc.label_ids)}")

# Load gene properties for study-bias holdout
# Note: wait after filtering is done to reduce time for filtering
lsc.load_entity_properties(PROPERTY_FP, "PubMed Count", 0, int)

# 3/2 train/test split using genes with higher PubMed Count for training
splitter = RatioHoldout(0.6, 0.4, ascending=False)

# Select model
mdl = OneHopPropagation()

# Setup trainer, use auroc as the evaluation metric
metrics = {"auroc": auroc}
trainer = LabelPropagationTrainer(metrics, g)

scores = []
for label_id in lsc.label_ids:
    y, masks, _ = lsc.split(
        splitter,
        target_ids=g.idmap.lst,
        labelset_name=label_id,
        property_name="PubMed Count",
    )
    results = trainer.train(mdl, y, masks)
    scores.append(results["test_auroc"])
    train_score, test_score = results["train_auroc"], results["test_auroc"]
    print(f"Train: {train_score:.4f}\tTest: {test_score:.4f}\t{label_id}")

print(f"Average test score = {np.mean(scores):.4f}, std = {np.std(scores):.4f}")

print(
    """
Expected outcome
--------------------------------------------------------------------------------
NETWORK='STRING'
LABEL='KEGGBP'
Number of labelsets before filtering: 139
Number of labelsets after filtering: 58
Average test score = 0.9530, std = 0.0406
--------------------------------------------------------------------------------
""",
)
