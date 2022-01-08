import os.path as osp
from time import perf_counter

import numpy as np
from NLEval import model
from NLEval.graph import DenseGraph
from NLEval.label import filters
from NLEval.label import LabelsetCollection
from NLEval.label.split import RatioPartition
from NLEval.model.label_propagation import OneHopPropagation
from NLEval.model_trainer import LabelPropagationTrainer
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
splitter = RatioPartition(0.6, 0.4, ascending=False)

# Select model
mdl = OneHopPropagation()

# Setup trainer, use auroc as the evaluation metric
metrics = {"auroc": auroc}
trainer = LabelPropagationTrainer(metrics, g)


def print_split_stats(lsc, name):
    y, masks = lsc.split(
        splitter,
        target_ids=g.node_ids,
        property_name="PubMed Count",
    )
    print(f"\nNumber of labelsets {name} split-filtering: {len(lsc.label_ids)}")
    for name, mask in masks.items():
        num_pos = y[mask[:, 0]].sum(0)
        print(
            f"{name}:\n\tMinimum number of positives = {min(num_pos)}\n"
            f"\tAverage number of positives = {np.mean(num_pos):.2f}",
        )


# Check minimum number of positives in each split before filtering
print_split_stats(lsc, "before")

# Apply split-filter
print(f"{'START FILTERING BY SPLITS':-^80}")
elapsed = perf_counter()
lsc.iapply(
    filters.LabelsetRangeFilterSplit(
        10,  # required minimum number of positives in each split
        splitter,
        property_name="PubMed Count",
        verbose=True,
    ),
)
elapsed = perf_counter() - elapsed
endstr = f"DONE ({elapsed=:.2f})"
print(f"{endstr:-^80}\n")

# Check minimum number of positives in each split after filtering
print_split_stats(lsc, "after")
