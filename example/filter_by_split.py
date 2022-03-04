from time import perf_counter

import numpy as np
from load_data import load_data
from NLEval.label.filters import LabelsetRangeFilterSplit
from NLEval.label.split import RatioPartition
from NLEval.model.label_propagation import OneHopPropagation
from NLEval.model_trainer import LabelPropagationTrainer
from sklearn.metrics import roc_auc_score as auroc


# Load daatset
g, lsc = load_data("STRING-EXP", "KEGGBP")

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
    LabelsetRangeFilterSplit(
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
