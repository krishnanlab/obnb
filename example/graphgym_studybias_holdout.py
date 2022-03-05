import numpy as np
import torch
from load_data import load_data
from NLEval.label.filters import LabelsetRangeFilterSplit
from NLEval.label.split import RatioPartition
from NLEval.metric import auroc
from NLEval.model_trainer.graphgym import graphgym_model_wrapper
from NLEval.model_trainer.graphgym import GraphGymTrainer

# Load dataset (with sparse graph)
g, lsc = load_data("STRING-EXP", "KEGGBP", sparse=True, filter_negative=False)

# 3/2 train/test split using genes with higher PubMed Count for training
splitter = RatioPartition(0.6, 0.2, 0.2, ascending=False)
lsc.iapply(
    LabelsetRangeFilterSplit(20, splitter, True, property_name="PubMed Count"),
)
n_tasks = len(lsc.label_ids)
print(f"{n_tasks=}\n")

# Set up trainer first, which then is used to construct model from the config
# file; use auroc as the evaluation metric
metrics = {"auroc": auroc}
trainer = GraphGymTrainer(
    metrics,
    g,
    device="auto",
    metric_best="auroc",
    cfg_file="example_config.yaml",
    cfg_opts={
        "optim.max_epoch": 100,
        "gnn.layers_pre_mp": 0,
        "gnn.layers_mp": 3,
        "gnn.dim_inner": 32,
        "train.eval_period": 10,
        "train.skip_train_eval": True,
    },
)

mdl = trainer.create_model(dim_in=1, dim_out=n_tasks)

y, masks = lsc.split(
    splitter,
    target_ids=g.node_ids,
    property_name="PubMed Count",
)

results = trainer.train(mdl, y, masks)
print(f"\nBest results:\n{results}\n")

# Check to see if the model is rewinded back to the best model correctly
y_pred, y_true = graphgym_model_wrapper(mdl)(trainer.data, y)
for split in "train", "val", "test":
    mask = masks[split][:, 0]
    print(f"{split:>5}: {auroc(y_true[mask], y_pred[mask])}")
