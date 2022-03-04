import numpy as np
import torch
from load_data import load_data
from NLEval.label.filters import LabelsetRangeFilterSplit
from NLEval.label.split import RatioPartition
from NLEval.metric.graphgym_metric import graphgym_auroc
from NLEval.model_trainer.graphgym import GraphGymTrainer

# Load dataset (with sparse graph)
g, lsc = load_data("BioGRID", "KEGGBP", sparse=True, filter_negative=False)

# 3/2 train/test split using genes with higher PubMed Count for training
splitter = RatioPartition(0.6, 0.2, 0.2, ascending=False)
lsc.iapply(
    LabelsetRangeFilterSplit(10, splitter, True, property_name="PubMed Count"),
)
n_tasks = len(lsc.label_ids)

# Set up trainer first, which then is used to construct model from the config
# file; use auroc as the evaluation metric
metrics = {"auroc": graphgym_auroc}
trainer = GraphGymTrainer(
    metrics,
    g,
    device="auto",
    metric_best="auroc",
    cfg_file="example_config.yaml",
    cfg_opts=[
        "optim.max_epoch",
        200,
        "gnn.layers_pre_mp",
        0,
    ],
)

mdl = trainer.create_model(dim_in=1, dim_out=n_tasks)

y, masks = lsc.split(
    splitter,
    target_ids=g.node_ids,
    property_name="PubMed Count",
)

trainer.train(mdl, y, masks)

# results = trainer.train(mdl, y, masks)
# print(f"\nBest results:\n{results}\n")
#
## Check to see if the model is rewinded back to the best model correctly
# args = (trainer.data.x, trainer.data.edge_index, trainer.data.edge_weight)
# y_pred = mdl(*args).detach().cpu().numpy()
# for mask_name in "train", "val", "test":
#    mask = masks[mask_name][:, 0]
#    print(f"{mask_name:>5}: {auroc(y[mask], y_pred[mask])}")
