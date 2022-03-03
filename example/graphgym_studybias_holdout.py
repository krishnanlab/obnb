import numpy as np
import torch
from load_data import load_data
from NLEval.label.split import RatioPartition
from NLEval.model_trainer.graphgym import GraphGymTrainer
from sklearn.metrics import roc_auc_score as auroc

# Load dataset (with sparse graph)
g, lsc = load_data("STRING-EXP", "KEGGBP", sparse=True, filter_negative=False)
n_tasks = len(lsc.label_ids)

# 3/2 train/test split using genes with higher PubMed Count for training
splitter = RatioPartition(0.6, 0.2, 0.2, ascending=False)

# Set up trainer first, which then is used to construct model from the config
# file; use auroc as the evaluation metric
# metrics = {"auroc": auroc}
metrics = {
    "trivial": lambda a, b, c: 1,
}  # TODO: create graphgym custom auroc func
trainer = GraphGymTrainer(
    metrics,
    g,
    device="auto",
    # metric_best="auroc",
    metric_best="trivial",
    cfg_file="example_config.yaml",
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
