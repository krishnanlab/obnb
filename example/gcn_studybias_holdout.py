import os.path as osp

import numpy as np
import torch
from NLEval.graph import SparseGraph
from NLEval.label import filters
from NLEval.label import LabelsetCollection
from NLEval.label.split import RatioPartition
from NLEval.model_trainer.gnn import SimpleGNNTrainer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auroc
from torch_geometric.nn import GCN

NETWORK = "STRING-EXP"
LABEL = "KEGGBP"
DATA_DIR = osp.join(osp.pardir, "data")
GRAPH_FP = osp.join(DATA_DIR, "networks", f"{NETWORK}.edg")
LABEL_FP = osp.join(DATA_DIR, "labels", f"{LABEL}.gmt")
PROPERTY_FP = osp.join(DATA_DIR, "properties", "PubMedCount.txt")

print(f"{NETWORK=}\n{LABEL=}")

# Load data
g = SparseGraph.from_edglst(GRAPH_FP, weighted=True, directed=False)
lsc = LabelsetCollection.from_gmt(LABEL_FP)

# Filter labels
print(f"Number of labelsets before filtering: {len(lsc.label_ids)}")
lsc.iapply(filters.EntityExistenceFilter(g.idmap.lst))
lsc.iapply(filters.LabelsetRangeFilterSize(min_val=50))
print(f"Number of labelsets after filtering: {len(lsc.label_ids)}")
n_tasks = len(lsc.label_ids)

# Load gene properties for study-bias holdout
# Note: wait after filtering is done to reduce time for filtering
lsc.load_entity_properties(PROPERTY_FP, "PubMed Count", 0, int)

# 3/2 train/test split using genes with higher PubMed Count for training
splitter = RatioPartition(0.6, 0.2, 0.2, ascending=False)

# Select model
mdl = GCN(in_channels=1, hidden_channels=64, num_layers=5, out_channels=n_tasks)

# Setup trainer, use auroc as the evaluation metric
metrics = {"auroc": auroc}
device = "cuda" if torch.cuda.is_available() else "cpu"
trainer = SimpleGNNTrainer(
    metrics,
    g,
    device=device,
    metric_best="auroc",
    log=True,
)

y, masks = lsc.split(
    splitter,
    target_ids=g.idmap.lst,
    property_name="PubMed Count",
)

results = trainer.train(mdl, y, masks, epochs=200, lr=0.1)
print(f"\nBest results:\n{results}\n")

# Check to see if the model is rewinded back to the best model correctly
args = (trainer.data.x, trainer.data.edge_index, trainer.data.edge_weight)
y_pred = mdl(*args).detach().cpu().numpy()
for mask_name in "train", "val", "test":
    mask = masks[mask_name][:, 0]
    print(f"{mask_name:>5}: {auroc(y[mask], y_pred[mask])}")
