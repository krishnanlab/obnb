import torch
from sklearn.metrics import roc_auc_score as auroc
from torch_geometric.nn import GCN
from utils import load_data

from NLEval import Dataset
from NLEval.label.split import RatioPartition
from NLEval.model_trainer.gnn import SimpleGNNTrainer

# Load dataset (with sparse graph)
g, lsc, converter = load_data(
    sparse=True,
    filter_negative=False,
    filter_holdout_split=True,
)
splitter = RatioPartition(0.6, 0.2, 0.2, ascending=False, property_converter=converter)
n_tasks = len(lsc.label_ids)
print(f"{n_tasks=}\n")

# Select model
mdl = GCN(in_channels=1, hidden_channels=64, num_layers=5, out_channels=n_tasks)

# Setup trainer, use auroc as the evaluation metric
metrics = {"auroc": auroc}
device = "cuda" if torch.cuda.is_available() else "cpu"
trainer = SimpleGNNTrainer(
    metrics,
    device=device,
    metric_best="auroc",
    epochs=100,
    lr=0.1,
)

y, masks = lsc.split(
    splitter,
    target_ids=g.node_ids,
    property_name="PubMed Count",
)
dataset = Dataset(graph=g, y=y, masks=masks)

results = trainer.train(mdl, dataset)
print(f"\nBest results:\n{results}\n")

# Check to see if the model is rewinded back to the best model correctly
data = dataset.to_pyg_data(device=device)
args = (data.x, data.edge_index, data.edge_weight)
y_pred = mdl(*args).detach().cpu().numpy()
for mask_name in "train", "val", "test":
    mask = masks[mask_name][:, 0]
    print(f"{mask_name:>5}: {auroc(y[mask], y_pred[mask])}")
