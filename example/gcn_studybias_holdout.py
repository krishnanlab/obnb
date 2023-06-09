import torch
from torch_geometric.nn import GCN
from utils import load_data

from obnb import Dataset
from obnb.label.split import RatioPartition
from obnb.metric import auroc
from obnb.model_trainer.gnn import SimpleGNNTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset (with sparse graph)
g, lsc, converter = load_data(
    sparse=True,
    filter_negative=False,
    filter_holdout_split=True,
)
splitter = RatioPartition(0.6, 0.2, 0.2, ascending=False, property_converter=converter)
dataset = Dataset(graph=g, label=lsc, splitter=splitter)
data = dataset.to_pyg_data(device=device)

feat_dim = data.x.shape[1]
n_tasks = len(lsc.label_ids)
print(f"{feat_dim=}\n{n_tasks=}\n")

# Setup model and trainer, use auroc as the evaluation metric
metrics = {"auroc": auroc}
trainer = SimpleGNNTrainer(
    metrics=metrics,
    device=device,
    metric_best="auroc",
    epochs=100,
    lr=0.1,
    log_path="test_log/gcn/run.log",
)
mdl = GCN(in_channels=feat_dim, hidden_channels=64, num_layers=5, out_channels=n_tasks)

results = trainer.train(mdl, dataset)
print(f"\nBest results:\n{results}\n")

# Check to see if the model is rewinded back to the best model correctly
y_pred = mdl(data.x, data.edge_index).detach().cpu().numpy()
for mask_name in "train", "val", "test":
    mask = dataset.masks[mask_name][:, 0]
    print(f"{mask_name:>5}: {auroc(dataset.y[mask], y_pred[mask])}")
