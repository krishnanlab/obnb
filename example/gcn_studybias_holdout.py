import torch
from obnb import Dataset
from obnb.label.split import RatioPartition
from obnb.metric import auroc
from obnb.model_trainer.gnn import SimpleGNNTrainer
from torch_geometric.nn import GCN
from utils import load_data

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
    log_path="test_log/gcn/run.log",
)

dataset = Dataset(graph=g, label=lsc, splitter=splitter)
results = trainer.train(mdl, dataset)
print(f"\nBest results:\n{results}\n")

# Check to see if the model is rewinded back to the best model correctly
data = dataset.to_pyg_data(device=device)
y_pred = mdl(data.x, data.edge_index).detach().cpu().numpy()
for mask_name in "train", "val", "test":
    mask = dataset.masks[mask_name][:, 0]
    print(f"{mask_name:>5}: {auroc(dataset.y[mask], y_pred[mask])}")
