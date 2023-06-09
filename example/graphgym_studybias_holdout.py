import torch
from utils import load_data

from obnb import Dataset
from obnb.label.split import RatioPartition
from obnb.metric import auroc
from obnb.model_trainer.graphgym import GraphGymTrainer, graphgym_model_wrapper

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset (with sparse graph)
g, lsc, converter = load_data(
    sparse=True,
    filter_negative=False,
    filter_holdout_split=True,
)
# 3/2 train/test split using genes with higher PubMed Count for training
splitter = RatioPartition(0.6, 0.2, 0.2, ascending=False, property_converter=converter)

dataset = Dataset(graph=g, label=lsc, splitter=splitter)
data = dataset.to_pyg_data(device=device)

feat_dim = data.x.shape[1]
n_tasks = len(lsc.label_ids)
print(f"{feat_dim=}\n{n_tasks=}\n")

# Set up trainer first, which then is used to construct model from the config
# file; use auroc as the evaluation metric
metrics = {"auroc": auroc}
trainer = GraphGymTrainer(
    metrics=metrics,
    device=device,
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
mdl = trainer.create_model(dim_in=32, dim_out=n_tasks)

results = trainer.train(mdl, dataset)
print(f"\nBest results:\n{results}\n")

# Check to see if the model is rewinded back to the best model correctly
y_pred, y_true = graphgym_model_wrapper(mdl)(data)
for split in "train", "val", "test":
    mask = dataset.masks[split][:, 0]
    print(f"{split:>5}: {auroc(y_true[mask], y_pred[mask])}")
