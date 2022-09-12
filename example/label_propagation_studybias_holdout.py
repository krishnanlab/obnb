from sklearn.metrics import roc_auc_score as auroc
from utils import load_data, print_expected

from nleval import Dataset
from nleval.label.split import RatioPartition
from nleval.model.label_propagation import OneHopPropagation
from nleval.model_trainer import LabelPropagationTrainer

# Load dataset
g, lsc, converter = load_data()

# 3/2 train/test split using genes with higher PubMed Count for training
splitter = RatioPartition(0.6, 0.4, ascending=False, property_converter=converter)

# Select model
mdl = OneHopPropagation()

# Setup trainer, use auroc as the evaluation metric
metrics = {"auroc": auroc}
trainer = LabelPropagationTrainer(metrics, log_level="INFO")

# Evaluate the model for a single task
dataset = Dataset(
    graph=g,
    label=lsc,
    splitter=splitter,
    labelset_name=lsc.label_ids[0],
    consider_negative=True,
)
print(trainer.train(mdl, dataset))

# Evaluate the model for all tasks
dataset = Dataset(graph=g, label=lsc, splitter=splitter)
results = trainer.eval_multi_ovr(mdl, dataset, consider_negative=True)
print(f"Average train score = {results['train_auroc']:.4f}")
print(f"Average test score = {results['test_auroc']:.4f}")

print_expected("Average test score = 0.6506", "Average test score = 0.6142")
