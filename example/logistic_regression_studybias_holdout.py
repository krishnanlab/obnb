from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auroc
from utils import load_data, print_expected

from nleval import Dataset
from nleval.label.split import RatioPartition
from nleval.model_trainer import SupervisedLearningTrainer

# Load dataset
g, lsc, converter = load_data()
feature = g.to_feature()

# 3/2 train/test split using genes with higher PubMed Count for training
splitter = RatioPartition(0.6, 0.4, ascending=False, property_converter=converter)

# Select model
mdl = LogisticRegression(penalty="l2", solver="lbfgs", n_jobs=1, max_iter=500)

# Setup trainer, use auroc as the evaluation metric
metrics = {"auroc": auroc}
trainer = SupervisedLearningTrainer(metrics, log_level="INFO")

y, masks = lsc.split(splitter, target_ids=g.node_ids)
dataset = Dataset(feature=feature, label=lsc, splitter=splitter)
results = trainer.eval_multi_ovr(mdl, dataset, consider_negative=True)
print(f"Average train score = {results['train_auroc']:.4f}")
print(f"Average test score = {results['test_auroc']:.4f}")

print_expected(
    "Average train score = 0.9971",
    "Average test score = 0.6986",
)
