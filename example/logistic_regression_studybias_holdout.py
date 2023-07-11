from sklearn.linear_model import LogisticRegression
from utils import load_data, print_expected

from obnb import Dataset
from obnb.label.split import RatioPartition
from obnb.metric import auroc
from obnb.model_trainer import SupervisedLearningTrainer

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

# Train a single model
# FIX: fix consider_negative
dataset = Dataset(
    feature=feature,
    label=lsc,
    splitter=splitter,
    labelset_name=lsc.label_ids[0],
    consider_negative=False,
)
print(trainer.train(mdl, dataset))

# Evaluate the model for all tasks
dataset = Dataset(feature=feature, label=lsc, splitter=splitter)
results = trainer.fit_and_eval(mdl, dataset, consider_negative=False, reduce="mean")
print(f"Average train score = {results['train_auroc']:.4f}")
print(f"Average test score = {results['test_auroc']:.4f}")

print_expected("Average train score = 0.9971", "Average test score = 0.6986")
