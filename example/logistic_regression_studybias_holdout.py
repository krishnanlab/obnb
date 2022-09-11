import numpy as np
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
trainer = SupervisedLearningTrainer(metrics)

scores = []
for label_id in lsc.label_ids:
    dataset = Dataset(
        feature=feature,
        label=lsc,
        splitter=splitter,
        labelset_name=label_id,
        consider_negative=True,
    )
    results = trainer.train(mdl, dataset)
    scores.append(results["test_auroc"])
    train_score, test_score = results["train_auroc"], results["test_auroc"]
    print(f"Train: {train_score:.4f}\tTest: {test_score:.4f}\t{label_id}")

print(f"Average test score = {np.mean(scores):.4f}, std = {np.std(scores):.4f}")
print_expected("Average test score = 0.7000, std = 0.0949")
