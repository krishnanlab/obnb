import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auroc
from utils import load_data, print_expected

from nleval import Dataset
from nleval.label.split import RatioPartition
from nleval.model_trainer import SupervisedLearningTrainer
from nleval.util.parallel import ParDatMap

progressbar = False

# Load dataset
g, lsc, converter = load_data()

# 3/2 train/test split using genes with higher PubMed Count for training
splitter = RatioPartition(0.5, 0.5, ascending=False, property_converter=converter)

# Select model
mdl = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=500)

# Setup trainer, use auroc as the evaluation metric
metrics = {"auroc": auroc}
trainer = SupervisedLearningTrainer(metrics)


@ParDatMap(lsc.label_ids, n_workers=6, verbose=progressbar)
def predict_all_labelsets(label_id):
    # TODO: do this in the Parallel object?
    np.random.seed()  # initialize random states for parallel processes

    y, masks = lsc.split(
        splitter,
        target_ids=g.node_ids,
        labelset_name=label_id,
        consider_negative=True,
    )
    dataset = Dataset(feature=g.to_feature(), y=y, masks=masks)
    results = trainer.train(mdl, dataset)
    train_score, test_score = results["train_auroc"], results["test_auroc"]

    if not progressbar:
        print(f"Train: {train_score:.4f}\tTest: {test_score:.4f}\t{label_id}")

    return train_score, test_score


train_scores, test_scores = zip(*predict_all_labelsets())
train_scores, test_scores = np.array(train_scores), np.array(test_scores)
print(
    f"Average training score = {np.mean(train_scores):.4f}, "
    f"std = {np.std(train_scores):.4f}\n"
    f"Average testing score = {np.mean(test_scores):.4f}, "
    f"std = {np.std(test_scores):.4f}",
)

print_expected(
    "Average training score = 0.9955, std = 0.0018",
    "Average testing score = 0.6308, std = 0.0887",
)
