import numpy as np
from load_data import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auroc

from NLEval import Dataset
from NLEval.label.split import RatioPartition
from NLEval.model_trainer import SupervisedLearningTrainer
from NLEval.util.parallel import ParDatMap

progressbar = True

# Load dataset
g, lsc = load_data("STRING-EXP", "KEGGBP")
dataset = Dataset(feature=g.to_feature())

# 3/2 train/test split using genes with higher PubMed Count for training
splitter = RatioPartition(0.6, 0.4, ascending=False)

# Select model
mdl = LogisticRegression(penalty="l2", solver="lbfgs")

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
        property_name="PubMed Count",
        consider_negative=True,
    )
    results = trainer.train(mdl, dataset, y, masks)
    tr_score, ts_score = results["train_auroc"], results["test_auroc"]

    if not progressbar:
        print(f"Train: {tr_score:>3.2f} Test: {ts_score:>3.2f} {label_id:<60}")

    return tr_score, ts_score


train_scores, test_scores = zip(*predict_all_labelsets())
train_scores, test_scores = np.array(train_scores), np.array(test_scores)
print(
    f"Average training score = {np.mean(train_scores):.4f}, "
    f"std = {np.std(train_scores):.4f}\n"
    f"Average testing score = {np.mean(test_scores):.4f}, "
    f"std = {np.std(test_scores):.4f}",
)

print(
    """
Expected outcome (TODO: this does not seem correct, check against logreg ex)
--------------------------------------------------------------------------------
Average training score = 0.9999, std = 0.0003
Average testing score = 0.8579, std = 0.0941
--------------------------------------------------------------------------------
""",
)
