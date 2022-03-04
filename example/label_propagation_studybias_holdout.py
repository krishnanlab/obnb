import numpy as np
from load_data import load_data
from NLEval.label.split import RatioPartition
from NLEval.model.label_propagation import OneHopPropagation
from NLEval.model_trainer import LabelPropagationTrainer
from sklearn.metric import roc_auc_score as auroc


# Load dataset
g, lsc = load_data("STRING-EXP", "KEGGBP")

# 3/2 train/test split using genes with higher PubMed Count for training
splitter = RatioPartition(0.6, 0.4, ascending=False)

# Select model
mdl = OneHopPropagation()

# Setup trainer, use auroc as the evaluation metric
metrics = {"auroc": auroc}
trainer = LabelPropagationTrainer(metrics, g)

scores = []
for label_id in lsc.label_ids:
    y, masks = lsc.split(
        splitter,
        target_ids=g.node_ids,
        labelset_name=label_id,
        property_name="PubMed Count",
        consider_negative=True,
    )
    results = trainer.train(mdl, y, masks)
    scores.append(results["test_auroc"])
    train_score, test_score = results["train_auroc"], results["test_auroc"]
    print(f"Train: {train_score:.4f}\tTest: {test_score:.4f}\t{label_id}")

print(f"Average test score = {np.mean(scores):.4f}, std = {np.std(scores):.4f}")

print(
    """
Expected outcome
--------------------------------------------------------------------------------
NETWORK='STRING-EXP'
LABEL='KEGGBP'
Number of labelsets before filtering: 139
Number of labelsets after filtering: 54
Average test score = 0.7513, std = 0.1181
--------------------------------------------------------------------------------
""",
)
