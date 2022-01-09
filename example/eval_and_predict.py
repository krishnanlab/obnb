import os.path as osp

import numpy as np
from NLEval import metrics
from NLEval import model
from NLEval.graph import DenseGraph
from NLEval.label import filters
from NLEval.label.collection import LabelsetCollection
from NLEval.label.split import RatioPartition
from NLEval.model_trainer import SupervisedLearningTrainer
from NLEval.util.parallel import ParDatMap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auroc

NETWORK = "STRING-EXP"
LABEL = "KEGGBP"
DATA_DIR = osp.join(osp.pardir, "data")
GRAPH_FP = osp.join(DATA_DIR, "networks", f"{NETWORK}.edg")
LABEL_FP = osp.join(DATA_DIR, "labels", f"{LABEL}.gmt")
PROPERTY_FP = osp.join(DATA_DIR, "properties", "PubMedCount.txt")

workers = 8  # number of parallel processes
n_split = 5  # cross validation split number for evalution
p_thresh = 0.05  # p-val for hypergeometric test to determin negatives
min_labelset_size = 50  # minimum number of positive required in a labelset
score_cutoff = 1.2  # minimum score required for prediction
progressbar = True

g = DenseGraph.from_edglst(GRAPH_FP, weighted=False, directed=False)
lsc = LabelsetCollection.from_gmt(LABEL_FP)

lsc.iapply(filters.EntityExistenceFilter(target_lst=g.idmap.lst))
lsc.iapply(filters.LabelsetRangeFilterSize(min_val=min_labelset_size))
lsc.iapply(filters.NegativeGeneratorHypergeom(p_thresh=p_thresh))
print(f"Number of labelsets after filtering = {len(lsc.label_ids)}")

lsc.load_entity_properties(PROPERTY_FP, "PubMed Count", 0, int)
splitter = RatioPartition(0.6, 0.4, ascending=False)

mdl = LogisticRegression(penalty="l2", solver="lbfgs")
metrics = {"auroc": auroc}
trainer = SupervisedLearningTrainer(metrics, g)


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
    results = trainer.train(mdl, y, masks)
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
Expected outcome
--------------------------------------------------------------------------------
Average training score = 0.9999, std = 0.0003
Average testing score = 0.8579, std = 0.0941
--------------------------------------------------------------------------------
""",
)
