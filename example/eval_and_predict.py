import os.path as osp

import numpy as np
from NLEval import metrics
from NLEval import model
from NLEval.graph import SparseGraph
from NLEval.label import filters
from NLEval.label.collection import SplitLSC
from NLEval.util.parallel import ParDatExe
from NLEval.valsplit.Interface import SklSKF

DATA_DIR = osp.join(osp.pardir, "data")
GRAPH_FP = osp.join(DATA_DIR, "networks", "BioGRID_3.4.136.edg")
LABEL_FP = osp.join(DATA_DIR, "labels", "KEGGBP.gmt")

workers = 8  # number of parallel processes
n_split = 5  # cross validation split number for evalution
p_thresh = 0.05  # p-val for hypergeometric test to determin negatives
min_labelset_size = 50  # minimum number of positive required in a labelset
score_cutoff = 1.2  # minimum score required for prediction
progressbar = True

g = SparseGraph.from_edglst(GRAPH_FP, weighted=False, directed=False)
lsc = SplitLSC.from_gmt(LABEL_FP)
lsc.valsplit = SklSKF(shuffle=True, skl_kws={"n_splits": n_split})

lsc.iapply(filters.EntityExistenceFilter(target_lst=g.idmap.lst))
lsc.iapply(filters.LabelsetRangeFilterSize(min_val=min_labelset_size))
lsc.iapply(filters.NegativeFilterHypergeom(p_thresh=p_thresh))
print(f"Number of labelsets after filtering = {len(lsc.label_ids)}")

# scoring_obj = lambda estimator, X, y: metrics.log2_auprc_prior(
#    y,
#    estimator.decision_function(X),
# )
# mdl = model.SupervisedLearning.LogRegCV(
#    g,
#    penalty="l2",
#    solver="liblinear",
#    max_iter=500,
#    cv=3,
#    Cs=np.logspace(-6, 2, 10),
#    class_weight="balanced",
#    n_jobs=1,
#    scoring=scoring_obj,
# )
mdl = model.SupervisedLearning.LogReg(
    g,
    penalty="l2",
    solver="liblinear",
)


@ParDatExe(lsc.label_ids, n_workers=6, verbose=progressbar)
def predict_all_labelsets(label_id):
    np.random.seed()  # initialize random states for parallel processes

    pos_ids_set = lsc.get_labelset(label_id)
    neg_ids_set = lsc.get_negative(label_id)

    y_true, y_predict = mdl.test2(lsc.split_labelset(label_id))
    score = np.mean(
        [metrics.log2_auprc_prior(i, j) for i, j in zip(y_true, y_predict)],
    )

    if score > score_cutoff:
        status_str = "(Prediction saved)"
    #        score_dict = mdl.predict(pos_ids_set, neg_ids_set)
    #
    #        with open(f"predictions/{label_id}_score={score:3.2f}.tsv", 'w') as f:
    #            f.write("gene_id\tprediction_score\tannotation\n")
    #            for geneID, prediction_score in score_dict.items():
    #                if geneID in pos_ids_set:
    #                    annotation = '+'
    #                elif geneID in neg_ids_set:
    #                    annotation = '-'
    #                else:
    #                    annotation = '0'
    #
    #                f.write(f"{geneID}\t{prediction_score}\t{annotation}\n")
    else:
        status_str = "(Discarded)"

    if not progressbar:
        print(
            f"{label_id:<60} num_pos={len(pos_ids_set):>4}, "
            f"num_neg={len(neg_ids_set):>4}, score={score:>3.2f} {status_str}",
        )


predict_all_labelsets()
