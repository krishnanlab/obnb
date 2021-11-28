from sys import path

path.append("../")
import numpy as np
from NLEval import label, graph, valsplit, model, metrics, wrapper

data_path = "../../data/"
network_fp = data_path + "networks/BioGRID_3.4.136.edg"
labelset_fp = (
    data_path
    + "labels/c2.cp.kegg.v6.1.entrez.BP.gsea-min10-max200-ovlppt7-jacpt5.nonred.gmt"
)

workers = 8  # number of parallel processes
n_split = 5  # cross validation split number for evalution
p_thresh = 0.05  # p-val for hypergeometric test to determin negatives
min_labelset_size = 50  # minimum number of positive required in a labelset
score_cutoff = 1.2  # minimum score required for prediction

g = graph.SparseGraph.SparseGraph.from_edglst(
    network_fp, weighted=False, directed=False
)
lsc = label.LabelsetCollection.SplitLSC.from_gmt(labelset_fp)
lsc.valsplit = valsplit.Interface.SklSKF(
    shuffle=True, skl_kws={"n_splits": n_split}
)

lsc.apply(
    label.Filter.EntityExistanceFilter(target_lst=g.IDmap.lst), inplace=True
)
lsc.apply(
    label.Filter.LabelsetRangeFilterSize(min_val=min_labelset_size),
    inplace=True,
)
lsc.apply(label.Filter.NegativeFilterHypergeom(p_thresh=p_thresh), inplace=True)
print(
    f"After filtering, there are {len(lsc.labelIDlst)} number of effective labelsets"
)

scoring_obj = lambda estimator, X, y: metrics.auPRC(
    y, estimator.decision_function(X)
)
mdl = model.SupervisedLearning.LogRegCV(
    g,
    penalty="l2",
    solver="liblinear",
    max_iter=500,
    cv=3,
    Cs=np.logspace(-8, 4, 20),
    class_weight="balanced",
    n_jobs=1,
    scoring=scoring_obj,
)


@wrapper.ParWrap.ParDat(lsc.labelIDlst, n_workers=1)
def predict_all_labelsets(labelID):
    np.random.seed()  # initialize random states for parallel processes

    pos_ID_set = lsc.getLabelset(labelID)
    neg_ID_set = lsc.getNegative(labelID)

    y_true, y_predict = mdl.test2(lsc.splitLabelset(labelID))
    score = np.mean([metrics.auPRC(i, j) for i, j in zip(y_true, y_predict)])

    if score > score_cutoff:
        status_str = "(Prediction saved)"
    #        score_dict = mdl.predict(pos_ID_set, neg_ID_set)
    #
    #        with open(f"predictions/{labelID}_score={score:3.2f}.tsv", 'w') as f:
    #            f.write("gene_id\tprediction_score\tannotation\n")
    #            for geneID, prediction_score in score_dict.items():
    #                if geneID in pos_ID_set:
    #                    annotation = '+'
    #                elif geneID in neg_ID_set:
    #                    annotation = '-'
    #                else:
    #                    annotation = '0'
    #
    #                f.write(f"{geneID}\t{prediction_score}\t{annotation}\n")
    else:
        status_str = "(Discarded)"

    print(
        f"{labelID:<60} num_pos={len(pos_ID_set):>4}, "
        f"num_neg={len(neg_ID_set):>4}, score={score:>3.2f} {status_str}"
    )


# depoly processes
for i in predict_all_labelsets():
    pass
