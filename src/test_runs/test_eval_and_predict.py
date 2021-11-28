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
    label.Filter.EntityExistanceFilter(target_lst=g.idmap.lst), inplace=True
)
lsc.apply(
    label.Filter.LabelsetRangeFilterSize(min_val=min_labelset_size),
    inplace=True,
)
lsc.apply(label.Filter.NegativeFilterHypergeom(p_thresh=p_thresh), inplace=True)
print(
    f"After filtering, there are {len(lsc.label_ids)} number of effective labelsets"
)

scoring_obj = lambda estimator, X, y: metrics.log2_auprc_prior(
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


@wrapper.ParWrap.ParDat(lsc.label_ids, n_workers=1)
def predict_all_labelsets(label_id):
    np.random.seed()  # initialize random states for parallel processes

    pos_ids_set = lsc.get_labelset(label_id)
    neg_ids_set = lsc.get_negative(label_id)

    y_true, y_predict = mdl.test2(lsc.split_labelset(label_id))
    score = np.mean(
        [metrics.log2_auprc_prior(i, j) for i, j in zip(y_true, y_predict)]
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

    print(
        f"{label_id:<60} num_pos={len(pos_ids_set):>4}, "
        f"num_neg={len(neg_ids_set):>4}, score={score:>3.2f} {status_str}"
    )


# depoly processes
for i in predict_all_labelsets():
    pass
