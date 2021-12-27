from sys import path

path.append("../")
from NLEval import graph, valsplit, label, model
from sklearn.metrics import roc_auc_score as auroc
import numpy as np

data_path = "../../data/"

g = graph.DenseGraph.DenseGraph.from_edglst(
    data_path + "networks/STRING-EXP.edg",
    weighted=True,
    directed=False,
)
lsc = label.LabelsetCollection.SplitLSC.from_gmt(
    data_path + "labels/KEGGBP.gmt",
)
lsc.apply(label.Filter.EntityExistanceFilter(g.idmap.lst), inplace=True)
lsc.apply(label.Filter.LabelsetRangeFilterSize(min_val=50), inplace=True)
lsc.load_entity_properties(
    data_path + "/properties/pubcnt.txt",
    "Pubmed Count",
    0,
    int,
)
# lsc.valsplit = valsplit.Holdout.BinHold(3, shuffle=True)
lsc.valsplit = valsplit.Holdout.TrainValTest(
    train_ratio=1 / 3,
    test_ratio=1 / 3,
    shuffle=True,
)

print(f"Number of labelsets before filtering: {len(lsc.label_ids)}")
lsc.train_test_setup(g, prop_name="Pubmed Count", min_pos=10)
print(f"Number of labelsets after filtering: {len(lsc.label_ids)}")

mdl = model.SupervisedLearning.LogReg(g, penalty="l2", solver="lbfgs")

score_lst = []
for label_id in lsc.label_ids:
    score = auroc(*(mdl.test(lsc.split_labelset(label_id))))
    score_lst.append(score)
    print(f"{score:.4f}\t{label_id}")

print(
    f"Average score = {np.mean(score_lst):.4f}, std = {np.std(score_lst):.4f}",
)

"""
#for printing average properties in training/testing sets
for label_id in lsc.label_ids:
    tr, _, ts, _ = next(lsc.split_labelset(label_id))
    avg_tr_prop = np.array([lsc.entity.getProp(i, 'Pubmed Count') for i in tr]).mean()
    avg_ts_prop = np.array([lsc.entity.getProp(i, 'Pubmed Count') for i in ts]).mean()
    print('Avg train prop: %.6f\t Avg test prop: %.6f'%\
        (avg_tr_prop, avg_ts_prop))
"""
