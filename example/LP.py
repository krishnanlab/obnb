import numpy as np
from NLEval import graph
from NLEval import label
from NLEval import model
from NLEval import valsplit
from sklearn.metrics import roc_auc_score as auroc

DATA_DIR = "../data/"

g = graph.DenseGraph.DenseGraph.from_edglst(
    DATA_DIR + "networks/STRING-EXP.edg",
    weighted=True,
    directed=False,
)
lsc = label.labelset_collection.SplitLSC.from_gmt(
    DATA_DIR + "labels/KEGGBP.gmt",
)
lsc.apply(
    label.labelset_filter.EntityExistanceFilter(g.idmap.lst),
    inplace=True,
)
lsc.apply(
    label.labelset_filter.LabelsetRangeFilterSize(min_val=50),
    inplace=True,
)
lsc.load_entity_properties(
    DATA_DIR + "/properties/pubcnt.txt",
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

mdl = model.LabelPropagation.LP(g)

score_lst = []
for label_id in lsc.label_ids:
    score = auroc(*(mdl.test(lsc.split_labelset(label_id))))
    score_lst.append(score)
    print(f"{score:.4f}\t{label_id}")

print(
    f"Average score = {np.mean(score_lst):.4f}, std = {np.std(score_lst):.4f}",
)
