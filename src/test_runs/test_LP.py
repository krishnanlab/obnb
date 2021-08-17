from sys import path
path.append('../')
from NLEval import graph, valsplit, label, model
from sklearn.metrics import roc_auc_score as auroc
import numpy as np

data_path = '../../data/'

g = graph.DenseGraph.DenseGraph.from_edglst(data_path \
    + 'networks/STRING-EXP.edg', weighted=True, directed=False)
lsc = label.LabelsetCollection.SplitLSC.from_gmt(data_path + 'labels/KEGGBP.gmt')
lsc.apply(label.Filter.EntityExistanceFilter(g.IDmap.lst), inplace=True)
lsc.apply(label.Filter.LabelsetRangeFilterSize(min_val=50), inplace=True)
lsc.load_entity_properties(data_path + '/properties/pubcnt.txt', \
        'Pubmed Count', 0, int)
#lsc.valsplit = valsplit.Holdout.BinHold(3, shuffle=True)
lsc.valsplit = valsplit.Holdout.TrainValTest(train_ratio=1/3, test_ratio=1/3, shuffle=True)

print(f"Number of labelsets before filtering: {len(lsc.labelIDlst)}")
lsc.train_test_setup(g, prop_name='Pubmed Count', min_pos=10)
print(f"Number of labelsets after filtering: {len(lsc.labelIDlst)}")

mdl = model.LabelPropagation.LP(g)

score_lst = []
for labelID in lsc.labelIDlst:
    score = auroc(*(mdl.test(lsc.splitLabelset(labelID))))
    score_lst.append(score)
    print(f"{score:.4f}\t{labelID}")

print(f"Average score = {np.mean(score_lst):.4f}, std = {np.std(score_lst):.4f}")
