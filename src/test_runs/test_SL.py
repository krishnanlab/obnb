from sys import path
path.append('../')
from NLEval import graph, valsplit, label, model
from sklearn.metrics import roc_auc_score as auroc
import numpy as np

data_path = '../../data/'

g = graph.DenseGraph.DenseGraph.from_edglst(data_path \
	+ 'networks/STRING-EXP.edg', weighted=True, directed=False)
lsc = label.LabelsetCollection.SplitLSC.from_gmt(data_path + 'labels/KEGGBP.gmt')
lsc.load_entity_properties(data_path + '/properties/pubcnt.txt', \
		'Pubmed Count', 0, int)
lsc.valsplit = valsplit.Holdout.BinHold(3, shuffle=True)
lsc.train_test_setup(g, prop_name='Pubmed Count', min_pos=10)

mdl = model.SupervisedLearning.LogReg(g, penalty='l2', solver='lbfgs')

for labelID in lsc.labelIDlst:
	print('%.4f\t%s'%(auroc(*(mdl.test(lsc.splitLabelset(labelID)))), labelID))

"""
#for printing average properties in training/testing sets
for labelID in lsc.labelIDlst:
	tr, _, ts, _ = next(lsc.splitLabelset(labelID))
	avg_tr_prop = np.array([lsc.entity.getProp(i, 'Pubmed Count') for i in tr]).mean()
	avg_ts_prop = np.array([lsc.entity.getProp(i, 'Pubmed Count') for i in ts]).mean()
	print('Avg train prop: %.6f\t Avg test prop: %.6f'%\
		(avg_tr_prop, avg_ts_prop))
"""