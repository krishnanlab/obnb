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

mdl = model.LabelPropagation.LP(g)

for labelID in lsc.labelIDlst:
	print('%.4f\t%s'%(auroc(*(mdl.test(lsc.splitLabelset(labelID)))), labelID))