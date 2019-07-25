from common import *
from copy import deepcopy
from NLEval.graph import BaseGraph, SparseGraph, DenseGraph

def shuffle_adjlst(adjlst):
	n = adjlst.size
	shuffle_idx = np.random.choice(n, size=n, replace=False)
	new_adjlst = SparseGraph.SparseGraph(weighted=adjlst.weighted, directed=adjlst.directed)
	for i in shuffle_idx:
		ID = adjlst.IDmap.lst[i]
		new_adjlst.addID(ID)
	for idx1, ID1 in enumerate(adjlst.IDmap):
		for idx2, weight in adjlst.edge_data[adjlst.IDmap[ID1]].items():
			ID2 = adjlst.IDmap.lst[idx2]
			new_adjlst.addEdge(ID1, ID2, weight)
	return new_adjlst

class test_case1:
	def __init__(self):
		self.tw_fp = SAMPLE_DATA_PATH + 'toy1_weighted.edg'
		self.tu_fp = SAMPLE_DATA_PATH + 'toy1_unweighted.edg'
		self.IDlst = ['1','3','4','2','5']
		self.data_unweighted = [{1:1,2:1},{0:1,4:1},{3:1,0:1},{2:1},{1:1}]
		self.data_weighted = [{1:0.4},{0:0.4,4:0.1},{3:0.3},{2:0.3},{1:0.1}]
		self.data_mat = np.array([
			[1,		0,		0,		0.4,	0,		0],
			[4,		0,		0,		0,		0.3,	0],
			[3,		0.4,	0,		0,		0,		0.1],
			[2,		0,		0.3,	0,		0,		0],
			[5,		0,		0,		0.1,	0,		0]])

class TestSparseGraph(unittest.TestCase):
	def setUp(self):
		self.case = test_case1()
		self.lst = [['1','4'], ['2','5'], ['5','3','2']]

	def test_read_edglst_unweighted(self):
		graph = SparseGraph.SparseGraph.from_edglst(self.case.tu_fp, weighted=False, directed=False)
		self.assertEqual(graph.IDmap.lst, self.case.IDlst)
		self.assertEqual(graph.edge_data, self.case.data_unweighted)

	def test_read_edglst_weighted(self):
		graph = SparseGraph.SparseGraph.from_edglst(self.case.tw_fp, weighted=True, directed=False)
		self.assertEqual(graph.IDmap.lst, self.case.IDlst)
		self.assertEqual(graph.edge_data, self.case.data_weighted)

	def test_read_npymat_weighted(self):
		graph = SparseGraph.SparseGraph.from_npy(self.case.data_mat, weighted=True, directed=False)
		self.assertEqual(graph.IDmap.lst, self.case.IDlst)
		self.assertEqual(graph.edge_data, self.case.data_weighted)	

	def template_test_construct_adj_vec(self, weighted, directed, lst=None):
		graph = SparseGraph.SparseGraph.from_npy(self.case.data_mat, weighted=weighted, directed=directed)
		adjmat = graph.to_adjmat()
		if not lst:
			lst = graph.IDmap.lst
		for ID_lst in graph.IDmap.lst:
			with self.subTest(i=ID_lst):
				idx_lst = graph.IDmap[ID_lst]
				self.assertEqual(list(graph[ID_lst]), list(adjmat[idx_lst]))

	def test_construct_adj_vec_weighted(self):
		self.template_test_construct_adj_vec(weighted=True, directed=False)

	def test_construct_adj_vec_unweighted(self):
		self.template_test_construct_adj_vec(weighted=False, directed=False)

	def test_construct_adj_vec_weighted_multiple(self):
		self.template_test_construct_adj_vec(weighted=True, directed=False, lst=self.lst)

	def test_construct_adj_vec_unweighted_multiple(self):
		self.template_test_construct_adj_vec(weighted=False, directed=False, lst=self.lst)

	def test_eq(self):
		graph = SparseGraph.SparseGraph.from_npy(self.case.data_mat, weighted=True, directed=False)
		for i in range(5):
			#repeat shuffle test
			with self.subTest(i=i):
				self.assertTrue(graph == shuffle_adjlst(graph))
		graph2 = deepcopy(graph)
		graph2.addID('x')
		self.assertFalse(graph == graph2)

class TestDenseGraph(unittest.TestCase):
	pass

class TestFeatureVec(unittest.TestCase):
	pass

if __name__ == '__main__':
	unittest.main()