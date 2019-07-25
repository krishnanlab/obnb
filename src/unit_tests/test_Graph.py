from common import *
from copy import deepcopy
from NLEval.graph import BaseGraph, SparseGraph, DenseGraph

def shuffle_sparse(graph):
	n = graph.size
	shuffle_idx = np.random.choice(n, size=n, replace=False)
	new_graph = SparseGraph.SparseGraph(weighted=graph.weighted, directed=graph.directed)
	for i in shuffle_idx:
		ID = graph.IDmap.lst[i]
		new_graph.addID(ID)
	for idx1, ID1 in enumerate(graph.IDmap):
		for idx2, weight in graph.edge_data[graph.IDmap[ID1]].items():
			ID2 = graph.IDmap.lst[idx2]
			new_graph.addEdge(ID1, ID2, weight)
	return new_graph

def shuffle_dense(graph):
	n = graph.size
	shuffle_idx = np.random.choice(n, size=n, replace=False)
	new_graph = DenseGraph.DenseGraph()
	new_graph.mat = np.zeros(graph.mat.shape)

	for i in shuffle_idx:
		ID = graph.IDmap.lst[i]
		new_graph.IDmap.addID(ID)
	for idx1_new, idx1_old in enumerate(shuffle_idx):
		for idx2_new, idx2_old in enumerate(shuffle_idx):
			new_graph.mat[idx1_new, idx2_new] = graph.mat[idx1_old, idx2_old]
	return new_graph

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
	@classmethod
	def setUpClass(cls):
		cls.case = test_case1()
		cls.lst = [['1','4'], ['2','5'], ['5','3','2']]

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
				self.assertTrue(graph == shuffle_sparse(graph))
		graph2 = deepcopy(graph)
		graph2.addID('x')
		self.assertFalse(graph == graph2)

class TestDenseGraph(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.case = test_case1()

	def test_size(self):
		graph = DenseGraph.DenseGraph.from_edglst(self.case.tw_fp, weighted=True, directed=False)
		self.assertEqual(graph.size, len(self.case.IDlst))

	def check_graph(self, graph):
		"""compare graph with data, true if identical"""
		mat = self.case.data_mat[:,1:]
		IDlst = [str(int(i)) for i in self.case.data_mat[:,0]]
		for idx1, ID1 in enumerate(IDlst):
			for idx2, ID2 in enumerate(IDlst):
				with self.subTest(idx1=idx1, idx2=idx2, ID1=type(ID1), ID2=ID2):
					self.assertEqual(mat[idx1, idx2], graph.get_edge(ID1, ID2))

	def test_from_edglst(self):
		graph = DenseGraph.DenseGraph.from_edglst(self.case.tw_fp, weighted=True, directed=False)
		self.check_graph(graph)

	def test_from_mat(self):
		graph = DenseGraph.DenseGraph.from_mat(self.case.data_mat)
		self.check_graph(graph)

	def test_eq(self):
		graph = DenseGraph.DenseGraph.from_edglst(self.case.tw_fp, weighted=True, directed=False)
		for i in range(5):
			#repeat shuffle test
			with self.subTest(i=i):
				self.assertTrue(graph == shuffle_dense(graph))
		graph2 = deepcopy(graph)
		graph2.mat[2,2] = 1
		self.assertFalse(graph == graph2)

class TestFeatureVec(unittest.TestCase):
	pass

if __name__ == '__main__':
	unittest.main()