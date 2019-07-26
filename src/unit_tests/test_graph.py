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

	for i in shuffle_idx:
		ID = graph.IDmap.lst[i]
		new_graph.IDmap.addID(ID)
	new_graph.mat = np.zeros(graph.mat.shape)
	for idx1_new, idx1_old in enumerate(shuffle_idx):
		for idx2_new, idx2_old in enumerate(shuffle_idx):
			new_graph.mat[idx1_new, idx2_new] = graph.mat[idx1_old, idx2_old]
	return new_graph

class test_case1:
	def __init__(self):
		self.tw_fp = SAMPLE_DATA_PATH + 'toy1_weighted.edg'
		self.tu_fp = SAMPLE_DATA_PATH + 'toy1_unweighted.edg'
		self.temd_fp = SAMPLE_DATA_PATH + 'toy_STRING-EXP.emd'
		self.IDlst = ['1','3','4','2','5']
		self.data_unweighted = [{1:1,2:1},{0:1,4:1},{3:1,0:1},{2:1},{1:1}]
		self.data_weighted = [{1:0.4},{0:0.4,4:0.1},{3:0.3},{2:0.3},{1:0.1}]
		self.data_mat = np.array([
			[1,		0,		0,		0.4,	0,		0],
			[4,		0,		0,		0,		0.3,	0],
			[3,		0.4,	0,		0,		0,		0.1],
			[2,		0,		0.3,	0,		0,		0],
			[5,		0,		0,		0.1,	0,		0]])

class TestBaseGraph(unittest.TestCase):
	def setUp(self):
		self.graph = BaseGraph.BaseGraph()

	def test_IDmap_setter(self):
		with self.assertRaises(TypeError):
			self.graph.IDmap = 'asdg'

	def test_size(self):
		self.assertEqual(self.graph.size, 0)
		for i in range(5):
			with self.subTest(i=i):
				self.graph.IDmap.addID(i)
				self.assertEqual(self.graph.size, i+1)

	def test_isempty(self):
		self.assertTrue(self.graph.isempty())
		self.graph.IDmap.addID('a')
		self.assertFalse(self.graph.isempty())

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
			idx_lst = graph.IDmap[ID_lst]
			with self.subTest(ID_lst=ID_lst, idx_lst=idx_lst):
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

	def check_graph(self, graph):
		"""compare graph with data, true if identical"""
		mat = self.case.data_mat[:,1:]
		IDlst = [str(int(i)) for i in self.case.data_mat[:,0]]
		for idx1, ID1 in enumerate(IDlst):
			for idx2, ID2 in enumerate(IDlst):
				with self.subTest(idx1=idx1, idx2=idx2, ID1=type(ID1), ID2=ID2):
					self.assertEqual(mat[idx1, idx2], graph.get_edge(ID1, ID2))

	def test_mat(self):
		graph = DenseGraph.DenseGraph()
		graph.IDmap.addID('a')
		graph.IDmap.addID('b')
		graph.mat = np.random.random((2,2))
		#test type check: only numpy array allowed
		with self.assertRaises(TypeError):
			graph.mat = [[1,5],[2,5]]
		#test dtype check: only numeric numpy array allowed
		with self.assertRaises(TypeError):
			graph.mat = np.ones((2,2),dtype=str)
		#test ndim check: only 2D or empty matrix allowed
		with self.assertRaises(ValueError):
			graph.mat = np.ones((2,2,2))
		#test shape check: matrix should have same number of rows as the size of IDmap
		with self.assertRaises(ValueError):
			graph.mat = np.ones((3,2))
		graph.mat = np.random.random((2,2))

	def test_get_edge(self):
		graph = DenseGraph.DenseGraph.from_mat(self.case.data_mat)
		mat = self.case.data_mat[:,1:]
		for i in range(mat.shape[0]):
			for j in range(mat.shape[1]):
				ID1 = graph.IDmap.lst[i]
				ID2 = graph.IDmap.lst[j]
				self.assertEqual(graph.get_edge(ID1, ID2), mat[i,j])

	def test_construc_graph(self):
		idmap = DenseGraph.IDmap()
		idmap.addID('a')
		idmap.addID('b')
		mat1 = np.random.random((2,2))
		mat2 = np.random.random((3,2))
		#test consistent size input --> success
		graph = DenseGraph.DenseGraph.construct_graph(idmap, mat1)
		#test inconsistent size input --> error
		self.assertRaises(AssertionError, DenseGraph.DenseGraph.construct_graph, idmap, mat2)

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
	@classmethod
	def setUpClass(cls):
		cls.case = test_case1()
		cls.graph = DenseGraph.FeatureVec.from_emd(cls.case.temd_fp)

	def test_dim(self):
		graph = DenseGraph.FeatureVec()
		#test type int --> success
		graph.dim = int(10)
		self.assertEqual(graph.dim, 10)
		#test type np.int --> success
		graph.dim = np.int(9)
		self.assertEqual(graph.dim, 9)
		#test type np.int64 --> success
		graph.dim = np.int64(10)
		self.assertEqual(graph.dim, 10)
		#test int less than 1 --> error
		with self.assertRaises(ValueError):
			graph.dim = 0
		with self.assertRaises(ValueError):
			graph.dim = int(-10)
		#test type float --> error
		with self.assertRaises(TypeError):
			graph.dim = float(5)
		self.assertEqual(graph.dim, 10)
		#test type str --> error
		with self.assertRaises(TypeError):
			graph.dim = '5'
		self.assertEqual(graph.dim, 10)
		#test type np.float --> error
		with self.assertRaises(TypeError):
			graph.dim = np.float(5)
		self.assertEqual(graph.dim, 10)

if __name__ == '__main__':
	unittest.main()