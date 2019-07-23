from common import *
from NLEval.Graph import SparseGraph

class TestIDmap(unittest.TestCase):
	def setUp(self):
		self.path = '../sample_data/'
		self.t1w_pth = self.path + 'toy1_weighted.edg'
		self.t1u_pth = self.path + 'toy1_unweighted.edg'
		self.IDlst1 = ['1','3','4','2','5']
		self.data_unweighted1 = [{1:1,2:1},{0:1,4:1},{3:1,0:1},{2:1},{1:1}]
		self.data_weighted1 = [{1:0.4},{0:0.4,4:0.1},{3:0.3},{2:0.3},{1:0.1}]
		self.data_mat1 = np.array([
			[1,		0,		0,		0.4,	0,		0],
			[4,		0,		0,		0,		0.3,	0],
			[3,		0.4,	0,		0,		0,		0.1],
			[2,		0,		0.3,	0,		0,		0],
			[5,		0,		0,		0.1,	0,		0]])

	def tearDown(self):
		self.graph = None

	def test_read_edglst_unweighted1(self):
		self.graph = SparseGraph(weighted=False, directed=False)
		self.graph.read(self.t1u_pth)
		self.assertEqual(self.graph.IDmap.lst, self.IDlst1)
		self.assertEqual(self.graph.edge_data, self.data_unweighted1)

	def test_read_edglst_weighted1(self):
		self.graph = SparseGraph(weighted=True, directed=False)
		self.graph.read(self.t1w_pth)
		self.assertEqual(self.graph.IDmap.lst, self.IDlst1)
		self.assertEqual(self.graph.edge_data, self.data_weighted1)

	def test_read_npymat_weighted1(self):
		self.graph = SparseGraph(weighted=False, directed=False)
		self.graph.read(self.data_mat1, reader='npy')
		self.assertEqual(self.graph.IDmap.lst, self.IDlst1)
		self.assertEqual(self.graph.edge_data, self.data_weighted1)

if __name__ == '__main__':
	unittest.main()