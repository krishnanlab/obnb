from common import *
from Graph import SparseGraph

class TestIDmap(unittest.TestCase):
	def setUp(self):
		self.path = '../sample_data/'
		self.t1w_pth = self.path + 'toy1_weighted.edg'
		self.t1u_pth = self.path + 'toy1_unweighted.edg'
		self.IDlst1 = ['1','3','4','2','5']
		self.data_unweighted1 = [{1:1,2:1},{0:1,4:1},{3:1,0:1},{2:1},{1:1}]
		self.graph = SparseGraph()

	def tearDown(self):
		self.graph = None

	def test_read_edglst_unweighted1(self):
		self.graph.read_edglst(self.t1u_pth, weighted=False, directed=False)
		self.assertEqual(self.graph.IDmap.lst, self.IDlst1)
		self.assertEqual(self.graph.edge_data, self.data_unweighted1)


if __name__ == '__main__':
	unittest.main()