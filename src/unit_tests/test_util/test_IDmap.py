from common import *
from NLEval.util.IDmap import IDmap

class TestIDmap(unittest.TestCase):
	def setUp(self):
		self.IDmap = IDmap()
		self.IDmap.addID('a')

	def tearDown(self):
		self.IDmap = None

	def test_size(self):
		self.assertEqual(self.IDmap.size, 1)
		self.IDmap.addID('b')
		self.assertEqual(self.IDmap.size, 2)

	def test_data(self):
		self.assertEqual(self.IDmap.data, {'a':0})
		self.IDmap.addID('b')
		self.assertEqual(self.IDmap.data, {'a':0,'b':1})

	def test_lst(self):
		self.assertEqual(self.IDmap.lst, ['a'])
		self.IDmap.addID('b')
		self.assertEqual(self.IDmap.lst, ['a','b'])

	def test_newID(self):
		self.assertRaises(AssertionError, self.IDmap.addID, 'a')

	def test_getitem(self):
		self.assertEqual(self.IDmap['a'], 0)

	def test_idx2ID(self):
		self.assertEqual(self.IDmap.idx2ID(0), 'a')

	def test_IDary2idxary(self):
		self.IDmap.addID('b')
		self.assertEqual(self.IDmap[['b','a']][0], 1)
		self.assertEqual(self.IDmap[['b','a']][1], 0)
		self.assertEqual(self.IDmap[np.array(['a','b'])][0], 0)
		self.assertEqual(self.IDmap[np.array(['a','b'])][1], 1)


if __name__ == '__main__':
	unittest.main()
