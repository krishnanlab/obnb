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

	def test_contains(self):
		self.assertTrue('a' in self.IDmap)
		self.assertFalse('b' in self.IDmap)
		self.IDmap.addID('b')
		self.assertTrue('b' in self.IDmap)

	def test_eq(self):
		self.IDmap.addID('b')
		idmap = IDmap()
		idmap.addID('b')
		idmap.addID('a')
		self.assertTrue(self.IDmap == idmap)
		self.IDmap.addID('c')
		self.IDmap.addID('d')
		idmap.addID('d')
		self.assertFalse(self.IDmap == idmap)
		idmap.addID('c')
		self.assertTrue(self.IDmap == idmap)

	def test_iter(self):
		self.IDmap.addID('b')
		self.IDmap.addID('x')
		lst = ['a', 'b', 'x']
		for i, j in enumerate(self.IDmap):
			with self.subTest(i=i):
				self.assertEqual(j, lst[i])

class TestCheckers(unittest.TestCase):
	pass

if __name__ == '__main__':
	unittest.main()
