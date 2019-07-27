from common import *
from NLEval.util import IDHandler
from NLEval.util import checkers

class TestIDlst(unittest.TestCase):
	def setUp(self):
		self.IDlst1 = IDHandler.IDlst()
		self.IDlst2 = IDHandler.IDlst()
		self.IDlst3 = IDHandler.IDlst()
		self.IDlst4 = IDHandler.IDlst()
		self.lst = ['a','b','c']
		for i in self.lst:
			self.IDlst1.addID(i)
			self.IDlst2.addID(i)
		self.IDlst3.addID('a')
		self.IDlst3.addID('b')
		self.IDlst4.addID('c')

	def test_iter(self):
		for i, j in zip(self.IDlst1, self.lst):
			self.assertEqual(i, j)

	def template_test_type_consistency(self, fun):
		self.assertRaises(TypeError, fun, self.lst)
		self.assertRaises(TypeError, fun, 10)

	def test_eq(self):
		self.assertTrue(self.IDlst1 == self.IDlst2)
		self.assertFalse(self.IDlst1 != self.IDlst2)
		self.IDlst1.addID('d')
		self.assertTrue(self.IDlst1 != self.IDlst2)
		self.IDlst2.addID('d')
		self.assertTrue(self.IDlst1 == self.IDlst2)
		#test if orders matter
		self.IDlst1.addID('e')
		self.IDlst1.addID('f')
		self.IDlst2.addID('f')
		self.IDlst2.addID('e')
		self.assertTrue(self.IDlst1 == self.IDlst2)
		self.template_test_type_consistency(self.IDlst1.__eq__)

	def test_add(self):
		self.assertTrue(self.IDlst1 == self.IDlst3 + self.IDlst4)
		self.IDlst4.addID('d')
		self.assertFalse(self.IDlst1 == self.IDlst3 + self.IDlst4)
		self.IDlst1.addID('d')
		self.assertTrue(self.IDlst1 == self.IDlst3 + self.IDlst4)
		self.template_test_type_consistency(self.IDlst1.__add__)

	def test_sub(self):
		self.assertTrue(self.IDlst3 == self.IDlst1 - self.IDlst4)
		self.assertTrue(self.IDlst4 == self.IDlst1 - self.IDlst3)
		self.assertTrue(IDHandler.IDlst() == self.IDlst1 - self.IDlst1)
		self.template_test_type_consistency(self.IDlst1.__sub__)

	def test_contains(self):
		for i in self.lst:
			self.assertTrue(i in self.IDlst1)
		self.assertTrue('c' not in self.IDlst3)

	def test_getitem(self):
		for idx, ID in enumerate(self.lst):
			self.assertEqual(self.IDlst1[ID], idx)
		self.assertRaises(AssertionError, self.IDlst1.__getitem__, 'd')
		self.assertTrue(all(self.IDlst1[self.lst] == np.array([0, 1, 2])))
		self.assertRaises(AssertionError, self.IDlst3.__getitem__, self.lst)
		self.assertRaises(TypeError, self.IDlst3.__getitem__, ['a', 0])

	def test_size(self):
		self.assertEqual(self.IDlst1.size, 3)
		self.assertEqual(self.IDlst3.size, 2)
		self.assertEqual(IDHandler.IDlst().size, 0)

	def test_copy(self):
		idlst_shallow_copy = self.IDlst1
		idlst_deep_copy = self.IDlst1.copy()
		#shallow
		self.IDlst1.addID('d')
		self.assertEqual(idlst_shallow_copy, self.IDlst1)
		#deep
		self.assertNotEqual(idlst_deep_copy, self.IDlst1)

	def test_popID(self):
		self.IDlst1.popID('c')
		self.assertEqual(self.IDlst1, self.IDlst3)
		self.assertRaises(AssertionError, self.IDlst1.popID, 'c')
		self.assertRaises(TypeError, self.IDlst1.popID, 1)

	def test_addID(self):
		self.assertEqual(self.IDlst1.lst, ['a', 'b', 'c'])
		#test basic addID with string
		self.IDlst1.addID('d')
		self.assertEqual(self.IDlst1.lst, ['a', 'b', 'c', 'd'])
		#test addID with transformation from int to string
		self.IDlst1.addID(10)
		self.assertEqual(self.IDlst1.lst, ['a', 'b', 'c', 'd', '10'])
		#test addID with transformation from float type inteter to string
		self.IDlst1.addID(11.0)
		self.assertEqual(self.IDlst1.lst, ['a', 'b', 'c', 'd', '10', '11'])
		#test addID with transformation from float to string
		self.IDlst1.addID(11.1)
		self.assertEqual(self.IDlst1.lst, ['a', 'b', 'c', 'd', '10', '11', '11.1'])
		#test add existing ID --> error
		self.assertRaises(AssertionError, self.IDlst1.addID, '10')
		#test type checking
		self.assertRaises(TypeError, self.IDlst1.addID, (1,2,))

	def test_getID(self):
		for idx, ID in enumerate(self.lst):
			self.assertEqual(self.IDlst1.getID(idx), ID)
		#test type check
		self.assertRaises(TypeError, self.IDlst1.getID, 'asdf')

class TestIDmap(unittest.TestCase):
	def setUp(self):
		self.IDmap = IDHandler.IDmap()
		self.IDmap.addID('a')

	def tearDown(self):
		self.IDmap = None

	def test_size(self):
		self.assertEqual(self.IDmap.size, 1)
		self.IDmap.addID('b')
		self.assertEqual(self.IDmap.size, 2)

	def test_map(self):
		self.assertEqual(self.IDmap.map, {'a':0})
		self.IDmap.addID('b')
		self.assertEqual(self.IDmap.map, {'a':0,'b':1})

	def test_lst(self):
		self.assertEqual(self.IDmap.lst, ['a'])
		self.IDmap.addID('b')
		self.assertEqual(self.IDmap.lst, ['a','b'])

	def test_addID(self):
		self.assertRaises(TypeError, self.IDmap.addID, (1, 2, 3))
		self.assertRaises(TypeError, self.IDmap.addID, [1, 2, 3])
		self.assertRaises(AssertionError, self.IDmap.addID, 'a')
		self.IDmap.addID('10.0')
		self.assertRaises(AssertionError, self.IDmap.addID, 10)
		self.IDmap.addID('10.1')
		self.IDmap.addID('abc')
		self.assertEqual(self.IDmap.lst, ['a', '10', '10.1', 'abc'])

	def test_getitem(self):
		self.assertEqual(self.IDmap['a'], 0)

	def test_getID(self):
		self.assertEqual(self.IDmap.getID(0), 'a')

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
		idmap = IDHandler.IDmap()
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

	def test_copy(self):
		idmap_shallow_copy = self.IDmap
		idmap_deep_copy = self.IDmap.copy()
		#shallow
		self.IDmap.addID('b')
		self.assertEqual(idmap_shallow_copy, self.IDmap)
		#deep
		self.assertNotEqual(idmap_deep_copy, self.IDmap)

	def test_popID(self):
		self.IDmap.addID('b')
		self.IDmap.addID('c')
		self.assertEqual(self.IDmap.lst, ['a', 'b', 'c'])
		self.assertEqual(self.IDmap.map, {'a':0, 'b':1, 'c':2})
		self.assertRaises(AssertionError, self.IDmap.popID, 'd')
		self.IDmap.popID('b')
		#make sure both lst and data poped
		self.assertEqual(self.IDmap.lst, ['a', 'c'])
		#make sure data updated with new mapping
		self.assertEqual(self.IDmap.map, {'a':0, 'c':1})

	def test_add(self):
		idmap = IDHandler.IDmap()
		idmap.addID('b')
		idmap_combined = self.IDmap + idmap
		self.assertEqual(idmap_combined.map, {'a':0, 'b':1})
		self.assertEqual(idmap_combined.lst, ['a', 'b'])

	def test_sub(self):
		idmap = self.IDmap.copy()
		idmap.addID('b')
		self.IDmap.addID('c')
		diff = idmap - self.IDmap
		self.assertEqual(diff.map, {'b':0})
		self.assertEqual(diff.lst, ['b'])
		diff = self.IDmap - idmap
		self.assertEqual(diff.map, {'c':0})
		self.assertEqual(diff.lst, ['c'])

class TestCheckers(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		n = 10
		self.n = n
		self.n_str = str(n)
		self.n_int = int(n)
		self.n_npint = np.int(n)
		self.n_npint64 = np.int64(n)
		self.n_float = float(n)
		self.n_npfloat = np.float(n)
		self.n_npfloat128 = np.float128(n)
		self.n_int_tuple = (int(n), int(n), int(n))
		self.n_int_lst = [int(n), int(n), int(n)]
		self.n_int_npary = np.array([n, n, n], dtype=int)
		self.n_float_tuple = (float(n), float(n), float(n))
		self.n_float_lst = [float(n), float(n), float(n)]
		self.n_float_npary = np.array([n, n, n], dtype=float)

	def test_INT_TYPE(self):
		self.assertIsInstance(self.n_int, checkers.INT_TYPE)
		self.assertIsInstance(self.n_npint, checkers.INT_TYPE)
		self.assertIsInstance(self.n_npint64, checkers.INT_TYPE)
		self.assertNotIsInstance(self.n_float, checkers.INT_TYPE)
		self.assertNotIsInstance(self.n_npfloat, checkers.INT_TYPE)
		self.assertNotIsInstance(self.n_npfloat128, checkers.INT_TYPE)

	def test_FLOAT_TYPE(self):
		self.assertNotIsInstance(self.n_int, checkers.FLOAT_TYPE)
		self.assertNotIsInstance(self.n_npint, checkers.FLOAT_TYPE)
		self.assertNotIsInstance(self.n_npint64, checkers.FLOAT_TYPE)
		self.assertIsInstance(self.n_float, checkers.FLOAT_TYPE)
		self.assertIsInstance(self.n_npfloat, checkers.FLOAT_TYPE)
		self.assertIsInstance(self.n_npfloat128, checkers.FLOAT_TYPE)

	def test_ITERABLE_TYPE(self):
		n_int_tuple = (1, 2, 3)
		n_int_lst = [1, 2, 3]
		n_int_ary = np.array([1, 2, 3])
		self.assertIsInstance(n_int_tuple, checkers.ITERABLE_TYPE)
		self.assertIsInstance(n_int_lst, checkers.ITERABLE_TYPE)
		self.assertIsInstance(n_int_ary, checkers.ITERABLE_TYPE)

	def test_checkType(self):
		checkers.checkType('n_int', int, self.n_int)
		self.assertRaises(TypeError, checkers.checkType, 'n_int', int, self.n_float)
		checkers.checkType('n_float', float, self.n_float)
		self.assertRaises(TypeError, checkers.checkType, 'n_float', float, self.n_int)
		checkers.checkType('n_str', str, self.n_str)

	def test_checkTypeInIterable(self):
		checkers.checkTypesInIterable('n_int_tuple', checkers.INT_TYPE, self.n_int_tuple)
		checkers.checkTypesInIterable('n_int_lst', checkers.INT_TYPE, self.n_int_lst)
		checkers.checkTypesInIterable('n_int_npary', checkers.INT_TYPE, self.n_int_npary)
		self.assertRaises(TypeError, checkers.checkTypesInIterable, \
			'n_int_tuple', checkers.FLOAT_TYPE, self.n_int_tuple)
		self.assertRaises(TypeError, checkers.checkTypesInIterable, \
			'n_int_lst', checkers.FLOAT_TYPE, self.n_int_lst)
		self.assertRaises(TypeError, checkers.checkTypesInIterable, \
			'n_int_npary', checkers.FLOAT_TYPE, self.n_int_npary)
		checkers.checkTypesInIterable('n_float_tuple', checkers.FLOAT_TYPE, self.n_float_tuple)
		checkers.checkTypesInIterable('n_float_lst', checkers.FLOAT_TYPE, self.n_float_lst)
		checkers.checkTypesInIterable('n_float_npary', checkers.FLOAT_TYPE, self.n_float_npary)
		self.assertRaises(TypeError, checkers.checkTypesInIterable, \
			'n_float_tuple', checkers.INT_TYPE, self.n_float_tuple)
		self.assertRaises(TypeError, checkers.checkTypesInIterable, \
			'n_float_lst', checkers.INT_TYPE, self.n_float_lst)
		self.assertRaises(TypeError, checkers.checkTypesInIterable, \
			'n_float_npary', checkers.INT_TYPE, self.n_float_npary)

	def test_checkTypeErrNone(self):
		self.assertRaises(ValueError, checkers.checkTypeErrNone, 'n', int, None)
		self.assertRaises(ValueError, checkers.checkTypeErrNone, 'n', float, None)
		self.assertRaises(ValueError, checkers.checkTypeErrNone, 'n', str, None)

	def test_checkTypeAllowNone(self):
		checkers.checkTypeAllowNone('n', int, None)
		checkers.checkTypeAllowNone('n', float, None)
		checkers.checkTypeAllowNone('n', str, None)

	def test_checkNumpyArrayNDim(self):
		ary1 = np.ones(2)
		ary2 = np.ones((2,2))
		ary3 = np.ones((2,2,2))
		checkers.checkNumpyArrayNDim('ary1', 1, ary1)
		checkers.checkNumpyArrayNDim('ary2', 2, ary2)
		checkers.checkNumpyArrayNDim('ary3', 3, ary3)
		self.assertRaises(ValueError, checkers.checkNumpyArrayNDim, 'ary1', 4, ary1)
		self.assertRaises(ValueError, checkers.checkNumpyArrayNDim, 'ary2', 4, ary2)
		self.assertRaises(ValueError, checkers.checkNumpyArrayNDim, 'ary3', 4, ary3)

	def test_checkNumpyArrayShape(self):
		ary1 = np.ones(2)
		ary2 = np.ones((2,2))
		ary3 = np.ones((2,2,2))
		checkers.checkNumpyArrayShape('ary1', 2, ary1)
		checkers.checkNumpyArrayShape('ary1', (2,), ary1)
		checkers.checkNumpyArrayShape('ary2', (2, 2), ary2)
		checkers.checkNumpyArrayShape('ary3', (2, 2, 2), ary3)
		self.assertRaises(ValueError, checkers.checkNumpyArrayShape, 'ary1', 1, ary1)
		self.assertRaises(ValueError, checkers.checkNumpyArrayShape, 'ary1', (1, 2,), ary1)
		self.assertRaises(ValueError, checkers.checkNumpyArrayShape, 'ary2', (2, 1), ary2)
		self.assertRaises(ValueError, checkers.checkNumpyArrayShape, 'ary3', 1, ary3)

	def test_checkNumpyArrayIsNumeric(self):
		#test numpy numeric array --> success
		ary1 = np.random.random(3).astype(int)
		ary2 = np.random.random((3, 3)).astype(float)
		ary3 = np.random.random(5)
		checkers.checkNumpyArrayIsNumeric('ar1', ary1)
		checkers.checkNumpyArrayIsNumeric('ar2', ary2)
		checkers.checkNumpyArrayIsNumeric('ar3', ary3)

		#test numpy string array --> error
		ary1 = np.ones(3, dtype=str)
		ary2 = np.ones((3,3), dtype=str)
		self.assertRaises(TypeError, checkers.checkNumpyArrayIsNumeric, 'ar1', ary1)
		self.assertRaises(TypeError, checkers.checkNumpyArrayIsNumeric, 'ar2', ary2)

		#test list --> error
		lst = [1,2,3]
		self.assertRaises(TypeError, checkers.checkNumpyArrayIsNumeric, 'lst', lst)

		#test string --> error
		string = 'asdfgasdglafknklasd'
		self.assertRaises(TypeError, checkers.checkNumpyArrayIsNumeric, 'string', string)

if __name__ == '__main__':
	unittest.main()
