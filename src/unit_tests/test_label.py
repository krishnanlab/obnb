from common import *
from NLEval.label import LabelsetCollection

class TestLabelsetCollection(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.toy1_fp = SAMPLE_DATA_PATH + 'toy1.gmt'
		self.toy1_labelIDlst = ['Group1', 'Group2', 'Group3', 'Group4']
		self.toy1_InfoLst = ['Description1', 'Description2', 'Description3', 'Description4']
		self.toy1_labelsets = [ {'ID1', 'ID2', 'ID3'}, \
								{'ID2', 'ID4', 'ID5', 'ID6'}, \
								{'ID2', 'ID6'}, \
								{'ID7'}]

	def setUp(self):
		self.lsc = LabelsetCollection.LabelsetCollection()
		self.lsc.addLabelset(['a', 'b', 'c'], 'Labelset1', 'Description1')
		self.lsc.addLabelset(['b', 'd'], 'Labelset2', 'Description2')
		
	def template_test_input_for_getters(self, fun):
		"""Template for testing inputs for methods with only one positional 
		argument as ID, i.e. `.getInfo`, `getLabelset`, and `getNoccur`."""
		#input type other than str --> TypeError
		self.assertRaises(TypeError, fun, 1)
		self.assertRaises(TypeError, fun, ['1'])
		#input unknown ID --> IDNotExistError
		self.assertRaises(IDNotExistError, fun, '1')

	def test_getInfo(self):
		self.template_test_input_for_getters(self.lsc.getInfo)
		self.assertEqual(self.lsc.getInfo('Labelset1'), 'Description1')
		self.assertEqual(self.lsc.getInfo('Labelset2'), 'Description2')

	def test_getLabelset(self):
		self.template_test_input_for_getters(self.lsc.getLabelset)
		self.assertEqual(self.lsc.getLabelset('Labelset1'), {'a', 'b', 'c'})
		self.assertEqual(self.lsc.getLabelset('Labelset2'), {'b', 'd'})

	def test_getNoccur(self):
		self.template_test_input_for_getters(self.lsc.getNoccur)
		self.assertEqual(self.lsc.getNoccur('a'), 1)
		self.assertEqual(self.lsc.getNoccur('b'), 2)
		self.assertEqual(self.lsc.getNoccur('c'), 1)
		self.assertEqual(self.lsc.getNoccur('d'), 1)

	def test_eq(self):
		#make two identical labelset collections by shuffling the order of labelset
		shuffle_idx = [3, 0, 2, 1]
		lsc1 = LabelsetCollection.LabelsetCollection()
		lsc2 = LabelsetCollection.LabelsetCollection()
		for idx1 in range(4):
			idx2 = shuffle_idx[idx1]
			for lsc, idx in zip((lsc1, lsc2), (idx1, idx2)):
				lsc.addLabelset(list(self.toy1_labelsets[idx]), \
					self.toy1_labelIDlst[idx], self.toy1_InfoLst[idx])
		self.assertEqual(lsc1, lsc2)
		#test if different description
		lsc3 = lsc2.copy()
		lsc3.setProp('Group1', 'Info', 'Some other description')
		self.assertNotEqual(lsc1, lsc3)
		#test if different labelset with same labelID
		lsc3 = lsc2.copy()
		lsc3.updateLabelset(['a'], 'Group1')
		self.assertNotEqual(lsc1, lsc3)
		#make sure lsc2 still the same as lsc1
		self.assertEqual(lsc1, lsc2)

	def test_from_gmt(self):
		lsc = LabelsetCollection.LabelsetCollection.from_gmt(self.toy1_fp)
		lsc2 = LabelsetCollection.LabelsetCollection.from_gmt(self.toy1_fp)
		self.assertEqual(lsc, lsc2)

	def test_addLabelset(self):
		with self.subTest(msg='Input checks'):
			#test lst input type, only list of string allowed
			self.assertRaises(TypeError, self.lsc.addLabelset, 1, 'Labelset3')
			self.assertRaises(TypeError, self.lsc.addLabelset, ['1', 2], 'Labelset3')
			self.assertRaises(TypeError, self.lsc.addLabelset, '123', 'Labelset3')
			#test label ID input type --> TypeError
			self.assertRaises(TypeError, self.lsc.addLabelset, ['a'], 123)
			self.assertRaises(TypeError, self.lsc.addLabelset, ['a'], [1,2])
			self.assertRaises(TypeError, self.lsc.addLabelset, ['a'], ['Labelset'])
			#test label info input type --> TypeError
			self.assertRaises(TypeError, self.lsc.addLabelset, ['a'], 'Labelset3', [1,2,3])
			self.assertRaises(TypeError, self.lsc.addLabelset, ['a'], 'Labelset3', ['Description'])
			#make sure no new label added with exception
			self.assertEqual(self.lsc.labelIDlst, ['Labelset1', 'Labelset2'])
			#test add existing label ID --> IDExistsError
			self.assertRaises(IDExistsError, self.lsc.addLabelset, ['e', 'f'], 'Labelset1')
			#test label info specification --> Info default to 'NA' if not specified
			self.lsc.addLabelset(['e'], 'Labelset3')
			self.assertEqual(self.lsc._prop['Info'], ['Description1', 'Description2', 'NA'])
		with self.subTest(msg='Labelset loading checks'):
			#test input empty labelset
			self.lsc.addLabelset([], 'Labelset4')
			#check if labelset loaded correctly
			self.assertEqual(self.lsc._prop['Labelset'], [{'a', 'b', 'c'}, {'b', 'd'}, {'e'}, set()])
			#check if entity map setup correctly
			self.assertEqual(self.lsc.entity.map, {'a':0, 'b':1, 'c':2,'d':3, 'e':4})

	def test_popLabelset(self):
		with self.subTest(msg='Input checks'):
			#test wrong labelID type --> TypeError
			self.assertRaises(TypeError, self.lsc.popLabelset, 1)
			self.assertRaises(TypeError, self.lsc.popLabelset, ['Labelset1'])
			#test not exist labelID --> IDNotExistError
			self.assertRaises(IDNotExistError, self.lsc.popLabelset, 'Labelset3')
			#make sure nothing poped
			self.assertEqual(self.lsc.lst, ['Labelset1', 'Labelset2'])
		#make sure enties that are no longer in any labelset are popped
		self.lsc.popLabelset('Labelset1')
		self.assertEqual(self.lsc.labelIDlst, ['Labelset2'])
		self.assertEqual(self.lsc.entity.map, {'b':0, 'd':1})
		self.lsc.popLabelset('Labelset2')
		self.assertEqual(self.lsc.labelIDlst, [])
		self.assertEqual(self.lsc.entity.map, {})

	def test_updateLabelset(self):
		with self.subTest(msg='Input checks'):
			#test lst input, only list of string allowed
			self.assertRaises(TypeError, self.lsc.updateLabelset, 1, 'Labelset1')
			self.assertRaises(TypeError, self.lsc.updateLabelset, ['1', 2], 'Labelset1')
			self.assertRaises(TypeError, self.lsc.updateLabelset, '123', 'Labelset1')
			#test labelID input type
			self.assertRaises(TypeError, self.lsc.updateLabelset, ['a'], 123)
			self.assertRaises(TypeError, self.lsc.updateLabelset, ['a'], [1,2])
			self.assertRaises(TypeError, self.lsc.updateLabelset, ['a'], ['Labelset1'])
			#test reset not exist labelID --> IDNotExistError
			self.assertRaises(IDNotExistError, self.lsc.updateLabelset, ['a'], 'Labelset3')
		#test update nothing --> labelset stays the same
		self.lsc.updateLabelset([], 'Labelset1')
		self.assertEqual(self.lsc.getLabelset('Labelset1'), {'a', 'b', 'c'})
		#test update existing --> labelset stays the same
		self.lsc.updateLabelset(['a', 'b', 'c'], 'Labelset1')
		self.assertEqual(self.lsc.getLabelset('Labelset1'), {'a', 'b', 'c'})
		#test update partially new
		self.lsc.updateLabelset(['a', 'd'], 'Labelset1')
		self.assertEqual(self.lsc.getLabelset('Labelset1'), {'a', 'b', 'c', 'd'})
		#test update all new
		self.lsc.updateLabelset(['e'], 'Labelset1')
		self.assertEqual(self.lsc.getLabelset('Labelset1'), {'a', 'b', 'c', 'd', 'e'})
		#check if new entity added to list
		self.assertEqual(self.lsc.entity.map, {'a':0, 'b':1, 'c':2, 'd':3, 'e':4})

	def test_resetLabelset(self):
		self.template_test_input_for_getters(self.lsc.resetLabelset)
		#check if labelset reset to empty set correctly
		self.lsc.resetLabelset('Labelset1')
		self.assertEqual(self.lsc.getLabelset('Labelset1'), set())
		self.assertEqual(self.lsc.getLabelset('Labelset2'), {'b', 'd'})
		#makesure list of labelsets untouched
		self.assertEqual(self.lsc.labelIDlst, ['Labelset1', 'Labelset2'])
		#make sure entities that are nolongler in any labelset are popped
		self.assertEqual(self.lsc.entity.map, {'b':0, 'd':1})

if __name__ == '__main__':
	unittest.main()