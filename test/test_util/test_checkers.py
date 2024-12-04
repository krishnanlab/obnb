import unittest

import numpy as np

from obnb.alltypes import FLOAT_TYPE, INT_TYPE, Iterable
from obnb.util import checkers


class TestCheckers(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        n = 10
        self.n = n
        self.n_str = str(n)
        self.n_int = int(n)
        self.n_npint = int(n)
        self.n_npint64 = np.int64(n)
        self.n_float = float(n)
        self.n_npfloat = float(n)
        self.n_int_tuple = (int(n), int(n), int(n))
        self.n_int_lst = [int(n), int(n), int(n)]
        self.n_int_npary = np.array([n, n, n], dtype=int)
        self.n_float_tuple = (float(n), float(n), float(n))
        self.n_float_lst = [float(n), float(n), float(n)]
        self.n_float_npary = np.array([n, n, n], dtype=float)

    def test_INT_TYPE(self):
        self.assertIsInstance(self.n_int, INT_TYPE)
        self.assertIsInstance(self.n_npint, INT_TYPE)
        self.assertIsInstance(self.n_npint64, INT_TYPE)
        self.assertNotIsInstance(self.n_float, INT_TYPE)
        self.assertNotIsInstance(self.n_npfloat, INT_TYPE)

    def test_FLOAT_TYPE(self):
        self.assertNotIsInstance(self.n_int, FLOAT_TYPE)
        self.assertNotIsInstance(self.n_npint, FLOAT_TYPE)
        self.assertNotIsInstance(self.n_npint64, FLOAT_TYPE)
        self.assertIsInstance(self.n_float, FLOAT_TYPE)
        self.assertIsInstance(self.n_npfloat, FLOAT_TYPE)

    def test_ITERABLE_TYPE(self):
        n_int_tuple = (1, 2, 3)
        n_int_lst = [1, 2, 3]
        n_int_ary = np.array([1, 2, 3])
        self.assertIsInstance(n_int_tuple, Iterable)
        self.assertIsInstance(n_int_lst, Iterable)
        self.assertIsInstance(n_int_ary, Iterable)

    def test_checkType(self):
        checkers.checkType("n_int", int, self.n_int)
        self.assertRaises(
            TypeError,
            checkers.checkType,
            "n_int",
            int,
            self.n_float,
        )

        checkers.checkType("n_float", float, self.n_float)
        self.assertRaises(
            TypeError,
            checkers.checkType,
            "n_float",
            float,
            self.n_int,
        )
        checkers.checkType("n_str", str, self.n_str)

        self.assertRaises(ValueError, checkers.checkType, "n", int, None)
        self.assertRaises(
            ValueError,
            checkers.checkType,
            "n",
            float,
            None,
        )
        self.assertRaises(ValueError, checkers.checkType, "n", str, None)

    def test_checkTypeInIterable(self):
        checkers.checkTypesInIterable("n_int_tuple", INT_TYPE, self.n_int_tuple)
        checkers.checkTypesInIterable("n_int_lst", INT_TYPE, self.n_int_lst)
        checkers.checkTypesInIterable("n_int_npary", INT_TYPE, self.n_int_npary)
        self.assertRaises(
            TypeError,
            checkers.checkTypesInIterable,
            "n_int_tuple",
            FLOAT_TYPE,
            self.n_int_tuple,
        )
        self.assertRaises(
            TypeError,
            checkers.checkTypesInIterable,
            "n_int_lst",
            FLOAT_TYPE,
            self.n_int_lst,
        )
        self.assertRaises(
            TypeError,
            checkers.checkTypesInIterable,
            "n_int_npary",
            FLOAT_TYPE,
            self.n_int_npary,
        )
        checkers.checkTypesInIterable(
            "n_float_tuple",
            FLOAT_TYPE,
            self.n_float_tuple,
        )
        checkers.checkTypesInIterable(
            "n_float_lst",
            FLOAT_TYPE,
            self.n_float_lst,
        )
        checkers.checkTypesInIterable(
            "n_float_npary",
            FLOAT_TYPE,
            self.n_float_npary,
        )
        self.assertRaises(
            TypeError,
            checkers.checkTypesInIterable,
            "n_float_tuple",
            INT_TYPE,
            self.n_float_tuple,
        )
        self.assertRaises(
            TypeError,
            checkers.checkTypesInIterable,
            "n_float_lst",
            INT_TYPE,
            self.n_float_lst,
        )
        self.assertRaises(
            TypeError,
            checkers.checkTypesInIterable,
            "n_float_npary",
            INT_TYPE,
            self.n_float_npary,
        )

    def test_checkNullableType(self):
        checkers.checkNullableType("n", int, None)
        checkers.checkNullableType("n", float, None)
        checkers.checkNullableType("n", str, None)

    def test_checkNumpyArrayNDim(self):
        ary1 = np.ones(2)
        ary2 = np.ones((2, 2))
        ary3 = np.ones((2, 2, 2))
        checkers.checkNumpyArrayNDim("ary1", 1, ary1)
        checkers.checkNumpyArrayNDim("ary2", 2, ary2)
        checkers.checkNumpyArrayNDim("ary3", 3, ary3)
        self.assertRaises(
            ValueError,
            checkers.checkNumpyArrayNDim,
            "ary1",
            4,
            ary1,
        )
        self.assertRaises(
            ValueError,
            checkers.checkNumpyArrayNDim,
            "ary2",
            4,
            ary2,
        )
        self.assertRaises(
            ValueError,
            checkers.checkNumpyArrayNDim,
            "ary3",
            4,
            ary3,
        )

    def test_checkNumpyArrayShape(self):
        ary1 = np.ones(2)
        ary2 = np.ones((2, 2))
        ary3 = np.ones((2, 2, 2))
        checkers.checkNumpyArrayShape("ary1", 2, ary1)
        checkers.checkNumpyArrayShape("ary1", (2,), ary1)
        checkers.checkNumpyArrayShape("ary2", (2, 2), ary2)
        checkers.checkNumpyArrayShape("ary3", (2, 2, 2), ary3)
        self.assertRaises(
            ValueError,
            checkers.checkNumpyArrayShape,
            "ary1",
            1,
            ary1,
        )
        self.assertRaises(
            ValueError,
            checkers.checkNumpyArrayShape,
            "ary1",
            (
                1,
                2,
            ),
            ary1,
        )
        self.assertRaises(
            ValueError,
            checkers.checkNumpyArrayShape,
            "ary2",
            (2, 1),
            ary2,
        )
        self.assertRaises(
            ValueError,
            checkers.checkNumpyArrayShape,
            "ary3",
            1,
            ary3,
        )

    def test_checkNumpyArrayIsNumeric(self):
        # test numpy numeric array --> success
        ary1 = np.random.random(3).astype(int)
        ary2 = np.random.random((3, 3)).astype(float)
        ary3 = np.random.random(5)
        checkers.checkNumpyArrayIsNumeric("ar1", ary1)
        checkers.checkNumpyArrayIsNumeric("ar2", ary2)
        checkers.checkNumpyArrayIsNumeric("ar3", ary3)

        # test numpy string array --> error
        ary1 = np.ones(3, dtype=str)
        ary2 = np.ones((3, 3), dtype=str)
        self.assertRaises(
            TypeError,
            checkers.checkNumpyArrayIsNumeric,
            "ar1",
            ary1,
        )
        self.assertRaises(
            TypeError,
            checkers.checkNumpyArrayIsNumeric,
            "ar2",
            ary2,
        )

        # test list --> error
        lst = [1, 2, 3]
        self.assertRaises(
            TypeError,
            checkers.checkNumpyArrayIsNumeric,
            "lst",
            lst,
        )

        # test string --> error
        string = "asdfgasdglafknklasd"
        self.assertRaises(
            TypeError,
            checkers.checkNumpyArrayIsNumeric,
            "string",
            string,
        )


if __name__ == "__main__":
    unittest.main()
