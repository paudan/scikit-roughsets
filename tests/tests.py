import unittest
import numpy as np
from scikit_roughsets.roughsets import RoughSetsReducer

class TestRoughsets(unittest.TestCase):

    red = RoughSetsReducer()
    S = np.array([[0, 0], [0, 0], [0, 0], [0, 1], [1, 1], [1, 1], [1, 1], [1, 2], [2, 2], [2, 2]])
    X = np.array([1, 2, 3, 4, 5])
    a = np.array([1, 2])

    D = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T
    C = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1],
                  [1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
                  [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
                  [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
                  [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1],
                  [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
                  [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1],
                  [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
                  [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1]])


    def test_indiscernibility(self):
        np.array_equal([[1]], self.red.indisc(self.a, self.X))

    def test_indiscernibility2(self):
        result = np.array([[ 1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
                          [ 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
                          [ 0, 0, 0, 0, 5, 6, 7, 0, 0, 0],
                          [ 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
                          [ 0, 0, 0, 0, 0, 0, 0, 0, 9, 10]])
        self.assertTrue(np.array_equal(result, self.red.indisc(self.a, self.S)))

    def test_rslower(self):
        self.assertListEqual([1, 2, 3, 4], self.red.rslower(self.X, self.a, self.S).tolist())

    def test_rsupper(self):
        self.assertListEqual([1, 2, 3, 4, 5, 6, 7], self.red.rsupper(self.X, self.a, self.S).tolist())

    def test_core(self):
        C = np.array([[1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 1],
                      [1, 1, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 1, 0, 1, 1],
                      [1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 0, 1, 1]])
        D = np.array([range(0, 10)]).T
        self.assertListEqual([1, 2, 5, 6, 7], self.red.core(C, D).tolist())

    def test_reduct(self):
        self.assertListEqual([], self.red.core(self.C, self.D).tolist())
        Y = self.red.reduce(self.C, self.D).tolist()
        self.assertListEqual([3, 4], Y)

    def test_scikit(self):
        from scikit_roughsets.rs_reduction import RoughSetsSelector
        selector = RoughSetsSelector()
        X_selected = selector.fit(self.C, self.D).transform(self.C)
        self.assertEqual(X_selected.shape[1], 2)