import sys
import unittest
sys.path.append('..')
sys.path.append("build") 

from zenann import IndexBase


class TestIndexBase(unittest.TestCase):
    def setUp(self):
        self.dim = 64
        self.index = IndexBase(dim=self.dim)
        self.data = [[0.1] * self.dim for _ in range(10)]

    def test_dimension(self):
        self.assertEqual(self.index.dimension, self.dim)

    def test_build_and_train(self):
        self.index.build(self.data)
        self.index.train()


if __name__ == '__main__':
    unittest.main()
