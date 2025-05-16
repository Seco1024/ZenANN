import sys
import unittest
sys.path.append('build')

from zenann import KDTreeIndex


class TestKDTreeBasic(unittest.TestCase):
    def setUp(self):
        self.dim = 4
        self.data = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        self.index = KDTreeIndex(self.dim)
        self.index.build(self.data)

    def test_knn_k1(self):
        query = [1.0, 0.0, 0.0, 0.0]
        result = self.index.search(query, k=1)
        self.assertEqual(len(result.indices), 1)
        self.assertEqual(result.indices[0], 0)
        self.assertAlmostEqual(result.distances[0], 0.0, places=6)

    def test_knn_k2(self):
        query = [0.9, 0.0, 0.0, 0.0]
        result = self.index.search(query, k=2)
        self.assertEqual(len(result.indices), 2)
        self.assertIn(0, result.indices)
        self.assertLessEqual(result.distances[0], result.distances[1])

if __name__ == "__main__":
    unittest.main()
