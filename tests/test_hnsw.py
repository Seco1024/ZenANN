import unittest
import sys
sys.path.append('..')
sys.path.append("build") 
import numpy as np
from zenann import HNSWIndex

class TestHNSWRecall(unittest.TestCase):
    def setUp(self):
        # one-hot vectors
        self.dim = 8
        self.data = [ [1.0 if i==j else 0.0 for i in range(self.dim)] for j in range(self.dim) ]
        self.M = self.dim
        self.efC = 16
        self.efS = self.dim
        self.index = HNSWIndex(dim=self.dim, M=self.M, efConstruction=self.efC)
        self.index.set_ef_search(self.efS)
        self.index.build(self.data)

    def test_perfect_recall_k1(self):
        for true_id, q in enumerate(self.data):
            res = self.index.search(q, k=1)
            self.assertEqual(len(res.indices), 1)
            self.assertEqual(res.indices[0], true_id)
            self.assertAlmostEqual(res.distances[0], 0.0, places=6)

    def test_recall_rate_k2(self):
        hits = 0
        for true_id, q in enumerate(self.data):
            res = self.index.search(q, k=2)
            self.assertGreaterEqual(len(res.indices), 1)
            if true_id in res.indices:
                hits += 1
        recall = hits / len(self.data)
        self.assertEqual(recall, 1.0)

if __name__ == '__main__':
    unittest.main()
