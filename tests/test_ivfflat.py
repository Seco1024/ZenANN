import sys
import unittest

# allow import of the built extension
sys.path.append('..')
sys.path.append('build')

from zenann import IVFFlatIndex, SearchResult

class TestIVFBasic(unittest.TestCase):
    def setUp(self):
        # one-hot vectors in 4-d for simple testing
        self.dim = 4
        self.data = []
        for i in range(self.dim):
            v = [0.0] * self.dim
            v[i] = 1.0
            self.data.append(v)

    def test_search_sorted(self):
        # Test that search returns distances sorted in ascending order
        idx = IVFFlatIndex(dim=self.dim, nlist=self.dim, nprobe=self.dim)
        idx.build(self.data)
        result = idx.search(self.data[0], k=3)
        # distances should be non-decreasing
        dists = result.distances
        self.assertTrue(all(dists[i] <= dists[i+1] for i in range(len(dists)-1)))

    def test_search_batch(self):
        idx = IVFFlatIndex(dim=self.dim, nlist=self.dim, nprobe=2)
        idx.build(self.data)
        # batch of two queries
        batch = [self.data[0], self.data[1]]
        results = idx.search_batch(batch, k=2)
        self.assertEqual(len(results), 2)
        for res, query in zip(results, batch):
            self.assertIsInstance(res, SearchResult)
            # verify the true index is in results
            true_id = self.data.index(query)
            self.assertIn(true_id, res.indices)

    def test_nprobe_effect(self):
        # nprobe=1 returns only 1 result
        idx1 = IVFFlatIndex(dim=self.dim, nlist=self.dim, nprobe=1)
        idx1.build(self.data)
        res1 = idx1.search(self.data[2], k=2)
        self.assertEqual(len(res1.indices), 1)
        # nprobe=2 can return up to 2
        idx2 = IVFFlatIndex(dim=self.dim, nlist=self.dim, nprobe=2)
        idx2.build(self.data)
        res2 = idx2.search(self.data[2], k=2)
        self.assertTrue(1 <= len(res2.indices) <= 2)

if __name__ == '__main__':
    unittest.main()

