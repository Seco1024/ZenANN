import sys
import os
import time
import argparse
import numpy as np

# allow import of the built extension
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', 'build')))
from zenann import IVFFlatIndex


def read_fvecs(fname):
    """
    Read FLANN .fvecs file: each vector stored as int dim, then float data
    """
    data = np.fromfile(fname, dtype='int32')
    dim = data[0]
    assert data.size % (dim + 1) == 0, "Invalid .fvecs file"
    vects = data.reshape(-1, dim+1)[:, 1:].astype('float32')
    return vects


def read_ivecs(fname):
    """
    Read FLANN .ivecs file: each entry stored as int dim, then int indices
    """
    data = np.fromfile(fname, dtype='int32')
    k = data[0]
    assert data.size % (k + 1) == 0, "Invalid .ivecs file"
    vects = data.reshape(-1, k+1)[:, 1:]
    return vects


def compute_recall(results, groundtruth, k):
    """
    Compute recall@k: fraction of queries where gt[0] in results
    """
    num_q = results.shape[0]
    hits = 0
    for i in range(num_q):
        if groundtruth[i, 0] in results[i, :k]:
            hits += 1
    return hits / num_q


def main(args):
    # Load dataset
    print("Loading dataset...")
    base = read_fvecs(args.base)
    queries = read_fvecs(args.query)
    gt = read_ivecs(args.groundtruth)

    print(f"Base vectors: {base.shape}")
    print(f"Queries: {queries.shape}")
    print(f"Groundtruth: {gt.shape}")

    # Build IVF index
    print("Building IVF index...")
    index = IVFFlatIndex(dim=base.shape[1], nlist=args.nlist, nprobe=args.nprobe)
    t0 = time.time()
    index.build(base)
    t_build = time.time() - t0
    print(f"Index build time: {t_build:.3f} s")

    # Search
    print("Running search...")
    K = args.k
    results = np.empty((queries.shape[0], K), dtype=np.int64)
    t0 = time.time()
    for i, q in enumerate(queries):
        res = index.search(q.tolist(), K)
        results[i, :len(res.indices)] = res.indices
    t_search = time.time() - t0
    qps = queries.shape[0] / t_search

    # Compute recall
    recall = compute_recall(results, gt, K)

    print(f"Recall@{K}: {recall * 100:.2f}%")
    print(f"Search QPS: {qps:.2f} queries/sec")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("ZenANN IVF SIFT1M benchmark")
    parser.add_argument("--base", required=True, help="Path to sift1M_base.fvecs")
    parser.add_argument("--query", required=True, help="Path to sift1M_query.fvecs")
    parser.add_argument("--groundtruth", required=True, help="Path to sift1M_groundtruth.ivecs")
    parser.add_argument("--nlist", type=int, default=1024, help="Number of IVF clusters")
    parser.add_argument("--nprobe", type=int, default=4, help="Number of clusters to probe")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors (k)")
    args = parser.parse_args()
    main(args)
