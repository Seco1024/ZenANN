import sys
import os
import time
import argparse
import numpy as np

# allow import of the built extension
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', 'build')))
from zenann import IVFFlatIndex


def load_fvecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    print(dim)
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def load_ivecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.int32)
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    fv = fv[:, 1:]
    return fv

def compute_recall(predicted, groundtruth, k=10):
    recalls = []
    for true_neighbors, pred_neighbors in zip(groundtruth, predicted):
        true_set = set(true_neighbors[:k])
        pred_set = set(pred_neighbors[:k])
        recall = len(true_set.intersection(pred_set)) / len(true_set)
        recalls.append(recall)
    return np.mean(recalls) 


def main(args):
    # Load dataset
    print("Loading dataset...")
    base = load_fvecs(args.base)
    queries = load_fvecs(args.query)
    gt = load_ivecs(args.groundtruth)

    print(f"Base vectors: {base.shape}")
    print(f"Queries: {queries.shape}")
    print(f"Groundtruth: {gt.shape}")

    # Initialize or load index
    if args.index_file and os.path.exists(args.index_file):
        print(f"Loading index from {args.index_file} ...")
        index = IVFFlatIndex.read_index(args.index_file)
        print("Index loaded.")
    else:
        print("Building IVF index...")
        index = IVFFlatIndex(dim=base.shape[1], nlist=args.nlist, nprobe=args.nprobe)
        t0 = time.time()
        index.build(base)
        t_build = time.time() - t0
        print(f"Index build time: {t_build:.3f} s")
        if args.index_file:
            print(f"Writing index to {args.index_file} ...")
            index.write_index(args.index_file)
            print("Index written.")

    # Batch search
    print("Running batch search...")
    K = args.k
    t0 = time.time()
    all_results = index.search_batch(queries.tolist(), K, args.nprobe)
    t_search = time.time() - t0
    qps = len(queries) / t_search

    # collect indices into numpy array
    results = np.zeros((len(all_results), K), dtype=np.int64)
    for i, res in enumerate(all_results):
        n = len(res.indices)
        results[i, :n] = res.indices

    # Compute recall
    recall = compute_recall(results, gt, K)

    print(f"Recall@{K}: {recall * 100:.2f}%")
    print(f"Search QPS: {qps:.2f} queries/sec")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("ZenANN IVF SIFT1M benchmark")
    parser.add_argument("--base", required=True, help="Path to sift1M_base.fvecs")
    parser.add_argument("--query", required=True, help="Path to sift1M_query.fvecs")
    parser.add_argument("--groundtruth", required=True, help="Path to sift1M_groundtruth.ivecs")
    parser.add_argument("--index_file", default=None, help="If provided and exists, load index; otherwise write index here")
    parser.add_argument("--nlist", type=int, default=1024, help="Number of IVF clusters")
    parser.add_argument("--nprobe", type=int, default=4, help="Number of clusters to probe")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors (k)")
    args = parser.parse_args()
    main(args)
