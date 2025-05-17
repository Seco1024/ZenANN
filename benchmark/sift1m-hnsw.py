# scripts/hnsw_benchmark.py
import sys
import os
import time
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', 'build')))
from zenann import HNSWIndex

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
    base = load_fvecs(args.base)
    queries = load_fvecs(args.query)
    gt = load_ivecs(args.groundtruth)
    print(f"Base: {base.shape}, Queries: {queries.shape}, GT: {gt.shape}")

    if args.index_file and os.path.exists(args.index_file):
        print("Loading index...")
        idx = HNSWIndex.read_index(args.index_file)
    else:
        print("Building HNSW index...")
        idx = HNSWIndex(dim=base.shape[1], M=args.M, efConstruction=args.efConstruction)
        t0 = time.time()
        idx.build(base)
        t_build = time.time() - t0
        print(f"Build time: {t_build:.3f}s")
        if args.index_file:
            idx.write_index(args.index_file)

    idx.set_ef_search(args.efSearch)

    print("Searching...")
    K = args.k
    mapping_file = args.mapping_file or "bfs_mapping.bin"
    idx.reorder_layout(mapping_file)

    t0 = time.time()
    all_res = idx.search_batch(queries.tolist(), K, args.efSearch)
    t_search = time.time() - t0
    qps = len(queries) / t_search

    predicted_flat = []
    for r in all_res:
        padded = r.indices + [-1] * (K - len(r.indices))
        predicted_flat.extend(padded[:K])

    recall = idx.compute_recall_with_mapping(
        gt.tolist(), predicted_flat, len(queries), K, mapping_file
    )
    
    print(f"Recall@{K}: {recall * 100:.2f}%")
    print(f"QPS: {qps:.2f}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--index_file", default=None)
    p.add_argument("--M", type=int, default=32)
    p.add_argument("--efConstruction", type=int, default=200)
    p.add_argument("--efSearch", type=int, default=32)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--mapping_file", default=None, help="Where to save mapping file")
    args = p.parse_args()
    main(args)
