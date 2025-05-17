#include "HNSWIndex.h"
#include <faiss/index_io.h>
#include <vector>
#include <algorithm>
#include <fstream>

namespace zenann {

HNSWIndex::HNSWIndex(size_t dim, size_t M, size_t efConstruction)
    : IndexBase(dim), idx_(new faiss::IndexHNSWFlat(dim, M)) {
    idx_->hnsw.efConstruction = efConstruction;
}

HNSWIndex::~HNSWIndex() = default;

void HNSWIndex::train() {
    const auto& data = datastore_->getAll();
    if (data.empty()) return;
    size_t n = data.size();
    std::vector<float> flat(n * dimension_);
    for (size_t i = 0; i < n; ++i) {
        std::copy(data[i].begin(), data[i].end(), flat.begin() + i * dimension_);
    }
    idx_->add(n, flat.data());
}

SearchResult HNSWIndex::search(const Vector& query, size_t k) const {
    std::vector<faiss::idx_t> labels(k);
    std::vector<float> distances(k);
    idx_->search(1, query.data(), k, distances.data(), labels.data());

    SearchResult result;
    result.indices.assign(labels.begin(), labels.end());
    result.distances.assign(distances.begin(), distances.end());
    return result;
}

SearchResult HNSWIndex::search(const Vector& query, size_t k, size_t efSearch) const {
    idx_->hnsw.efSearch = efSearch;
    return search(query, k);
}

std::vector<SearchResult> HNSWIndex::search_batch(const Dataset& queries, size_t k) const {
    return search_batch(queries, k, idx_->hnsw.efSearch);
}

std::vector<SearchResult> HNSWIndex::search_batch(const Dataset& queries, size_t k, size_t efSearch) const {
    idx_->hnsw.efSearch = efSearch;
    size_t nq = queries.size();
    std::vector<float> flat(nq * dimension_);
    for (size_t i = 0; i < nq; ++i) {
        std::copy(queries[i].begin(), queries[i].end(), flat.begin() + i * dimension_);
    }
    std::vector<faiss::idx_t> labels(nq * k);
    std::vector<float> distances(nq * k);
    idx_->search(nq, flat.data(), k, distances.data(), labels.data());

    std::vector<SearchResult> results(nq);
    for (size_t i = 0; i < nq; ++i) {
        auto lb = labels.begin() + i * k;
        auto dd = distances.begin() + i * k;
        results[i].indices.assign(lb, lb + k);
        results[i].distances.assign(dd, dd + k);
    }
    return results;
}

void HNSWIndex::set_ef_search(size_t efSearch) {
    idx_->hnsw.efSearch = efSearch;
}

void HNSWIndex::reorder_layout() {
    idx_->bfs_reorder();
}

void HNSWIndex::reorder_layout(const std::string& mapping_file) {
    auto new_order = idx_->bfs_reorder();
    std::ofstream ofs(mapping_file, std::ios::binary);
    size_t sz = new_order.size();
    ofs.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    ofs.write(reinterpret_cast<const char*>(new_order.data()), sz * sizeof(int));
}


void HNSWIndex::write_index(const std::string& filename) const {
    faiss::write_index(idx_.get(), filename.c_str());
}

std::shared_ptr<HNSWIndex> HNSWIndex::read_index(const std::string& filename) {
    faiss::Index* base = faiss::read_index(filename.c_str());
    auto raw = dynamic_cast<faiss::IndexHNSWFlat*>(base);
    auto inst = std::make_shared<HNSWIndex>(raw->d, raw->hnsw.nb_neighbors(0));
    inst->idx_.reset(raw);
    return inst;
}

double HNSWIndex::compute_recall_with_mapping(
    const std::vector<std::vector<faiss::idx_t>>& groundtruth,
    const std::vector<faiss::idx_t>& predicted_flat,
    size_t nq, size_t k,
    const std::string& mapping_file) {
    std::ifstream ifs(mapping_file, std::ios::binary);
    size_t sz;
    ifs.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    std::vector<int> old_to_new(sz);
    ifs.read(reinterpret_cast<char*>(old_to_new.data()), sz * sizeof(int));
    std::vector<std::vector<faiss::idx_t>> pred(nq, std::vector<faiss::idx_t>(k));
    for (size_t i = 0; i < nq; ++i)
      for (size_t j = 0; j < k; ++j)
        pred[i][j] = predicted_flat[i*k + j];
    std::vector<double> recalls;
    for (size_t i = 0; i < nq; ++i) {
        std::set<faiss::idx_t> ts;
        for (size_t j = 0; j < k; ++j) {
            auto oldid = groundtruth[i][j];
            ts.insert(old_to_new[oldid]);
        }
        std::set<faiss::idx_t> ps(pred[i].begin(), pred[i].end());
        size_t hits=0;
        for (auto pid: ps) if (ts.count(pid)) ++hits;
        recalls.push_back(double(hits)/ts.size());
    }
    return std::accumulate(recalls.begin(), recalls.end(), 0.0)/recalls.size();
}

} 
