#pragma once
#include "IndexBase.h"
#include <faiss/IndexHNSW.h>
#include <string>
#include <vector>

namespace zenann {

class HNSWIndex : public IndexBase {
public:
    HNSWIndex(size_t dim, size_t M, size_t efConstruction = 200);
    ~HNSWIndex() override;
    void train() override;
    SearchResult search(const Vector& query, size_t k) const override;
    SearchResult search(const Vector& query, size_t k, size_t efSearch) const;
    std::vector<SearchResult> search_batch(const Dataset& queries, size_t k) const;
    std::vector<SearchResult> search_batch(const Dataset& queries, size_t k, size_t efSearch) const;

    void set_ef_search(size_t efSearch);
    void reorder_layout();
    void reorder_layout(const std::string& mapping_file);

    void write_index(const std::string& filename) const;
    static std::shared_ptr<HNSWIndex> read_index(const std::string& filename);

    static double compute_recall_with_mapping(
        const std::vector<std::vector<faiss::idx_t>>& groundtruth,
        const std::vector<faiss::idx_t>& predicted_flat,
        size_t nq, size_t k,
        const std::string& mapping_file);

private:
    std::unique_ptr<faiss::IndexHNSWFlat> idx_;
};

} 
