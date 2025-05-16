#pragma once
#include "IndexBase.h"

namespace zenann {

/// Simple KD-Tree for exact k-NN baseline with persistence
class KDTreeIndex : public IndexBase {
public:
    explicit KDTreeIndex(size_t dim);
    ~KDTreeIndex() override;

    /// Build tree structure from data
    void train() override;

    /// k-NN search
    SearchResult search(const Vector& query, size_t k) const override;

    /// Persist index to binary file
    void write_index(const std::string& filename) const;
    /// Load index from binary file
    static std::shared_ptr<KDTreeIndex> read_index(const std::string& filename);

private:
    struct Node {
        size_t      idx;   ///< index in original dataset
        size_t      dim;   ///< splitting dimension
        float       val;   ///< splitting value
        Node*       left;
        Node*       right;
        Node(size_t i, size_t d, float v) : idx(i), dim(d), val(v), left(nullptr), right(nullptr) {}
    };

    Node* buildTree(std::vector<size_t>& idxs, int depth);
    void  freeTree(Node* node);
    void  knnSearch(Node* node, const Vector& q, int k, std::vector<std::pair<float,size_t>>& heap) const;

    Node* root_ = nullptr;
};

} // namespace zenann