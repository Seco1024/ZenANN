#include "KDTreeIndex.h"
#include <algorithm>
#include <cmath>
#include <fstream>

namespace zenann {

KDTreeIndex::KDTreeIndex(size_t dim)
    : IndexBase(dim), root_(nullptr) {}

KDTreeIndex::~KDTreeIndex() {
    freeTree(root_);
}

void KDTreeIndex::freeTree(Node* node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}

KDTreeIndex::Node* KDTreeIndex::buildTree(std::vector<size_t>& idxs, int depth) {
    if (idxs.empty()) return nullptr;
    size_t axis = depth % dimension_;
    auto& data = datastore_->getAll();
    auto comp = [&](size_t a, size_t b){ return data[a][axis] < data[b][axis]; };
    size_t mid = idxs.size() / 2;
    std::nth_element(idxs.begin(), idxs.begin()+mid, idxs.end(), comp);
    size_t pivot = idxs[mid];
    Node* node = new Node(pivot, axis, data[pivot][axis]);
    std::vector<size_t> left_ids(idxs.begin(), idxs.begin()+mid);
    node->left = buildTree(left_ids, depth+1);
    std::vector<size_t> right_ids(idxs.begin()+mid+1, idxs.end());
    node->right = buildTree(right_ids, depth+1);
    return node;
}

void KDTreeIndex::train() {
    const auto& data = datastore_->getAll();
    std::vector<size_t> idxs(data.size());
    for (size_t i = 0; i < data.size(); ++i) idxs[i] = i;
    root_ = buildTree(idxs, 0);
}

void KDTreeIndex::knnSearch(Node* node, const Vector& q, int k, std::vector<std::pair<float,size_t>>& heap) const {
    if (!node) return;
    const auto& pt = datastore_->getAll()[node->idx];
    float dist = 0;
    for (size_t i = 0; i < dimension_; ++i) {
        float d = q[i] - pt[i]; dist += d*d;
    }
    if ((int)heap.size() < k) {
        heap.emplace_back(dist, node->idx);
        if ((int)heap.size() == k) std::make_heap(heap.begin(), heap.end());
    } else if (dist < heap.front().first) {
        std::pop_heap(heap.begin(), heap.end()); heap.back() = {dist, node->idx}; std::push_heap(heap.begin(), heap.end());
    }
    Node* first = q[node->dim] < node->val ? node->left : node->right;
    Node* second = (first == node->left) ? node->right : node->left;
    knnSearch(first, q, k, heap);
    float diff = q[node->dim] - node->val;
    if ((int)heap.size() < k || diff*diff < heap.front().first) {
        knnSearch(second, q, k, heap);
    }
}

SearchResult KDTreeIndex::search(const Vector& query, size_t k) const {
    SearchResult result;
    std::vector<std::pair<float,size_t>> heap;
    heap.reserve(k);
    knnSearch(root_, query, k, heap);
    std::sort(heap.begin(), heap.end(), [](auto&a, auto&b){ return a.first < b.first; });
    result.distances.resize(heap.size()); result.indices.resize(heap.size());
    for (size_t i = 0; i < heap.size(); ++i) {
        result.distances[i] = heap[i].first;
        result.indices[i]   = heap[i].second;
    }
    return result;
}

void KDTreeIndex::write_index(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    // write dimension
    out.write(reinterpret_cast<const char*>(&dimension_), sizeof(dimension_));
    // write raw data
    const auto& data = datastore_->getAll();
    size_t N = data.size(); out.write(reinterpret_cast<const char*>(&N), sizeof(N));
    for (const auto& v : data) {
        out.write(reinterpret_cast<const char*>(v.data()), sizeof(float)*v.size());
    }
    // TODO: optionally serialize tree structure or rebuild from data when reading
}

std::shared_ptr<KDTreeIndex> KDTreeIndex::read_index(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    size_t dim, N;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&N), sizeof(N));
    Dataset data(N, Vector(dim));
    for (auto& v : data) {
        in.read(reinterpret_cast<char*>(v.data()), sizeof(float)*dim);
    }
    auto idx = std::make_shared<KDTreeIndex>(dim);
    idx->datastore_->add(data);
    idx->train();
    return idx;
}

} // namespace zenann