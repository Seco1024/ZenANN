#include "IVFFlatIndex.h"
#include <limits>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace zenann {

IVFFlatIndex::IVFFlatIndex(size_t dim, size_t nlist, size_t nprobe)
    : IndexBase(dim), nlist_(nlist), nprobe_(nprobe) {}

IVFFlatIndex::~IVFFlatIndex() = default;

void IVFFlatIndex::train() {
    const auto& data = datastore_->getAll();
    if (data.empty()) return;

    // Calculate centroids with K-means algorithm
    kmeans(data);

    // Construct inverted list structure
    lists_.assign(nlist_, idList());
    for (size_t id = 0; id < data.size(); ++id) {
        const auto& v = data[id];
        float best_dist = std::numeric_limits<float>::max();
        size_t best_c = 0;
        for (size_t c = 0; c < nlist_; ++c) {
            float d = 0.0f;
            for (size_t k = 0; k < dimension_; ++k) {
                float diff = v[k] - centroids_[c][k];
                d += diff * diff;
            }
            if (d < best_dist) {
                best_dist = d;
                best_c = c;
            }
        }
        lists_[best_c].push_back(id);
    }
}

SearchResult IVFFlatIndex::search(const Vector& query, size_t k) const {
    SearchResult result;

    // Calculate query <-> centroids 
    std::vector<std::pair<float, size_t>> cdist;
    cdist.reserve(nlist_);
    for (size_t c = 0; c < nlist_; ++c) {
        float d = 0.0f;
        for (size_t i = 0; i < dimension_; ++i) {
            float diff = query[i] - centroids_[c][i];
            d += diff * diff;
        }
        cdist.emplace_back(d, c);
    }
    // Sorting
    std::sort(cdist.begin(), cdist.end(), [](auto& a, auto& b){ return a.first < b.first; });

    // Top-k brute-force search
    using Pair = std::pair<float, size_t>;
    std::vector<Pair> heap;
    heap.reserve(k);

    const auto& data = datastore_->getAll();
    for (size_t i = 0; i < nprobe_; ++i) {
        size_t c = cdist[i].second;
        for (auto id : lists_[c]) {
            const auto& v = data[id];
            float dist = 0.0f;
            for (size_t j = 0; j < dimension_; ++j) {
                float diff = query[j] - v[j];
                dist += diff * diff;
            }
            if (heap.size() < k) {
                heap.emplace_back(dist, id);
                if (heap.size() == k) std::make_heap(heap.begin(), heap.end());
            } else if (dist < heap.front().first) {
                std::pop_heap(heap.begin(), heap.end());
                heap.back() = {dist, id};
                std::push_heap(heap.begin(), heap.end());
            }
        }
    }

    std::sort(heap.begin(), heap.end(), [](const Pair &a, const Pair &b){ return a.first < b.first; });

    result.distances.resize(heap.size());
    result.indices.resize(heap.size());
    for (size_t i = 0; i < heap.size(); ++i) {
        result.distances[i] = heap[i].first;
        result.indices[i] = heap[i].second;
    }
    return result;
}

std::vector<SearchResult> IVFFlatIndex::search_batch(const Dataset& queries, size_t k) const {
    std::vector<SearchResult> results;
    results.reserve(queries.size());
    for (const auto& q : queries) {
        results.emplace_back(search(q, k));
    }
    return results;
}

void IVFFlatIndex::kmeans(const Dataset& data, size_t iterations) {
    size_t n = data.size();
    std::mt19937 rng(123);
    std::uniform_int_distribution<size_t> dist(0, n-1);

    // Initialized with random centroids
    centroids_.clear();
    centroids_.reserve(nlist_);
    for (size_t i = 0; i < nlist_; ++i) {
        centroids_.push_back(data[dist(rng)]);
    }

    std::vector<size_t> assignment(n);
    for (size_t it = 0; it < iterations; ++it) {

        // Assignment step (E)
        for (size_t i = 0; i < n; ++i) {
            const auto& v = data[i];
            float best = std::numeric_limits<float>::max();
            size_t best_c = 0;
            for (size_t c = 0; c < nlist_; ++c) {
                float d = 0.0f;
                for (size_t k = 0; k < dimension_; ++k) {
                    float diff = v[k] - centroids_[c][k];
                    d += diff * diff;
                }
                if (d < best) {
                    best = d;
                    best_c = c;
                }
            }
            assignment[i] = best_c;
        }

        // Update Step (M)
        std::vector<Vector> sums(nlist_, Vector(dimension_, 0.0f));
        std::vector<size_t> counts(nlist_, 0);
        for (size_t i = 0; i < n; ++i) {
            size_t c = assignment[i];
            for (size_t k = 0; k < dimension_; ++k) {
                sums[c][k] += data[i][k];
            }
            counts[c]++;
        }
        for (size_t c = 0; c < nlist_; ++c) {
            if (counts[c] > 0) {
                for (size_t k = 0; k < dimension_; ++k) {
                    sums[c][k] /= counts[c];
                }
                centroids_[c].swap(sums[c]);
            }
        }
    }
}

} 
