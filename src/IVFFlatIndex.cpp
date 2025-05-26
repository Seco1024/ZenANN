#include "IVFFlatIndex.h"
#include "SimdUtils.h"
#include <omp.h>
#include <limits>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>

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
    using Pair = std::pair<float, size_t>;
    std::vector<Pair> cdist(nlist_);
    std::vector<Pair> heap;

    // calculate distance from centroids
#pragma omp parallel for schedule(static)
    for (size_t c = 0; c < nlist_; ++c) {
        float d = l2_simd(query.data(), centroids_[c].data(), dimension_);
        cdist[c] = {d, c};
    }
    std::partial_sort(cdist.begin(), cdist.begin() + nprobe_, cdist.end(), 
        [](auto& a, auto& b) { 
            return a.first < b.first; 
        }
    );

    // probe nprobe lists
    heap.reserve(k);
    const auto& data = datastore_->getAll();

#pragma omp parallel for schedule(dynamic)
    for (size_t pi = 0; pi < nprobe_; ++pi) {
        size_t c = cdist[pi].second;
        std::vector<Pair> local;
        local.reserve(k);

        for (size_t id : lists_[c]) {
            float dist = l2_simd(query.data(), data[id].data(), dimension_);

            if (local.size() < k) {
                local.emplace_back(dist, id);
                if (local.size() == k)
                    std::make_heap(local.begin(), local.end());
            } else if (dist < local.front().first) {
                std::pop_heap(local.begin(), local.end());
                local.back() = {dist, id};
                std::push_heap(local.begin(), local.end());
            }
        }

#pragma omp critical
{
            for (auto& p : local) {
                if (heap.size() < k) {
                    heap.emplace_back(p);
                    if (heap.size() == k)
                        std::make_heap(heap.begin(), heap.end());
                } else if (p.first < heap.front().first) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.back() = p;
                    std::push_heap(heap.begin(), heap.end());
                }
            }
        }
    }

    std::sort(heap.begin(), heap.end(),
              [](const Pair& a, const Pair& b) { return a.first < b.first; });

    SearchResult res;
    res.distances.resize(heap.size());
    res.indices.resize(heap.size());
    for (size_t i = 0; i < heap.size(); ++i) {
        res.distances[i] = heap[i].first;
        res.indices[i]   = heap[i].second;
    }
    return res;
}

SearchResult IVFFlatIndex::search(const Vector& query, size_t k, size_t nprobe) const {
    size_t old = nprobe_;
    const_cast<IVFFlatIndex*>(this)->nprobe_ = nprobe;
    SearchResult res = search(query, k);
    const_cast<IVFFlatIndex*>(this)->nprobe_ = old;
    return res;
}

std::vector<SearchResult> IVFFlatIndex::search_batch(const Dataset& queries, size_t k) const {
    const size_t nq = queries.size();
    std::vector<SearchResult> results(nq);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < nq; ++i) {
        results[i] = search(queries[i], k);
    }

    return results;            
}

std::vector<SearchResult> IVFFlatIndex::search_batch(const Dataset& queries, size_t k, size_t nprobe) const {
    size_t old = nprobe_;
    const_cast<IVFFlatIndex*>(this)->nprobe_ = nprobe;
    auto res = search_batch(queries, k);
    const_cast<IVFFlatIndex*>(this)->nprobe_ = old;
    return res;
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

void IVFFlatIndex::write_index(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    // Basic params
    out.write(reinterpret_cast<const char*>(&dimension_), sizeof(dimension_));
    out.write(reinterpret_cast<const char*>(&nlist_), sizeof(nlist_));
    out.write(reinterpret_cast<const char*>(&nprobe_), sizeof(nprobe_));
    
    // Write raw data
    const auto& data = datastore_->getAll();
    size_t N = data.size();
    out.write(reinterpret_cast<const char*>(&N), sizeof(N));
    for (const auto& v : data) {
        out.write(reinterpret_cast<const char*>(v.data()), sizeof(float)*v.size());
    }
    
    // Write centroids
    size_t C = centroids_.size();
    out.write(reinterpret_cast<const char*>(&C), sizeof(C));
    for (const auto& cvec : centroids_) {
        out.write(reinterpret_cast<const char*>(cvec.data()), sizeof(float)*cvec.size());
    }
    
    // Write inverted lists
    size_t L = lists_.size();
    out.write(reinterpret_cast<const char*>(&L), sizeof(L));
    for (const auto& lst : lists_) {
        size_t sz = lst.size();
        out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
        out.write(reinterpret_cast<const char*>(lst.data()), sizeof(size_t)*sz);
    }
}

std::shared_ptr<IVFFlatIndex> IVFFlatIndex::read_index(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    // Read basic params
    size_t dim, nlist, nprobe;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&nlist), sizeof(nlist));
    in.read(reinterpret_cast<char*>(&nprobe), sizeof(nprobe));

    auto idx = std::make_shared<IVFFlatIndex>(dim, nlist, nprobe);
    
    // Read raw data
    size_t N;
    in.read(reinterpret_cast<char*>(&N), sizeof(N));
    Dataset data(N, Vector(dim));
    for (auto& v : data) {
        in.read(reinterpret_cast<char*>(v.data()), sizeof(float)*dim);
    }
    idx->datastore_->add(data);
    
    // Read centroids
    size_t C;
    in.read(reinterpret_cast<char*>(&C), sizeof(C));
    idx->centroids_.resize(C, Vector(dim));
    for (auto& cvec : idx->centroids_) {
        in.read(reinterpret_cast<char*>(cvec.data()), sizeof(float)*dim);
    }
    
    // Read inverted lists
    size_t L;
    in.read(reinterpret_cast<char*>(&L), sizeof(L));
    idx->lists_.resize(L);
    for (auto& lst : idx->lists_) {
        size_t sz;
        in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        lst.resize(sz);
        in.read(reinterpret_cast<char*>(lst.data()), sizeof(size_t)*sz);
    }
    
    return idx;
}
} 
