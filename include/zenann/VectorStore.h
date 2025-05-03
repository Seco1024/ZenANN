#pragma once
#include <vector>

namespace zenann {

using Vector = std::vector<float>;
using Dataset = std::vector<Vector>;

class VectorStore {
public:
    VectorStore() = default;
    ~VectorStore() = default;

    VectorStore(const VectorStore&) = default;
    VectorStore(VectorStore&&) noexcept = default;
    VectorStore& operator=(const VectorStore&) = default;
    VectorStore& operator=(VectorStore&&) noexcept = default;

    void add(const Dataset& data) {
        for (auto& v : data) {
            vectors_.push_back(v);
        }
    }

    const Dataset& getAll() const noexcept {
        return vectors_;
    }

    size_t size() const noexcept { 
        return vectors_.size();
    }

private:
    Dataset vectors_;
};
}