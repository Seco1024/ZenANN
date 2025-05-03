#pragma once
#include <memory>
#include "VectorStore.h"

namespace zenann {
using idList = std::vector<size_t>;

struct SearchResult {
    idList              indices;
    std::vector<float>  distances;
};

class IndexBase {
public:
    explicit IndexBase(size_t dim);

    IndexBase(const IndexBase&);
    IndexBase(IndexBase &&) noexcept;
    IndexBase& operator=(const IndexBase&);
    IndexBase& operator=(IndexBase&&) noexcept;
    virtual ~IndexBase();

    virtual void build(const Dataset& data);
    virtual void train() = 0;
    virtual SearchResult search(const Vector& query, size_t k) const = 0;

    size_t dimension() const noexcept {
        return dimension_;
    }

    const Dataset& data() const noexcept { 
        return datastore_->getAll(); 
    }

protected:
    size_t                        dimension_;
    std::shared_ptr<VectorStore>  datastore_;
};
}