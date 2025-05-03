#include "IndexBase.h"

namespace zenann {
    IndexBase::IndexBase(size_t dim)
    : dimension_(dim), 
      datastore_(std::make_shared<VectorStore>()) {}

    IndexBase::IndexBase(const IndexBase &other) 
    : dimension_(other.dimension_),
      datastore_(std::make_shared<VectorStore>(*other.datastore_)) {}
    
    IndexBase::IndexBase(IndexBase&& other) noexcept
    : dimension_(other.dimension_),
      datastore_(std::move(other.datastore_)) {}

    IndexBase& IndexBase::operator=(const IndexBase &other) {
        if (this != &other) {
            dimension_ = other.dimension_;
            datastore_ = std::make_shared<VectorStore>(*other.datastore_);
        }
        return *this;
    }

    IndexBase& IndexBase::operator=(IndexBase&& other) noexcept{
        if (this != &other) {
            dimension_ = other.dimension_;
            datastore_ = std::move(other.datastore_);
        }
        return *this;
    }

    IndexBase::~IndexBase() = default;

    void IndexBase::build(const Dataset& data) {
        datastore_->add(data);
        train();
    }
}