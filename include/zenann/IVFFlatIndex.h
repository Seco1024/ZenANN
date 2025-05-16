#pragma once

#include "IndexBase.h"
#include <vector>

namespace zenann {

class IVFFlatIndex : public IndexBase {
public:
    IVFFlatIndex(size_t dim, size_t nlist, size_t nprobe = 1);
    ~IVFFlatIndex() override;
    void train() override;
    SearchResult search(const Vector& query, size_t k) const override;
    std::vector<SearchResult> search_batch(const Dataset& queries, size_t k) const;
    void write_index(const std::string& filename) const;
    static std::shared_ptr<IVFFlatIndex> read_index(const std::string& filename);
private:
    size_t                     nlist_;     
    size_t                     nprobe_;  
    Dataset                    centroids_; 
    std::vector<idList>        lists_;    
    void kmeans(const Dataset& data, size_t iterations = 10);
};

} 
