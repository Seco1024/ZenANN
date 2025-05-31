```mermaid
classDiagram
    direction LR

    class VectorStore {
        <<C++ Class>>
        -Dataset vectors_
        +add(const Dataset& data) void
        +getAll() const Dataset&
        +size() const size_t
    }

    class SearchResult {
        <<C++ Struct>>
        +idList indices
        +vector~float~ distances
    }

    class IndexBase {
        <<Abstract C++ Class>>
        #size_t dimension_
        #shared_ptr~VectorStore~ datastore_
        +IndexBase(size_t dim)
        +virtual build(const Dataset& data) void
        +virtual train()* void
        +virtual search(const Vector& query, size_t k) const SearchResult*
        +virtual write_index(const string& filename) const void* %% Added pure virtual method
        +dimension() const size_t
        +data() const Dataset&
    }

    IndexBase o-- "1" VectorStore : aggregates

    class KDTreeIndex {
        <<C++ Class>>
        -Node* root_
        +KDTreeIndex(size_t dim)
        +train() void
        +search(const Vector& query, size_t k) const SearchResult
        +write_index(const string& filename) const void    %% Overrides IndexBase::write_index
        +read_index(const string& filename) shared_ptr~KDTreeIndex~
    }
    IndexBase <|-- KDTreeIndex
    KDTreeIndex ..> SearchResult : returns
    KDTreeIndex "1" *-- "0..*" Node : contains

    class Node {
        <<Inner C++ Struct of KDTreeIndex>>
        +size_t idx
        +size_t dim
        +float val
        +Node* left
        +Node* right
        +Node(size_t i, size_t d, float v)
    }


    class IVFFlatIndex {
        <<C++ Class>>
        -size_t nlist_
        -size_t nprobe_
        -Dataset centroids_
        -vector~idList~ lists_
        +IVFFlatIndex(size_t dim, size_t nlist, size_t nprobe)
        +train() void
        +search(const Vector& query, size_t k) const SearchResult
        +search(const Vector& query, size_t k, size_t nprobe) const SearchResult
        +search_batch(const Dataset& queries, size_t k) const vector~SearchResult~
        +write_index(const string& filename) const void    %% Overrides IndexBase::write_index
        +read_index(const string& filename) shared_ptr~IVFFlatIndex~
        -kmeans(const Dataset& data, size_t iterations) void
    }
    IndexBase <|-- IVFFlatIndex
    IVFFlatIndex ..> SearchResult : returns
    IVFFlatIndex ..> SimdUtils : uses

    class HNSWIndex {
        <<C++ Class>>
        -unique_ptr~faiss::IndexHNSWFlat~ idx_
        +HNSWIndex(size_t dim, size_t M, size_t efConstruction)
        +train() void
        +search(const Vector& query, size_t k) const SearchResult
        +search(const Vector& query, size_t k, size_t efSearch) const SearchResult
        +search_batch(const Dataset& queries, size_t k) const vector~SearchResult~
        +set_ef_search(size_t efSearch) void
        +reorder_layout() void
        +reorder_layout(const string& mapping_file) void
        +write_index(const string& filename) const void    %% Overrides IndexBase::write_index
        +read_index(const string& filename) shared_ptr~HNSWIndex~
    }
    IndexBase <|-- HNSWIndex
    HNSWIndex ..> SearchResult : returns

    class PyIndexBase {
        <<C++ Trampoline Class for Pybind11>>
        +PyIndexBase(size_t dim)
        +train() void
        +search(const Vector&, size_t) const SearchResult
        +write_index(const string& filename) const void    %% Added override
    }
    IndexBase <|-- PyIndexBase

    class SimdUtils {
        <<C++ Utility Namespace/Static Class>>
        +static l2_simd(const float* a, const float* b, size_t dim) float
    }