# ZenANN: A High-Performance Vector Similarity Search Library for Python Users

## Basic Information

**ZenANN** is a high-performance approximate nearest neighbor (ANN) similarity search library designed to be user-friendly for Python developers. It provides multiple indexing methods, such as **IVF** (Inverted File Index), **HNSW** (Hierarchical Navigable Small World), and **hybrid-index structures** to balance between accuracy and speed. The computation kernel of ZenANN will be optimized for cache efficiency, SIMD acceleration, and algorithms enhancements beyond existing in-memory libraries.

## Problem to Solve
Similarity search is a fundamental problem in many domains, including information retrieval, natural language processing, and so on. The key challenge is to efficiently find out the nearest neighbors of a query vector in high-dimensional space. However, as the data size and dimensionality grows, the performance of traditional brute-force search (eg. KD-tree) may suffers from **Curse of Dimensionality**.

To solve this problem, **approximate nearest neighbor (ANN)** search aims to retrieve near-optimal results while significantly reducing computation time. It trades off a small loss in accuracy for significant speed improvements, making them ideal for high-dimensional vector search applications.

Although existing in-memory implementations (eg. Faiss) are highly optimized, there are still areas for improvement:
- Improved index data layout for a better cache locality
- SIMD acceleration for a specific algorithm
- Enhancements on data structures / algorithms to better match hardware characteristics

## Prospective Users
ZenANN is designed for developers and researchers working on large-scale similarity search applications, including:
- **Machine learning engineers** who use ANN search for embedding-based retrieval in NLP, computer vision, and recommendation systems.
- **Software developers** who build applications requiring fast vector search with a clear, user-friendly programming interface.
- **Data scientists** who perform large-scale similarity analysis on high-dimensional datasets.

## System Architecture
ZenANN will be implemented in C++ for high performance and exposes an intuitive Python API using pybind11.
### Index Hierarchy
There will be an abstract base index, which provides a unified interface for different index classes.
1. **Base Index Class**
    - `indexBase`: Defines the common API for all indexing methods (eg. `add()`, `search()`, `train()`, `reorder_layout()`)

2. **Derived Index Classes**
    - `indexHNSW`: A graph-based structure for accurate and efficient ANN
    - `indexIVF`: A cluster-based structure for large dataset
3. **Hybrid Index Classes** 
    - `indexIVF_HNSW` / `indexHNSW_IVF` : For fast-coveraging larger datasets
4. (Optional) **Quantization Index Classes**
    - `indexPQ`: Combined with product quantization for memory-limited scenarios

Note: Actual implementation detail of HNSW may be built on Faiss's interface according to development progress

### Processing Flow
1. Initialize an index (e.g., `indexBase`, `indexHNSW`)
2. Build an index
2-1. Add the given vector data using `add()` to a specific index instance.
2-2. Train index with  `train()` if needed
2-3. Optimize the index data layout with `reorder_layout()` to improve cache locality.
4. Perform a query on the specified index instance using `search()`.
5. Evaluate accuracy using the `get_statistics()` API.

## API Description
There is a simple python examples for understanding the API design
```
import zenann

# Initialize an HNSW index
index = zenann.HNSWIndex(dimension=128, ef_construction=200, M=16)

# Add vectors to the index and conduct reordering
index.add(data_vectors)
index.train()
index.reorder_layout()

# Perform a search
results = index.search(query_vector, k=5, efSearch=100)
recall = get_statistics(results, ground_truth)
```

## Engineering Infrastructure
### Automatic Build System
- GNU make
### Version Control
- Git
- Github
### Testing Framework
- C++: Google Test
- Python: pytest
### Documentation
- Markdown
### Continuous Integration
- Github Actions

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
