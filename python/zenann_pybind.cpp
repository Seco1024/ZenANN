#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "IndexBase.h"
#include "IVFFlatIndex.h"
#include "KDTreeIndex.h"

namespace py = pybind11;
using namespace zenann;

// Trampoline class to allow instantiation of abstract IndexBase
struct PyIndexBase : IndexBase {
    using IndexBase::IndexBase;
    void train() override { }
    SearchResult search(const Vector&, size_t) const override { return {}; }
};

PYBIND11_MODULE(zenann, m) {
    m.doc() = "ZenANN Library";

    // Bind SearchResult
    py::class_<SearchResult>(m, "SearchResult")
        .def_readonly("indices", &SearchResult::indices)
        .def_readonly("distances", &SearchResult::distances);

    // Bind IndexBase
    py::class_<IndexBase, PyIndexBase, std::shared_ptr<IndexBase>>(m, "IndexBase")
        .def(py::init<size_t>(), py::arg("dim"))
        .def("build", &IndexBase::build, py::arg("data"),
             "Add data to the index and train")
        .def("train", &IndexBase::train,
             "Train the index (abstract stub)")
        .def("search", &IndexBase::search, py::arg("query"), py::arg("k"),
             "Search k nearest neighbors")
        .def_property_readonly("dimension", &IndexBase::dimension,
             "Dimension of vectors");

    // Bind IVFFlatIndex
    py::class_<IVFFlatIndex, IndexBase, std::shared_ptr<IVFFlatIndex>>(m, "IVFFlatIndex")
        .def(py::init<size_t, size_t, size_t>(),
             py::arg("dim"), py::arg("nlist"), py::arg("nprobe") = 1)
        .def("build", &IVFFlatIndex::build, py::arg("data"),
             "Add data and train the IVF index")
        .def("train", &IVFFlatIndex::train,
             "Train the IVF index (run K-means and build inverted lists)")
        .def("search", &IVFFlatIndex::search, py::arg("query"), py::arg("k"),
             "Search top-k nearest neighbors")
        .def("search_batch", &IVFFlatIndex::search_batch, py::arg("queries"), py::arg("k"),
             "Search top-k for a batch of queries")
        // Persistence API
        .def("write_index", &IVFFlatIndex::write_index, py::arg("filename"),
             "Serialize the index to a file")
        .def_static("read_index", &IVFFlatIndex::read_index, py::arg("filename"),
             "Load an index from a file");

    py::class_<KDTreeIndex, IndexBase, std::shared_ptr<KDTreeIndex>>(m, "KDTreeIndex")
        .def(py::init<size_t>(), py::arg("dim"))
        .def("build", &KDTreeIndex::build, py::arg("data"),
             "Add data and build KDTree index")
        .def("train", &KDTreeIndex::train,
             "Rebuild KDTree index")
        .def("search", &KDTreeIndex::search, py::arg("query"), py::arg("k"),
             "Exact k-NN search using KDTree")
        .def("write_index", &KDTreeIndex::write_index, py::arg("filename"),
             "Serialize KDTree index to file")
        .def_static("read_index", &KDTreeIndex::read_index, py::arg("filename"),
             "Load KDTree index from file");
}
