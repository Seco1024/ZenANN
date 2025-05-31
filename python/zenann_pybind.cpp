#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "IndexBase.h"
#include "IVFFlatIndex.h"
#include "KDTreeIndex.h"
#include "HNSWIndex.h"

namespace py = pybind11;
using namespace zenann;

// Trampoline class to allow instantiation of abstract IndexBase
struct PyIndexBase : IndexBase {
    using IndexBase::IndexBase;
    void train() override { }
    SearchResult search(const Vector&, size_t) const override { return {}; }
    void write_index(const std::string& filename) const override {
        PYBIND11_OVERRIDE_PURE(
            void,     
            IndexBase,   
            write_index,  
            filename    
        );
    }
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
        .def("write_index", &IndexBase::write_index, py::arg("filename"),
             "Write index to file")
        .def_property_readonly("dimension", &IndexBase::dimension,
             "Dimension of vectors");

    // Bind IVFFlatIndex
    py::class_<IVFFlatIndex, IndexBase, std::shared_ptr<IVFFlatIndex>>(m, "IVFFlatIndex")
    .def(py::init<size_t, size_t, size_t>(),
         py::arg("dim"), py::arg("nlist"), py::arg("nprobe") = 1)
    .def("build", &IVFFlatIndex::build, py::arg("data"))
    .def("train", &IVFFlatIndex::train)
    .def("search",
         py::overload_cast<const Vector&, size_t>(
             &IVFFlatIndex::search,
             py::const_),
         py::arg("query"), py::arg("k"))
    .def("search",
         py::overload_cast<const Vector&, size_t, size_t>(
             &IVFFlatIndex::search,
             py::const_),
         py::arg("query"), py::arg("k"), py::arg("nprobe"))
    .def("search_batch",
         py::overload_cast<const Dataset&, size_t>(
             &IVFFlatIndex::search_batch,
             py::const_),
         py::arg("queries"), py::arg("k"))
    .def("search_batch",
         py::overload_cast<const Dataset&, size_t, size_t>(
             &IVFFlatIndex::search_batch,
             py::const_),
         py::arg("queries"), py::arg("k"), py::arg("nprobe"))
    .def("write_index", &IVFFlatIndex::write_index, py::arg("filename"))
    .def_static("read_index", &IVFFlatIndex::read_index, py::arg("filename"));

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

     py::class_<HNSWIndex, IndexBase, std::shared_ptr<HNSWIndex>>(m, "HNSWIndex")
          .def(py::init<size_t,size_t,size_t>(),
               py::arg("dim"), py::arg("M"), py::arg("efConstruction")=200)
          .def("build", &HNSWIndex::build, py::arg("data"))
          .def("train", &HNSWIndex::train)
          .def("search", (SearchResult (HNSWIndex::*)(const Vector&,size_t) const)
                         &HNSWIndex::search, py::arg("query"), py::arg("k"))
          .def("search", (SearchResult (HNSWIndex::*)(const Vector&,size_t,size_t) const)
                         &HNSWIndex::search,
               py::arg("query"), py::arg("k"), py::arg("efSearch"))
          .def("search_batch", (std::vector<SearchResult> (HNSWIndex::*)(const Dataset&,size_t) const)
                              &HNSWIndex::search_batch,
               py::arg("queries"), py::arg("k"))
          .def("search_batch", (std::vector<SearchResult> (HNSWIndex::*)(const Dataset&,size_t,size_t) const)
                              &HNSWIndex::search_batch,
               py::arg("queries"), py::arg("k"), py::arg("efSearch"))
          .def("set_ef_search", &HNSWIndex::set_ef_search, py::arg("efSearch"))
          .def("reorder_layout", (void (HNSWIndex::*)()) &HNSWIndex::reorder_layout)
          .def("reorder_layout", (void (HNSWIndex::*)(const std::string&))
               &HNSWIndex::reorder_layout, py::arg("mapping_file"))
          .def("write_index", &HNSWIndex::write_index, py::arg("filename"))
          .def_static("read_index", &HNSWIndex::read_index, py::arg("filename"))
          .def_static("compute_recall_with_mapping",
               &HNSWIndex::compute_recall_with_mapping,
               py::arg("groundtruth"), py::arg("predicted_flat"),
               py::arg("nq"), py::arg("k"), py::arg("mapping_file"))
          ;
}
