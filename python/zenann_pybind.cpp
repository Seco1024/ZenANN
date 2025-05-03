#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "IndexBase.h"

namespace py = pybind11;
using namespace zenann;

struct PyIndexBase : IndexBase {
    using IndexBase::IndexBase;
    void train() override {}
    SearchResult search(const Vector& , size_t) const override {
        return {};  
    }
};

PYBIND11_MODULE(zenann, m) {
    m.doc() = "ZenANN Library";

    py::class_<IndexBase, PyIndexBase>(m, "IndexBase")
        .def(py::init<size_t>(), py::arg("dim"))
        .def("build", &IndexBase::build, py::arg("data"),
             "Add data and train index")
        .def("train", &IndexBase::train,
             "Train index")
        .def("search", &IndexBase::search, py::arg("query"), py::arg("k"),
             "Search k-NN")
        .def_property_readonly("dimension", &IndexBase::dimension,
             "Vector dimension");

    py::class_<SearchResult>(m, "SearchResult")
        .def_readonly("indices", &SearchResult::indices)
        .def_readonly("distances", &SearchResult::distances);
}
