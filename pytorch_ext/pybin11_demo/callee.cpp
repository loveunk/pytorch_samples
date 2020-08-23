// filename: first_pybind11.cpp
#include <pybind11/pybind11.h>

namespace py = pybind11;

// define func 
int add(int i=1, int j=2) {
    return i + j;
}

//warper of the func 
PYBIND11_MODULE(callee, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers",
            py::arg("i")=1, py::arg("j")=2
            );
    m.attr("the_answer") = 42;
    m.attr("what") = "world";
}
