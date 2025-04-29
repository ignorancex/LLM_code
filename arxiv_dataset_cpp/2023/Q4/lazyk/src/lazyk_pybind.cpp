#include <pybind11/pybind11.h>  
#include <pybind11/stl.h>  
#include <pybind11/numpy.h>
#include "lazyk.cpp"  
#include <vector>
  
namespace py = pybind11;  

PYBIND11_MODULE(lazyk, m) {  
    py::class_<Lazyk>(m, "lazyk")  
        .def(py::init<vector<vector<double>>, bool>(), py::arg("probs"), py::arg("cache_assignments") = true)
        .def("__iter__", [](Lazyk &self) { return self; })  
        .def("__next__", [](Lazyk &self) {  
            if (self.end()) {  
                throw py::stop_iteration();  
            }  

            // Convert the vector<int> to NumPy array
            std::vector<int> vec = self.getAssignment();
            py::array_t<int> arr(vec.size());
            std::copy(vec.begin(), vec.end(), arr.mutable_data());
            ++self;  
            return arr;
        });  
}  
