#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <iostream>
#include <turboreg/turboreg.hpp>

namespace py = pybind11;
using namespace turboreg;

void bind_turboreg_gpu(py::module &m)
{
     py::class_<TurboRegGPU>(m, "TurboRegGPU")
         .def(py::init<int, float, int, float, float, const std::string &>(),
              py::arg("max_N"),
              py::arg("tau_length_consis"),
              py::arg("num_pivot"),
              py::arg("radiu_nms"),
              py::arg("tau_inlier"),
              py::arg("metric_str"))
         .def("run_reg", &TurboRegGPU::runRegCXXReturnTensor,
              py::arg("kpts_src_all"), py::arg("kpts_dst_all"),
              "Perform registration and return RigidTransform");
}

PYBIND11_MODULE(turboreg_gpu, m)
{
     m.doc() = "Python bindings for TurboRegGPU class using pybind11 and LibTorch";
     bind_turboreg_gpu(m);
}