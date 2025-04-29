#include <torch/extension.h>

#include "marching_cubes.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // utils
    m.def("marching_cubes_33", &marching_cubes_33, "marching cubes");
}