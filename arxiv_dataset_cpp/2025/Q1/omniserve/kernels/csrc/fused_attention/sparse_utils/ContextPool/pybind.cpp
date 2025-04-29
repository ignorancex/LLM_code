#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "context_pool_kernel.h"







PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_min_max_pool", &context_paged_min_max_pool, "context_paged_min_max_pool");
}