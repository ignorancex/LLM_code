#include <assert.h>
#include <torch/extension.h>
#include <stdint.h>

#include "feature_flow.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("feature_sweeping_gpu", &feature_sweeping_gpu, "cuda version of feature sweeping.");
	m.def("feature_adjusting_forward", &feature_adjusting_forward, "cuda version of feature adjusting.");
	m.def("feature_adjusting_backward", &feature_adjusting_backward, "cuda version of feature adjusting for backward.");
}