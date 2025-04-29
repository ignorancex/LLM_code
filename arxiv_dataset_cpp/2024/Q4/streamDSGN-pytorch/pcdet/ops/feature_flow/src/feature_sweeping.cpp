#include <assert.h>
#include <torch/extension.h>
#include <stdint.h>
// #include <stdio.h>


#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


at::Tensor feature_sweeping_launcher(
    const at::Tensor& features,
    const at::Tensor& shift_range,
    int depths,
    int channels,
    int height,
    int width
);

// void feature_sweeping_gpu(
//     at::Tensor features,
//     at::Tensor shift_range,
//     at::Tensor output
// );


at::Tensor feature_sweeping_gpu(
    at::Tensor& features,
    at::Tensor& shift_range
) 
{
    CHECK_INPUT(features);
    CHECK_INPUT(shift_range);

    int D = shift_range.size(0);
    int C = features.size(0);
    int H = features.size(1);
    int W = features.size(2);

    return feature_sweeping_launcher(
        features,
        shift_range,
        D, C, H, W
  );
}


// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("feature_sweeping_gpu", &feature_sweeping_gpu, "cuda version of feature sweeping.");
// }