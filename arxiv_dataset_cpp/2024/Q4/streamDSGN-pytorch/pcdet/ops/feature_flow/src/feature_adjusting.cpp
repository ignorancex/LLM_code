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


at::Tensor feature_adjusting_forward_launcher(
    const at::Tensor& features,
    const at::Tensor& shift_coord,
    at::Tensor& shift_counter,
    int height,
    int width,
    int channels
);


at::Tensor feature_adjusting_backward_launcher(
    const at::Tensor& d_output,
    const at::Tensor& shift_coord,
    const at::Tensor& shift_counter,
    int height,
    int width,
    int channels
);


at::Tensor feature_adjusting_forward(
    at::Tensor& features,
    at::Tensor& shift_coord,
    at::Tensor& shift_counter
) 
{
    CHECK_INPUT(features);
    CHECK_INPUT(shift_coord);
    CHECK_INPUT(shift_counter);

    // int N = features.size(0);
    int C = features.size(0);
    int H = features.size(1);
    int W = features.size(2);

    // printf("C: %d, H: %d, W: %d\n", C, H, W);
    return feature_adjusting_forward_launcher(
        features,
        shift_coord,
        shift_counter,
        H, W, C
    );
}

at::Tensor feature_adjusting_backward(
    at::Tensor& d_output,
    at::Tensor& shift_coord,
    at::Tensor& shift_counter
) 
{
    CHECK_INPUT(d_output);
    CHECK_INPUT(shift_coord);
    CHECK_INPUT(shift_counter);

    // int N = features.size(0);
    int C = d_output.size(0);
    int H = d_output.size(1);
    int W = d_output.size(2);

    // printf("C: %d, H: %d, W: %d\n", C, H, W);

    return feature_adjusting_backward_launcher(
        d_output,
        shift_coord,
        shift_counter,
        H, W, C
    );
}