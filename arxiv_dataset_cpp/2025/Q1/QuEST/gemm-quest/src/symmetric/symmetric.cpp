#include "symmetric/symmetric.h"

#include <torch/extension.h>

#include "symmetric/symmetric_internal.h"

namespace QUIK::symmetric {

torch::Tensor int4FusedDequantize(const torch::Tensor &A,
                                  const torch::Tensor &B,
                                  const torch::Tensor &scale_row,
                                  const torch::Tensor &scale_col,
                                  const torch::Tensor &vec_a_add,
                                  const torch::Tensor &vec_b_add,
                                  const torch::Tensor &y) {
  torch::checkAllContiguous("int4FusedDequantize", {{A, "A", 0},
                                                    {B, "B", 1},
                                                    {scale_row, "scale_row", 2},
                                                    {scale_col, "scale_col", 3},
                                                    {vec_a_add, "vec_a_add", 4},
                                                    {vec_b_add, "vec_b_add", 5},
                                                    {y, "y", 6}});
  torch::checkDeviceType("int4FusedDequantize", {A, B, scale_row, scale_col,  vec_a_add, vec_b_add, y},
                         at::DeviceType::CUDA);
  return int4FusedDequantizeCUDA(A, B, scale_row, scale_col, vec_a_add, vec_b_add, y);
}

void buildSubmodule(py::module &mod) {
  py::module m = mod.def_submodule("symmetric", "Symmetric Functions");
  m.def(
      "int4FusedDequantize", &int4FusedDequantize,
      "input: (A: torch.Tensor(M x K/2, UINT8, CUDA), B: torch.Tensor(N x K/2, "
      "UINT8, CUDA)\n"
      "scale_row: torch.Tensor(M x 1, FP16, CUDA), scale_col: torch.Tensor(1 x "
      "N, FP16, CUDA)"
      "y: torch.Tensor(M x N, FP16, CUDA))"
      "output: torch.Tensor(M x N, INT32, CUDA)\n"
      "output = int4Unpacking(A) @ int4Unpacking(B)^T * scale_row * scale_cal "
      "+ y",
      py::arg("A"), py::arg("B"), py::arg("scale_row"), py::arg("scale_col"), py::arg("vec_a_add"), py::arg("vec_b_add"),
      py::arg("y"));
}

}  // namespace QUIK::symmetric
