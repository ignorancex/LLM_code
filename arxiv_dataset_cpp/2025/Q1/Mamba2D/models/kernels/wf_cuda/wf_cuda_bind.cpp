#include <torch/extension.h>

void wf_fwd_launcher(torch::Tensor& hs,
                     const torch::Tensor& deltaAT,
                     const torch::Tensor& deltaAL,
                     const torch::Tensor& BXT,
                     const torch::Tensor& BXL);

void wf_fwd_binding(torch::Tensor& hs,
                    const torch::Tensor& deltaAT,
                    const torch::Tensor& deltaAL,
                    const torch::Tensor& BXT,
                    const torch::Tensor& BXL)
{
    wf_fwd_launcher(hs, deltaAT, deltaAL, BXT, BXL);
}

void wf_bwd_launcher(const torch::Tensor& hs,
                     const torch::Tensor& deltaAT,
                     const torch::Tensor& deltaAL,
                     const torch::Tensor& grad_output,
                     torch::Tensor& dDAT,
                     torch::Tensor& dDAL,
                     torch::Tensor& omega,
                     torch::Tensor& dBX);

void wf_bwd_binding(const torch::Tensor& hs,
                    const torch::Tensor& deltaAT,
                    const torch::Tensor& deltaAL,
                    const torch::Tensor& grad_output,
                    torch::Tensor& dDAT,
                    torch::Tensor& dDAL,
                    torch::Tensor& omega,
                    torch::Tensor& dBX)
{
    wf_bwd_launcher(hs, deltaAT, deltaAL, grad_output, dDAT, dDAL, omega, dBX);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "wf_fwd",
        &wf_fwd_binding,
        "Launch wf_fwd kernel"
    );
    
    m.def(
        "wf_bwd",
        &wf_bwd_binding,
        "Launch wf_bwd kernel"
    );
}
