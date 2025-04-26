#include <torch/extension.h>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERT(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA forward declarations
void quat_to_mat_fw_cuda(
    torch::Tensor x, 
    torch::Tensor y);

void quat_to_mat_bw_cuda(
    torch::Tensor x, 
    torch::Tensor g_y, 
    torch::Tensor g_x);

void voxelize_fw_cuda(
    torch::Tensor points,
    torch::Tensor voxel_centers,
    torch::Tensor voxel_radii,
    float         sigma,
    torch::Tensor voxel_grids);

void voxelize_bw_cuda(
    torch::Tensor points,
    torch::Tensor voxel_centers,
    torch::Tensor voxel_radii,
    float         sigma,
    torch::Tensor voxel_grids,
    torch::Tensor grad_voxel_grids,
    torch::Tensor grad_voxel_centers,
    torch::Tensor grad_voxel_radii);


// C++ interface
void quat_to_mat_fw(
    torch::Tensor x, 
    torch::Tensor y) {

    CHECK_INPUT(x);
    CHECK_INPUT(y);
    return quat_to_mat_fw_cuda(x, y);
}

void quat_to_mat_bw(
    torch::Tensor x, 
    torch::Tensor g_y, 
    torch::Tensor g_x) {

    CHECK_INPUT(x);
    CHECK_INPUT(g_y);
    CHECK_INPUT(g_x);
    return quat_to_mat_bw_cuda(x, g_y, g_x);
}

void voxelize_fw(
    torch::Tensor points,
    torch::Tensor voxel_centers,
    torch::Tensor voxel_radii,
    float         sigma,
    torch::Tensor voxel_grids) {

    CHECK_INPUT(points);
    CHECK_INPUT(voxel_centers);
    CHECK_INPUT(voxel_radii);
    CHECK_INPUT(voxel_grids);

    return voxelize_fw_cuda(
        points,
        voxel_centers,
        voxel_radii,
        sigma,
        voxel_grids
    );
}

void voxelize_bw(
    torch::Tensor points,
    torch::Tensor voxel_centers,
    torch::Tensor voxel_radii,
    float         sigma,
    torch::Tensor voxel_grids,
    torch::Tensor grad_voxel_grids,
    torch::Tensor grad_voxel_centers,
    torch::Tensor grad_voxel_radii) {

    CHECK_INPUT(points);
    CHECK_INPUT(voxel_centers);
    CHECK_INPUT(voxel_radii);
    CHECK_INPUT(voxel_grids);
    CHECK_INPUT(grad_voxel_grids);
    CHECK_INPUT(grad_voxel_centers);
    CHECK_INPUT(grad_voxel_radii);

    return voxelize_bw_cuda(
        points,
        voxel_centers,
        voxel_radii,
        sigma,
        voxel_grids,
        grad_voxel_grids,
        grad_voxel_centers,
        grad_voxel_radii
    );
}

// Binding to python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quat_to_mat_fw", &quat_to_mat_fw, "Quaternion To Rotation Matrix forward (CUDA)");
    m.def("quat_to_mat_bw", &quat_to_mat_bw, "Quaternion To Rotation Matrix backward (CUDA)");
    m.def("voxelize_fw", &voxelize_fw, "Voxelization forward (CUDA)");
    m.def("voxelize_bw", &voxelize_bw, "Voxelization backward (CUDA)");
}
