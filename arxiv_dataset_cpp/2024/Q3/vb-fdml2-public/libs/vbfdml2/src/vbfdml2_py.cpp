#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vbfdml2.hpp>
using namespace vbfdml2;

namespace py = pybind11;

std::vector<VoxelCloud> preprocess_scene(int n, BoxExtent& extent, std::vector<Segment> scene, bool bUseGPU = true) {
    VoxelCloud preprocess, mask;
    preprocess.Init(n);
    mask.Init(n);
    PreprocessScene(preprocess, mask, extent, scene, bUseGPU);

    std::vector<VoxelCloud> res;
    res.push_back(preprocess);
    res.push_back(mask);
    return res;
}

VoxelCloud marching_voxels(VoxelCloud& preprocess, BoxExtent& extent, float dx, float dy, float dtheta, float d, bool bUseGPU = true)
{
    VoxelCloud output;
    output.Init(preprocess.n);
    MarchingVoxels(output, extent, preprocess, dx, dy, dtheta, d, bUseGPU);
    return output;
}

VoxelCloud conv3d(VoxelCloud& m, bool bUseGPU = true)
{
    VoxelCloud output;
    output.Init(m.n);
    Conv3D(output, m, bUseGPU);
    return output;
}

VoxelCloud do_intersect(VoxelCloud& m1, VoxelCloud& m2, bool bUseGPU = true)
{
    VoxelCloud output;
    output.Init(m1.n);
    DoIntersect(output, m1, m2, bUseGPU);
    return output;
}

std::vector<Prediction> predict(VoxelCloud& vc, BoxExtent& extent)
{
    std::vector<Prediction> predictions;
    Predict(vc, extent, predictions);
    return predictions;
}

std::vector<Prediction> run_vbfdml(std::vector<Segment> scene, BoxExtent& extent, int n, std::vector<Measurement>& measurements, bool bUseGPU = true)
{
    std::vector<Prediction> predictions;
    RunVBFDML(scene, extent, n, measurements, predictions, bUseGPU);
    return predictions;
}

std::vector<Prediction> run_improved_vbfdml(std::vector<Segment> scene, BoxExtent& extent, std::vector<Measurement>& measurements, int n, int l, float epsilon) {
    std::vector<Prediction> predictions;
    RunImprovedVBFDML(predictions, scene, extent, measurements, n, l, epsilon);
    return predictions;
}


PYBIND11_MODULE(vbfdml2, m) {
    m.doc() = "New version of (voxel-based approximation of) few distance-measurement localization, with support for uncertain environments.";

    py::class_<Segment>(m, "Segment")
        .def(py::init<float, float, float, float>())
        .def_readwrite("x1", &Segment::x1)
        .def_readwrite("x2", &Segment::x2)
        .def_readwrite("y1", &Segment::y1)
        .def_readwrite("y2", &Segment::y2)
        .def("__repr__", &Segment::ToString)
    ;

    py::class_<BoxExtent>(m, "BoxExtent")
        .def(py::init<float, float, float, float, float, float>())
        .def_readwrite("width", &BoxExtent::width)
        .def_readwrite("height", &BoxExtent::height)
        .def_readwrite("depth", &BoxExtent::depth)
        .def_readwrite("x0", &BoxExtent::x0)
        .def_readwrite("y0", &BoxExtent::y0)
        .def_readwrite("theta0", &BoxExtent::theta0)
        .def("__repr__", &BoxExtent::ToString)
    ;

    py::class_<VoxelCloud>(m, "VoxelCloud")
        .def(py::init<>())
        .def("Init", &VoxelCloud::Init)
        .def("Delete", &VoxelCloud::Delete)
        .def("ToList", &VoxelCloud::ToList)
        .def_readonly("n", &VoxelCloud::n)
        .def("__repr__", &VoxelCloud::ToString)
        .def("ExportOBJ", &VoxelCloud::ExportOBJ)
    ;

    py::class_<Prediction>(m, "Prediction")
        .def(py::init<float, float, float, int>(), py::arg("x") = 0, py::arg("y") = 0, py::arg("theta") = 0, py::arg("cnt") = 0)
        .def_readwrite("x", &Prediction::x)
        .def_readwrite("y", &Prediction::y)
        .def_readwrite("theta", &Prediction::theta)
        .def_readwrite("cnt", &Prediction::cnt)
        .def("__repr__", &Prediction::ToString)
    ;

    py::class_<DynamicObstacle>(m, "DynamicObstacle")
        .def(py::init<float, float, float, std::vector<std::pair<float, float>>>())
        .def_readwrite("x", &DynamicObstacle::x)
        .def_readwrite("y", &DynamicObstacle::y)
        .def_readwrite("radius", &DynamicObstacle::radius)
        .def_readwrite("path", &DynamicObstacle::path)
        .def("__repr__", &DynamicObstacle::ToString)
    ;

    py::class_<Measurement>(m, "Measurement")
        .def(py::init<float, float, float, float, bool>(), py::arg("dx"), py::arg("dy"), py::arg("dtheta"), py::arg("dist"), py::arg("d_obstacle") = false)
        .def_readwrite("dx", &Measurement::dx)
        .def_readwrite("dy", &Measurement::dy)
        .def_readwrite("dtheta", &Measurement::dtheta)
        .def_readwrite("dist", &Measurement::dist)
        .def_readwrite("d_obstacle", &Measurement::d_obstacle)
        .def("__repr__", &Measurement::ToString)
    ;

    m.def("PreprocessScene", &PreprocessScene, 
        "[Original C++ syntax]\n"
        "Computes the distance function h(x,y,theta)\n"
        "This can be used as pre-processing and run only once per scene\n"
        "Also output a voxel cloud mask which is 1 for each voxel that is inside the room");
    m.def("preprocess_scene", &preprocess_scene, 
        "[Python syntax]\n"
        "Computes the distance function h(x,y,theta)\n"
        "This can be used as pre-processing and run only once per scene\n"
        "Also output a voxel cloud mask which is 1 for each voxel that is inside the room");
    
    m.def("MarchingVoxels", &MarchingVoxels,
        "[Original C++ syntax]\n"
        "Given a preprocessed scene, compute the preimage via the marching voxels algorithm, after\n"
        "some pre-determined planar motion (dx, dy, dtheta) and recieving a measurment (d)");
    m.def("marching_voxels", &marching_voxels,
        "[Python syntax]\n"
        "Given a preprocessed scene, compute the preimage via the marching voxels algorithm, after\n"
        "some pre-determined planar motion (dx, dy, dtheta) and recieving a measurment (d)");

    m.def("Conv3D", &Conv3D,
        "[Original C++ syntax]\n"
        "Apply 3D convolution to a voxel cloud (the heuristic described in the paper for achieveing better completeness)\n");
    m.def("conv3d", &conv3d,
        "[Python syntax]\n"
        "Apply 3D convolution to a voxel cloud (the heuristic described in the paper for achieveing better completeness)\n");


    m.def("DoIntersect", &DoIntersect,
        "[Original C++ syntax]\n"
        "Intersect voxel clouds\n");
    m.def("do_intersect", &do_intersect,
        "[Python syntax]\n"
        "Intersect voxel clouds\n");

    m.def("Predict", &Predict,
        "[Original C++ syntax]\n"
        "Outputs predictions of center of mass of each component in the voxel cloud\n");
    m.def("predict", &predict,
        "[Python syntax]\n"
        "Outputs predictions of center of mass of each component in the voxel cloud\n");

    m.def("RunVBFDML", &RunVBFDML,
        "[Original C++ syntax]\n"
        "Runs the whole algorithm\n");
    m.def("run_vbfdml", &run_vbfdml,
        "[Python syntax]\n"
        "Runs the whole algorithm\n");

    // m.def("RunImprovedVBFDML", &RunImprovedVBFDML,
    //     "[Original C++ syntax]\n"
    //     "Runs the whole (improved) algorithm\n");
    m.def("run_improved_vbfdml", &run_improved_vbfdml,
        "[Python syntax]\n"
        "Runs the whole (improved) algorithm\n");

}