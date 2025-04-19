#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <iostream>
// #include <unsupported/Eigen/CXX11/Tensor>

namespace py = pybind11;

template <class T>
inline T clamp(T in, T lb, T ub)
{
    return std::min<T>(ub, std::max<T>(in, lb));
}

template <class T>
inline T segment_distance(const Eigen::Matrix<T, -1, 1>& point1s, const Eigen::Matrix<T, -1, 1>& point1e, const Eigen::Matrix<T, -1, 1>& point2s, const Eigen::Matrix<T, -1, 1>& point2e)
{
    const Eigen::Matrix<T, -1, 1> d1 = point1e - point1s;
    const Eigen::Matrix<T, -1, 1> d2 = point2e - point2s;
    const Eigen::Matrix<T, -1, 1> d12 = point2s - point1s;

    T D1 = d1.dot(d1);
    T D2 = d2.dot(d2);

    T S1 = d1.dot(d12);
    T S2 = d2.dot(d12);
    T R = d1.dot(d2);

    T den = D1 * D2 - R * R;

    T u, t;
    if (D1 == 0.0 || D2 == 0.0)
    {
        if (D1 != 0.0)
        {
            u = 0.0;
            t = S1 / D1;
            t = clamp(t, 0.0, 1.0);
        }
        else if (D2 != 0.0)
        {
            t = 0.0;
            u = -S2 / D2;
            u = clamp(u, 0.0, 1.0);
        }
        else
        {
            t = 0.0;
            u = 0.0;
        }
    }
    else if (den == 0.0)
    {
        t = 0.0;
        u = -S2 / D2;
        T uf = clamp(u, 0.0, 1.0);
        if (uf != u)
        {
            t = (uf * R + S1) / D1;
            t = clamp(t, 0.0, 1.0);
            u = uf;
        }
    }
    else
    {
        t = (S1 * D2 - S2 * R) / den;
        t = clamp(t, 0.0, 1.0);
        u = (t * R - S2) / D2;
        T uf = clamp(u, 0.0, 1.0);
        if (uf != u)
        {
            t = (uf * R + S1) / D1;
            t = clamp(t, 0.0, 1.0);
            u = uf;
        }
    }

    return (d1 * t - d2 * u - d12).norm();
}

template <class T>
Eigen::Matrix<T, -1, 1> get_third_dimension_as_vector(T* in, const std::vector<ssize_t>& shape, int i, int j)
{
    Eigen::Matrix<T, -1, 1> out(shape[2]);
    for (int k = 0; k < shape[2]; ++k)
    {
        out(k) = in[shape[1] * shape[2] * i + shape[2] * j + k];
    }
    return out;
}

template <class T>
Eigen::Matrix<T, -1, -1> trajectory_segment_distance_pybind11(const py::array_t<T>& demos_array)
{
    // request a buffer descriptor from Python
    py::buffer_info buffer_info = demos_array.request();

    // extract data and shape of input array
    T* data = static_cast<T*>(buffer_info.ptr);
    std::vector<ssize_t> shape = buffer_info.shape;

    if (shape.size() != 3) throw std::runtime_error("Input shape must be 3-dimensional: (sample-dim, time-dim, state-dim).");
    // std::cout << "shape = (" << shape[0] << "," << shape[1] << "," << shape[2] << ")" << std::endl;

    // wrap ndarray in Eigen::Map:
    // the second template argument is the rank of the tensor and has to be known at compile time
    // Eigen::TensorMap<const Eigen::Tensor<T, 3>, Eigen::RowMajor> demos(data, {shape[0], shape[1], shape[2]});  // , Eigen::RowMajor

    ssize_t n = shape[1] - 1;
    ssize_t N = shape[0] * n;
    Eigen::Matrix<T, -1, -1> D = Eigen::Matrix<T, -1, -1>::Zero(N, N);

    Eigen::Matrix<T, -1, 1> a1(shape[2]), a2(shape[2]), b1(shape[2]), b2(shape[2]);
    T d;
    for (ssize_t i = 0; i < shape[0]; ++i)
    {
        for (ssize_t j = i; j < shape[0]; ++j)
        {
            for (ssize_t k = 0; k < n; ++k)
            {
                for (ssize_t l = 0; l < n; ++l)
                {
                    if (i == j && k == l)
                        continue;

                    a1 = get_third_dimension_as_vector<T>(data, shape, i, k);
                    a2 = get_third_dimension_as_vector<T>(data, shape, i, k + 1);
                    b1 = get_third_dimension_as_vector<T>(data, shape, j, l);
                    b2 = get_third_dimension_as_vector<T>(data, shape, j, l + 1);

                    d = segment_distance<T>(a1, a2, b1, b2);
                    D(i * n + k, j * n + l) = d;
                    D(j * n + l, i * n + k) = d;
                }
            }
        }
    }

    return D;
}

template <class T>
Eigen::Matrix<T, -1, -1> trajectory_segment_distance(const std::vector<Eigen::Matrix<T, -1, -1>>& demos_array)
{
    std::size_t num_samples = demos_array.size();
    ssize_t n = demos_array[0].rows() - 1;  // timesteps == rows
    ssize_t N = num_samples * n;
    // std::cout << "n=" << n << ", N=" << N << ", num_samples=" << num_samples << ", rows=" << demos_array[0].rows() << ", cols=" << demos_array[0].cols()<< std::endl;
    Eigen::Matrix<T, -1, -1> D = Eigen::Matrix<T, -1, -1>::Zero(N, N);

    Eigen::Matrix<T, -1, 1> a1(demos_array[0].rows()), a2(demos_array[0].rows()), b1(demos_array[0].rows()), b2(demos_array[0].rows());
    T d;
    for (ssize_t i = 0; i < num_samples; ++i)
    {
        for (ssize_t j = i; j < num_samples; ++j)
        {
            for (ssize_t k = 0; k < n; ++k)
            {
                for (ssize_t l = 0; l < n; ++l)
                {
                    if (i == j && k == l)
                        continue;

                    a1 = demos_array[i].row(k);
                    a2 = demos_array[i].row(k + 1);
                    b1 = demos_array[j].row(l);
                    b2 = demos_array[j].row(l + 1);

                    d = segment_distance<T>(a1, a2, b1, b2);
                    D(i * n + k, j * n + l) = d;
                    D(j * n + l, i * n + k) = d;
                }
            }
        }
    }

    return D;
}

template <class T>
void trajectory_mod(Eigen::Ref<Eigen::Matrix<T, -1, -1>> D, std::vector<ssize_t> shape, bool connect_start = true, bool connect_end = true)
{
    constexpr T distance_to_set_to_start_end = T(0.0);
    constexpr T distance_to_set_to_intermediate = T(1e-14);

    if (D.rows() != D.cols()) throw std::invalid_argument("Distance matrix must have same number of rows and columns.");
    if (shape.size() < 2) throw std::invalid_argument("Size of shape must be at least 2.");
    if (D.rows() != shape[0] * shape[1]) throw std::invalid_argument("Wrong shape.");

    for (Eigen::Index i = 0; i < shape[0]; ++i)
    {
        for (Eigen::Index j = i + 1; j < shape[0]; ++j)
        {
            if (connect_start)
            {
                D(i * shape[1], j * shape[1]) = distance_to_set_to_start_end;
                D(j * shape[1], i * shape[1]) = distance_to_set_to_start_end;
            }
            if (connect_end)
            {
                D((i + 1) * shape[1] - 1, (j + 1) * shape[1] - 1) = distance_to_set_to_start_end;
                D((j + 1) * shape[1] - 1, (i + 1) * shape[1] - 1) = distance_to_set_to_start_end;
            }
        }
        for (Eigen::Index j = 0; j < shape[1] - 1; ++j)
        {
            D(i * shape[1] + j, i * shape[1] + j + 1) = distance_to_set_to_intermediate;
            D(i * shape[1] + j + 1, i * shape[1] + j) = distance_to_set_to_intermediate;
        }
    }
}

template <class T>
Eigen::Matrix<T, -1, -1> compute_distance_matrix_from_trajectories_pybind11(const py::array_t<T>& demos)
{
    Eigen::Matrix<T, -1, -1> D = trajectory_segment_distance_pybind11<T>(demos);
    std::vector<ssize_t> shape = demos.request().shape;
    trajectory_mod<T>(D, {shape[0], shape[1] - 1});
    return D;
}

template <class T>
Eigen::Matrix<T, -1, -1> get_sample_as_matrix(T* in, const std::vector<ssize_t>& shape, int i)
{
    Eigen::Matrix<T, -1, -1> out(shape[1], shape[2]);
    for (int j = 0; j < shape[1]; ++j)
    {
        for (int k = 0; k < shape[2]; ++k)
        {
            out(j, k) = in[shape[1] * shape[2] * i + shape[2] * j + k];
        }
    }
    return out;
}

template <class T>
Eigen::Matrix<T, -1, -1> get_pairwise_trajectory_distance_matrix_simple(const py::array_t<T>& demos_array)
{
    // request a buffer descriptor from Python
    py::buffer_info buffer_info = demos_array.request();

    // extract data and shape of input array
    T* data = static_cast<T*>(buffer_info.ptr);
    std::vector<ssize_t> shape = buffer_info.shape;

    if (shape.size() != 3) throw std::runtime_error("Input shape must be 3-dimensional: (sample-dim, time-dim, state-dim).");

    ssize_t n = shape[0];  // Number of samples

    Eigen::Matrix<T, -1, -1> D = Eigen::Matrix<T, -1, -1>::Zero(n, n);

    Eigen::Matrix<T, -1, -1> traj_i(shape[1], shape[2]), traj_j(shape[1], shape[2]);
    Eigen::Matrix<T, -1, -1> D_tmp(n, n);
    T d;
    ssize_t n2;
    for (ssize_t i = 0; i < n - 1; ++i)
    {
        traj_i = get_sample_as_matrix<T>(data, shape, i);
        // std::cout << "traj_i: i = " << i << "\n" << traj_i << std::endl; // << traj_i.rows() << "x" << traj_i.cols() << std::endl;
        for (ssize_t j = i + 1; j < n; ++j)
        {
            traj_j = get_sample_as_matrix<T>(data, shape, j);
            // std::cout << "traj_j: " << traj_j.rows() << "x" << traj_j.cols() << std::endl;

            D_tmp = trajectory_segment_distance<T>({traj_i, traj_j});
            trajectory_mod<T>(D_tmp, {2, shape[1] - 1});

            n2 = std::floor<ssize_t>(D_tmp.rows() / 2);
            d = D_tmp.block(n2, 0, n2, n2).colwise().minCoeff().rowwise().maxCoeff()(0);
            // std::cout << "(" << i << "," << j << ")=" << d << std::endl;
            // d = D_tmp.template triangularView<Eigen::StrictlyLower>().m_matrix.rowwise().minCoeff().colwise().maxCoeff()(0);
            D(i, j) = d;
            D(j, i) = d;

            // if (i==1 && j == 2) {
            //     // std::cout << "cpp " << i << "," << j << "\n" << D_tmp << "\n\n" << std::endl;
            //     std::cout << "n2=" << n2 << ", D=(" << D_tmp.rows() << "," << D_tmp.cols() << ")" << std::endl;
            //     std::cout << "cpp-sub-D_\n" << D_tmp.block(n2, 0, n2, n2) << std::endl;
            //     // std::cout << "cpp-min: " << D_tmp.block(n2, 0, n2, n2).colwise().minCoeff() << std::endl;
            //     // std::cout << "max: " << D_tmp.block(n2, 0, n2, n2).colwise().minCoeff().rowwise().maxCoeff()(0) << std::endl;
            // }
        }
    }

    return D;
}

PYBIND11_MODULE(homology_clustering_py, m)
{
    m.def("segment_distance", &segment_distance<double>);
    m.def("trajectory_segment_distance", &trajectory_segment_distance_pybind11<double>, py::return_value_policy::move,
          py::arg("demos"));
    m.def("trajectory_mod", &trajectory_mod<double>);
    m.def("get_pairwise_trajectory_distance_matrix_simple", &get_pairwise_trajectory_distance_matrix_simple<double>);
    m.def("compute_distance_matrix_from_trajectories", &compute_distance_matrix_from_trajectories_pybind11<double>, py::return_value_policy::move,
          py::arg("demos"));
}
