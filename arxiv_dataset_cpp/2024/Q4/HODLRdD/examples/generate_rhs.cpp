#include <cstdlib>
#include "HDD_matrix.hpp"
#include "HDD_clusters.hpp"
#include "kernel_function.hpp"
#include "myHeaders.hpp"
#include "points_dt.hpp"
#include "LowRank_matrix.hpp"
#include "user_kernel.hpp"


using namespace std;

int main(int argc, char *argv[])
{

    std::vector<ptsnD> *gridPoints;
    if (argc > 1)
        numPoints = atoi(argv[1]);
    else
        numPoints = 10;
    N = pow(numPoints, NDIM);
    // Bounding box of the kernel
    Eigen::VectorXd X(NDIM), Y(NDIM);
    for (int i = 0; i < NDIM; ++i)
    {
        X(i) = -1;
        Y(i) = 1;
    }
    Vec x_test, rhs, rhs_;
    std::vector<size_t> v3;
    for (size_t i = 0; i < N; i++)
        v3.push_back(i);
    kernel_4d_test *ker = new kernel_4d_test();
    ker->get_points(gridPoints);
    kernel_function<kernel_4d_test> *kernelfunc = new kernel_function<kernel_4d_test>(ker);
    //HODLRdD_matrix Kmat = HODLRdD_matrix(ker, gridPoints, X, Y);
    x_test  = Vec::Random(N, 1);
    rhs     = Vec::Zero(N, 1);
    rhs_    = Vec::Zero(N, 1);
    double start;
    std::string x_file_name = data_directory + "x_1overR2_" + std::to_string(N) + ".bin";
    std::string rhs_file_name = data_directory + "rhs_1overR2_" + std::to_string(N) + ".bin";
    storedata::save_vec(x_file_name, x_test);
    start = omp_get_wtime();
#pragma omp parallel num_threads(nThreads) shared(x_test, v3, kernelfunc, rhs_, N)
    {
#pragma omp for schedule(dynamic, 10)
        for (size_t i = 0; i < N; i++)
            rhs_(i) = kernelfunc->getRow(i, v3).dot(x_test);
    }
    start = omp_get_wtime() - start;
    std::cout << "Time for parallel matvec.. "<< N << " " << start << std::endl;
    storedata::save_vec(rhs_file_name, rhs_);
    return 0;
}

/* Test case for normal matrix vector product to work as fine as the parallel version

    x_test = Vec::Zero(N);
    rhs_ = Vec::Zero(N);

    x_test = storedata::load_vec(x_file_name);
    rhs_ = storedata::load_vec(rhs_file_name);

    start = omp_get_wtime();
    rhs = kernelfunc->getMatrix(v3, v3) * x_test;
    start = omp_get_wtime() - start;
    std::cout << "Time for normal .. " << start << std::endl;
    std::cout << "Relative Error in matvec ... " << Vec_ops::relative_error(rhs, rhs_) << std::endl;

*/
