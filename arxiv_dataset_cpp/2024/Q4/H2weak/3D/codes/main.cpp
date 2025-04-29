#include "gmres.hpp"
int main(int argc, char *argv[])
{
    //  srand(time(NULL));
    //  rand();
    int cubeRootN = atoi(argv[1]);                  // Cube root of total particles, i.e., cbrt(N)
    int nParticlesInLeafAlong1D = atoi(argv[2]);    // Maximum number of partilces along 1D at leaf level, i.e, cbrt(n_max) as per paper.
    double L = atof(argv[3]);                       // Semi-length of the cluster
    int TOL_POW = atoi(argv[4]);                    // Tolerance of the ACA/NCA
    int Choice = atoi(argv[5]);                     // Integral equation = 0, RBF interpolation = 1
    int nLevels = ceil(3 * log(double(cubeRootN) / nParticlesInLeafAlong1D) / log(8));
    if (nLevels < 1)
    {
        std::cout << "Please give large cube root of N !!!" << std::endl;
        exit(0);
    }
    Vec b, Ab_true, sol;

    double start, end;

    start = omp_get_wtime();
    initialisation *p = new initialisation(cubeRootN, nLevels, nParticlesInLeafAlong1D, L, TOL_POW, Choice);
    p->tree_initialisation();
    end = omp_get_wtime();
    double elapsed_assem = end - start;

    b = Vec::Random(p->K->N);

    Ab_true = p->K->getTruePoten(b);

    // Initial vector GMRES
    Vec x = Vec::Ones(p->K->N);

    int TOL_GMRES = 10; // GMRES stopping tolerance.
    start = omp_get_wtime();
    iterative_solver *its = new iterative_solver(1000, TOL_GMRES);
    its->GMRES<initialisation>(p, x, Ab_true);
    end = omp_get_wtime();
    double elapsed_gmres = end - start;

#ifdef USE_nHODLRdD
    p->K->findMemory_n();
#endif

#ifdef USE_snHODLRdD
    p->K->findMemory_sn();
#endif

#ifdef USE_HODLRdD
    p->K->findMemory_nn();
#endif

#ifdef USE_nHODLRdD
    std::cout << "********** Summary of nHODLR3D (nested HODLRdD) accelerated GMRES to solve a system **********" << std::endl
              << std::endl
              << std::endl;
#endif
#ifdef USE_snHODLRdD
    std::cout << "********** Summary of s-nHODLR3D (semi-nested HODLRdD) accelerated GMRES to solve a system **********" << std::endl
              << std::endl
              << std::endl;
#endif
#ifdef USE_HODLRdD
    std::cout << "********** Summary of HODLR3D (non-nested HODLRdD) accelerated GMRES to solve a system **********" << std::endl
              << std::endl
              << std::endl;
#endif
    std::cout << "The number of particles taken: " << p->K->N << " and choice = " << (Choice == 0 ? "integral equation" : "RBF interpolation") << std::endl
              << std::endl;
    std::cout << "The maximum number of particles at leaf clusters: " << nParticlesInLeafAlong1D * nParticlesInLeafAlong1D * nParticlesInLeafAlong1D << std::endl
              << std::endl;
    std::cout << "Depth of the tree: " << nLevels << std::endl
              << std::endl;
    std::cout << "The final residual error is: " << its->resid << std::endl
              << std::endl;
    std::cout << "The number of iteration is: " << its->max_iter << std::endl
              << std::endl;
    std::cout << "Total Assembly time: " << elapsed_assem << "s\n"
              << std::endl;
    std::cout << "GMRES time: " << elapsed_gmres << "s\n"
              << std::endl;
    std::cout << "Storage (in GB): " << (p->K->memory) * 8 * pow(10.0, -9) << " GB\n"
              << std::endl;
    std::cout << "Compression ratio: " << (p->K->memory) / (1.0 * p->K->N * p->K->N) << "\n"
              << std::endl;
    std::cout << "The (norm-2) relative error in solution: " << (x - b).norm() / b.norm() << std::endl;
    std::cout << "===================================================================================================="
              << "\n"
              << std::endl;
    delete its;
    delete p;
    return 0;
}