#include "gmres.hpp"

int main(int argc, char *argv[])
{
   // srand(time(NULL));
   // rand();
   int N = atoi(argv[1]);                       // Total number of particles, i.e., N
   int nParticlesInLeaf = atoi(argv[2]);        // Maximum number of partilces at leaf level, i.e, n_max as per paper.
   int L = atoi(argv[3]);                       // Semi-length of the cluster
   int TOL_POW = atoi(argv[4]);                 // Tolerance of the ACA/NCA
   int Choice = atoi(argv[5]);                  // Integral equation = 0, RBF interpolation = 1

   int nLevels = ceil(log(double(N) / nParticlesInLeaf) / log(2));
   if (nLevels < 1)
   {
      std::cout << "Please give large sqrt root of N !!!" << std::endl;
      exit(0);
   }
   Vec b, Ab_true, sol;

   double start, end;

   start = omp_get_wtime();
   initialisation *p = new initialisation(N, nLevels, nParticlesInLeaf, L, TOL_POW, Choice);
   p->tree_initialisation();
   end = omp_get_wtime();
   double elapsed_assem = end - start;

   b = Vec::Random(p->K->N);

   Ab_true = p->K->getTruePoten(b); // True value

   // Initial vector of GMRES
   Vec x = Vec::Ones(p->K->N);

   int TOL_GMRES = 12; // GMRES stopping tolerance.
   start = omp_get_wtime();
   iterative_solver *its = new iterative_solver(500, TOL_GMRES);
   its->GMRES<initialisation>(p, x, Ab_true);
   end = omp_get_wtime();
   double elapsed_gmres = end - start;

#ifdef USE_nHODLRdD
   p->K->findMemory_n();
#endif

#ifdef USE_snHODLRdD
   p->K->findMemory_sn();
#endif

#ifdef USE_nHODLRdD
   std::cout << "********** Summary of HSS / HBS / nHODLR1D accelerated GMRES to solve a system **********" << std::endl
             << std::endl
             << std::endl;
#endif
#ifdef USE_snHODLRdD
   std::cout << "********** Summary of HODLR / HODLR1D accelerated GMRES to solve a system **********" << std::endl
             << std::endl
             << std::endl;
#endif
   std::cout << "The number of particles taken: " << p->K->N << " and choice =  " << (Choice == 0 ? "integral equation" : "RBF interpolation") << std::endl
             << std::endl;
   std::cout << "The maximum number of particles at leaf clusters: " << nParticlesInLeaf << std::endl
             << std::endl;
   std::cout << "Depth of the tree: " << nLevels << std::endl
             << std::endl;
   std::cout << "The final residual error is: " << its->resid << std::endl
             << std::endl;
   std::cout << "The total number of iteration is: " << its->max_iter << std::endl
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
   std::cout << "==========================================================================================="
             << "\n"
             << std::endl;
   delete its;
   delete p;
   return 0;
}