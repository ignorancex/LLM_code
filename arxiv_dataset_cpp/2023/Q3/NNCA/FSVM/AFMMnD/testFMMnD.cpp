#include "AFMMnD.hpp"

int main(int argc, char* argv[]) {
	int ndim = atoi(argv[1]);
	int n		=	atoi(argv[2]); // n=pow(N,1/ndim)
	int nParticlesInLeafAlong1D	=	atoi(argv[3]); // assuming the particles are located at tensor product chebyNodes
	double L			=	atof(argv[4]);
	int TOL_POW = atoi(argv[5]);
	int Qchoice = atoi(argv[6]);
	double start, end;
	// int nLevels		=	ceil(log(double(n)/nParticlesInLeafAlong1D)/log(2))+1;
	int nLevels		=	3;
	std::cout << "nLevels: " << nLevels << std::endl;
	int N = pow(n,ndim);
	std::vector<AptsnD> points = give_random_Nodes(N, ndim);
	/////////////////////////////////////////////////////////////////////////
	start	=	omp_get_wtime();
	AFMMnD* afmmnd = new AFMMnD(ndim, N, nParticlesInLeafAlong1D, L, TOL_POW, Qchoice, nLevels, points);
	end		=	omp_get_wtime();

	double timeCreateTree	=	(end-start);
	std::cout << std::endl << "Number of particles is: " << N << std::endl;
	std::cout << std::endl << "Time taken to initialise is: " << timeCreateTree << std::endl;
	/////////////////////////////////////////////////////////////////////////

	start	=	omp_get_wtime();
	afmmnd->assembleAFMMnD();
	end		=	omp_get_wtime();

	double timeAssemble =	(end-start);
	std::cout << std::endl << "Time taken to assemble is: " << timeAssemble << std::endl;
	/////////////////////////////////////////////////////////////////////////
	// Eigen::VectorXd b=Eigen::VectorXd::Ones(N);

	Eigen::VectorXd b=Eigen::VectorXd::Zero(N);
	int factN = N/500;
  srand(time(NULL));
  std::set<int> s;
  while(s.size() < factN) {
    int index	=	rand()%N;
    s.insert(index);
  }
  std::set<int>::iterator it;
  for (it = s.begin(); it != s.end(); it++) {
    b(*it) = 1.0;
  }

	Eigen::VectorXd AFMM_Ab;
	start	=	omp_get_wtime();
	AFMM_Ab = afmmnd->matVecProduct(b);
	end		=	omp_get_wtime();

	double timeMatVecProduct = end-start;
	std::cout << std::endl << "Time taken to do Mat-Vec product is: " << timeMatVecProduct << std::endl;
	std::cout << std::endl << "Total Time taken (assemble + MV): " << timeAssemble + timeMatVecProduct << std::endl;
//////////////////
	double err;

	Eigen::VectorXd true_Ab = Eigen::VectorXd::Zero(N);

  for (it = s.begin(); it != s.end(); it++) {
		Eigen::VectorXd temp = afmmnd->mykernel->getCol(*it);
    true_Ab = true_Ab + temp;
  }

	err = (true_Ab - AFMM_Ab).norm()/true_Ab.norm();
	std::cout << std::endl << "err: " << err << std::endl;

	double sum;
	delete afmmnd;
}
