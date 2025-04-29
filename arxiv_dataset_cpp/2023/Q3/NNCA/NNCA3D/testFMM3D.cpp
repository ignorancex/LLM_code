#include "kernel.hpp"
#include "ACA.hpp"
#include "FMM3DTree.hpp"

int main(int argc, char* argv[]) {
	int cubeRootN		=	atoi(argv[1]);
	int nParticlesInLeafAlong1D	=	atoi(argv[2]); // assuming the particles are located at tensor product chebyNodes
	double L			=	atof(argv[3]);
	int TOL_POW = atoi(argv[4]);
	int Qchoice = atoi(argv[5]);
	int yesToUniformDistribution = atoi(argv[6]);
	double start, end;
	int nLevels		=	ceil(3*log(double(cubeRootN)/nParticlesInLeafAlong1D)/log(8));
	// std::cout << "nLevels: " << nLevels << std::endl;
	/////////////////////////////////////////////////////////////////////////
	start	=	omp_get_wtime();
	std::vector<pts3D> particles;
	userkernel* mykernel		=	new userkernel(particles, Qchoice);
	FMM3DTree<userkernel>* A	=	new FMM3DTree<userkernel>(mykernel, cubeRootN, nLevels, nParticlesInLeafAlong1D, L, TOL_POW);

	double h = 2.0*L/nParticlesInLeafAlong1D;
	if (yesToUniformDistribution == 1) {
		A->set_Uniform_Nodes(h);
	}
	else {
		A->set_Standard_Cheb_Nodes();
	}
	A->createTree();
	A->assign_Tree_Interactions();
	A->assign_Center_Location();
	A->assignChargeLocations();
	A->assignNonLeafChargeLocations();
	end		=	omp_get_wtime();
	double timeCreateTree	=	(end-start);

	std::cout << std::endl << "Number of particles is: " << A->N << std::endl;
	// std::cout << std::endl << "Time taken to create the tree is: " << timeCreateTree << std::endl;
	/////////////////////////////////////////////////////////////////////////
	start	=	omp_get_wtime();
	A->getNodes();
	A->assemble_M2L();
	end		=	omp_get_wtime();

	double timeAssemble =	(end-start);
	std::cout << std::endl << "Time taken to assemble is: " << timeAssemble << std::endl;
	/////////////////////////////////////////////////////////////////////////
	int N = A->N;
	// to speed up the computation of finding error in the solution, b is considered as a vector of ones and zeros. The location of ones is defined as below
	// Eigen::VectorXd b=Eigen::VectorXd::Ones(N);
	Eigen::VectorXd b=Eigen::VectorXd::Zero(N);
  int n = N/500;
  srand(time(NULL));
  std::set<int> s;
  while(s.size() < n) {
    int index	=	rand()%N;
    s.insert(index);
  }
  std::set<int>::iterator it;
  for (it = s.begin(); it != s.end(); it++) {
    b(*it) = 1.0;
  }

	A->assignCharges(b);

	start	=	omp_get_wtime();
	A->evaluate_M2M();
	A->evaluate_M2L();
	A->evaluate_L2L();
	A->evaluate_NearField();
	Eigen::VectorXd AFMM_Ab;
	A->collectPotential(AFMM_Ab);
	A->reorder(AFMM_Ab);

	end		=	omp_get_wtime();
	double timeMatVecProduct = end-start;
	std::cout << std::endl << "Time taken to do Mat-Vec product is: " << timeMatVecProduct << std::endl << std::endl;
	//////////////////
	double err;
  std::string fname; //uncomment this
  Eigen::VectorXd true_Ab = Eigen::VectorXd::Zero(N);
  for (it = s.begin(); it != s.end(); it++) {
    true_Ab = true_Ab + mykernel->getCol(*it);
  }

	// Eigen::VectorXd true_Ab = Eigen::VectorXd::Zero(N);
	// #pragma omp parallel for
	// for (size_t i = 0; i < N; i++) {
	// 	#pragma omp parallel for
	// 	for (size_t j = 0; j < N; j++) {
	// 		true_Ab(i) += A->K->getMatrixEntry(i,j)*b(j);
	// 	}
	// }
	double sum;
	A->findMemory2(sum);
	std::cout << "Memory in GB: " << sum/8*pow(10,-9) << std::endl << std::endl;
	std::cout << "CR: " << double(sum)/N/N << std::endl << std::endl;
	std::cout << "max rank: " << A->getAvgRank() << std::endl << std::endl;

	err = (true_Ab - AFMM_Ab).norm()/true_Ab.norm();
	std::cout << "Error in matVec: " << err << std::endl << std::endl;

	delete A;
	delete mykernel;
}
