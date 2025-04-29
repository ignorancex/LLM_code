#include "kernel.hpp"
#include "ACA.hpp"
#include "FMM2DTree.hpp"

int main(int argc, char* argv[]) {
	int nLevels		=	atoi(argv[1]); // number of levels of quad-tree
	int nParticlesInLeafAlong1D	=	atoi(argv[2]); // assuming the particles are located at tensor product chebyNodes/uniform
	double L			=	atof(argv[3]); // half side length of square centered at origin
	int TOL_POW = atoi(argv[4]); //tolerance of ACA
	int yesToUniformDistribution = atoi(argv[5]);
	double start, end;

	/////////////////////////////////////////////////////////////////////////
	start	=	omp_get_wtime();
	std::vector<pts2D> particles_X, particles_Y; // these get defined in the FMM2DTree object
	userkernel* mykernel		=	new userkernel(particles_X, particles_Y);
	FMM2DTree<userkernel>* A	=	new FMM2DTree<userkernel>(mykernel, nLevels, nParticlesInLeafAlong1D, L, TOL_POW);

	if (yesToUniformDistribution == 1) {
		A->set_Uniform_Nodes();
	}
	else {
		A->set_Standard_Cheb_Nodes();
	}
	A->createTree();
	A->assign_Tree_Interactions();
	A->assign_Center_Location();
	A->assignLeafChargeLocations();
	end		=	omp_get_wtime();
	double timeCreateTree	=	(end-start);

	std::cout << std::endl << "Number of particles is: " << A->N << std::endl;
	// std::cout << std::endl << "Time taken to create the tree is: " << timeCreateTree << std::endl;
	/////////////////////////////////////////////////////////////////////////
	start	=	omp_get_wtime();
	A->getNodes();
	A->assemble_M2L();
	A->assemble_NearField();
	end		=	omp_get_wtime();

	double timeassemble=	(end-start);
	std::cout << std::endl << "Time taken to assemble is: " << timeassemble << std::endl;

	/////////////////////////////////////////////////////////////////////////
	int N = A->N;
	Eigen::VectorXd b = Eigen::VectorXd::Random(N); // the vector that needs to be applied to the matrix
	A->assignLeafCharges(b);

	start	=	omp_get_wtime();
	A->evaluate_M2M();
	A->evaluate_M2L();
	A->evaluate_L2L();
	A->evaluate_NearField();
	Eigen::VectorXd AFMM_Ab;
	A->collectPotential(AFMM_Ab);
	end		=	omp_get_wtime();
	double timeMatVecProduct=	(end-start);
	std::cout << std::endl << "Time taken to do Mat-Vec product is: " << timeMatVecProduct << std::endl;

	double sum;
	A->findMemory(sum);
	std::cout << std::endl << "Avg Rank is: " << A->getAvgRank() << std::endl;
	std::cout << std::endl << "Memory in GB is: " << sum/8*pow(10,-9) << std::endl << std::endl;

	//////////////////
	double err;
	start		=	omp_get_wtime();
	// Vec true_Ab = Afull*b;
	Eigen::VectorXd true_Ab = Eigen::VectorXd::Zero(N);
	#pragma omp parallel for collapse(2)
	for (size_t i = 0; i < N; i++) {
		// #pragma omp parallel for
		for (size_t j = 0; j < N; j++) {
			true_Ab(i) += A->K->getMatrixEntry(i,j)*b(j);
		}
	}
	end		=	omp_get_wtime();
	double exact_time =	(end-start);
	std::cout << "Time for direct MatVec: " << exact_time << std::endl << std::endl;
	std::cout << "Magnitude of Speed-Up: " << (exact_time / timeMatVecProduct) << std::endl << std::endl;

	err = (true_Ab - AFMM_Ab).norm()/true_Ab.norm();
	std::cout << "Error in the solution is: " << err << std::endl << std::endl;
}
