#include "kernel.hpp"
#include "ACA.hpp"
#include "FMM3DTree.hpp"
#include "gmres.hpp"

class AFMM3D {
public:
	FMM3DTree<userkernel>* A;
	AFMM3D(int cubeRootN, int nParticlesInLeafAlong1D, double L, int TOL_POW, int Qchoice) {
		std::vector<pts3D> particles;
		userkernel* mykernel		=	new userkernel(particles, Qchoice);
		int nLevels		=	ceil(3*log(double(cubeRootN)/nParticlesInLeafAlong1D)/log(8));
		A	=	new FMM3DTree<userkernel>(mykernel, cubeRootN, nLevels, nParticlesInLeafAlong1D, L, TOL_POW);
		double h = 2.0*L/cubeRootN; // each interval has 1 point

		A->set_Uniform_Nodes(h);
		particles = A->K->particles;
		A->K		=	new userkernel(particles, Qchoice);
		A->createTree();
		A->assign_Tree_Interactions();
		A->assign_Center_Location();
		A->assignChargeLocations();
		A->assignNonLeafChargeLocations();
	}
	void assemble() {
		A->getNodes();
		A->assemble_M2L();
	}
	void MatVecProduct(Eigen::VectorXd &b, Eigen::VectorXd &AFMM_Ab) {
		A->assignCharges(b);
		A->evaluate_M2M();
		A->evaluate_M2L();
		A->evaluate_L2L();
		A->evaluate_NearField();
		A->collectPotential(AFMM_Ab);
		A->reorder(AFMM_Ab);
	}
	~AFMM3D() {};
};

int main(int argc, char* argv[]) {
	int cubeRootN		=	atoi(argv[1]);
	int nParticlesInLeafAlong1D	=	atoi(argv[2]); // assuming the particles are located at tensor product chebyNodes
	double L			=	atof(argv[3]);
	int TOL_POW = atoi(argv[4]);
	int Qchoice = 17;
	double start, end;
	int nLevels		=	ceil(3*log(double(cubeRootN)/nParticlesInLeafAlong1D)/log(8));
	// std::cout << "nLevels: " << nLevels << std::endl;

	/////////////////////////////////////////////////////////////////////////
	start	=	omp_get_wtime();
	AFMM3D *H = new AFMM3D(cubeRootN, nParticlesInLeafAlong1D, L, TOL_POW, Qchoice);
	end		=	omp_get_wtime();
	double timeCreateTree	=	(end-start);

	std::cout << std::endl << "Number of particles is: " << H->A->N << std::endl;
	// std::cout << std::endl << "Time taken to create the tree is: " << timeCreateTree << std::endl;
	/////////////////////////////////////////////////////////////////////////
	start	=	omp_get_wtime();
	H->assemble();
	end		=	omp_get_wtime();

	double timeAFMM3DRepres =	(end-start);
	std::cout << std::endl << "Time taken to assemble AFMM3D representation is: " << timeAFMM3DRepres << std::endl << std::endl;

	/////////////////////////////////////////////////////////////////////////
	int N = H->A->N;
	/////////////////// defining x //////////////////
	// Eigen::VectorXd x=Eigen::VectorXd::Ones(N);
	// to speed up the computation of finding error in the solution, x_true is considered as a vector of ones and zeros. The location of ones is defined as below
	Eigen::VectorXd x=Eigen::VectorXd::Zero(N);
  int n = N/500; //randomly choosing n different indices where x is set to 1, x at the rest of the indices is set to 0
  srand(time(NULL));
  std::set<int> s;
  while(s.size() < n) {
    int index	=	rand()%N;
    s.insert(index);
  }
  std::set<int>::iterator it;
  for (it = s.begin(); it != s.end(); it++) {
    x(*it) = 1.0;
  }

	Eigen::VectorXd true_Ax = Eigen::VectorXd::Zero(N);
  for (it = s.begin(); it != s.end(); it++) {
		true_Ax = true_Ax + H->A->K->getCol(*it);
	}

	// #pragma omp parallel for
	// for (size_t i = 0; i < N; i++) {
	// 	#pragma omp parallel for
	// 	for (size_t j = 0; j < N; j++) {
	// 		true_Ax(i) += H->A->K->getMatrixEntry(i,j)*x(j);
	// 	}
	// }
	// err = (true_Ax - AFMM_Ab).norm()/true_Ax.norm();

	double sum;
	H->A->findMemory2(sum);
	std::cout << "Memory in GB: " << sum/8*pow(10,-9) << std::endl << std::endl;
	std::cout << "CR: " << double(sum)/N/N << std::endl << std::endl;
	std::cout << "max rank: " << H->A->getAvgRank() << std::endl << std::endl;

	///////////////////////////////////// solving system start ////////////////////////////////////
	int maxIterations = 400;
	double GMRES_threshold = 1e-10;
	double GMRES_residual;
	Vec HODLR3D_x;
	int noOfIterations;
	std::vector<double> e;
	classGMRES *G = new classGMRES();

	start	=	omp_get_wtime();
	G->gmres(H, true_Ax, maxIterations, GMRES_threshold, HODLR3D_x, GMRES_residual, noOfIterations, e);
	end		=	omp_get_wtime();
	double timeGMRES = end-start;
	std::cout << "time GMRES (solve): " << timeGMRES << std::endl << std::endl;
	std::cout << "GMRES residual err: " << GMRES_residual << std::endl << std::endl;
	std::cout << "GMRES no. of iterations: " << noOfIterations << std::endl << std::endl;
	Vec err = HODLR3D_x-x;
	std::cout << "relative forward error in solution: " << err.norm()/x.norm() << std::endl << std::endl;
	///////////////////////////////////// solving system done ////////////////////////////////////
	delete H;
}
