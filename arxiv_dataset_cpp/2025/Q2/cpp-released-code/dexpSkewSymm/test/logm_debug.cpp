#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <chrono>
// #include <string>
#include <map>

#include <fstream>
#include <iostream>

#include "skewSchFac.hpp"

#include "basics.hpp"

#define num_main_option 4

int main(int argc, char *argv[])
{

	INTE_TYPE dim = 10;
	INTE_TYPE seed = 9527;

	std::default_random_engine rng;
	rng.seed(seed);
	std::uniform_real_distribution<> interval(-1.0, 1.0);

	for (INTE_TYPE dim = 3; dim < 20; dim++)
	{
		printf("Debug task on the logarithm of a special orthogonal matrix:\n");
		printf("Dimension:\t %lld, Seed:\t %lld\n", dim, seed);
		auto MatA = SkewSymmMat(dim);
		auto MatM = SkewSymmMat(dim);
		MatA.init_low_vec();
		MatM.init_low_vec();

		MatA.Rand(rng, -1.0, 1.0);
		MatA.low2upp();

		MatA.printf("Random Skew Symmetric Matrix A:\n");

		auto MatQ = ColMat<REAL_TYPE>(dim, dim);
		auto MatN = ColMat<REAL_TYPE>(dim, dim);

		auto A_ssf = SkewSchurFactor(dim);
		auto Q_ssf = SkewSchurFactor(dim);

		A_ssf.Factor_SkewSymm(MatA.v, dim);
		A_ssf.Explict_Vector();
		A_ssf.Exponential(MatQ.v, dim);

		MatQ.printf("Special Orthogonal Matrix Q = exp(A):\n");

		Q_ssf.H.MatH.Assign(MatQ.v, dim);
		for (auto ind = 0; ind < dim; ind++)
			cblas_daxpby(dim, -0.5, MatQ.v + ind * dim, 1, 0.5, Q_ssf.H.MatH.v + ind, dim);

		Q_ssf.H.MatH.printf("Skew Symmetric Part skew(Q) = 0.5 * (Q - Q'):\n");

		Q_ssf.SchurAngular_SkewSymm();
		Q_ssf.Explict_Vector();

		Q_ssf.R.printf("Schur Vectors of Q:\n");
		Q_ssf.A.printf("Sine values of the Principal Angles in Q:\n");

		Q_ssf._principal_angles(MatQ.v, dim);
		Q_ssf.A.printf("Principal Angles in Q:\n");

		Q_ssf.GetSkewSymm(MatM.v, dim);
		MatM.low2upp();

		MatM.printf("Principal Logarithm X = logm(Q):\n");
	}
};