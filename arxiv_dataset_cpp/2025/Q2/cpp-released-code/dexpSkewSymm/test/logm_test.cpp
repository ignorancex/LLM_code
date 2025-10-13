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
	string def_val[num_main_option] = {"100", "100", "9527", "../figures/dexp"};
	string options[num_main_option] = {"-dim", "-loop", "-seed", "-file"};

	auto paras = read_paras(argc, argv, num_main_option, def_val, options);

	INTE_TYPE loops = stoi(paras["-loop"]);
	INTE_TYPE seed = stoi(paras["-seed"]);
	INTE_TYPE dim_end = stoi(paras["-dim"]);
	INTE_TYPE dim_beg = 3;

	INTE_TYPE record_cnt = 3;

	INTE_TYPE record_ind = 0;
	INTE_TYPE dim;
	INTE_TYPE dlog_pade_scale = 0;

	ColMat<REAL_TYPE> Record(dim_end - dim_beg + 1, record_cnt);
	ArrVec<REAL_TYPE> Error(dim_end - dim_beg + 1);

	ArrVec<REAL_TYPE> TimeLoops(loops);
	REAL_TYPE *tlv = TimeLoops.v;

	std::default_random_engine rng;
	rng.seed(seed);
	std::uniform_real_distribution<> interval(-1.0, 1.0);

	for (INTE_TYPE outerloop = 0; outerloop < 2; outerloop++)
	{
		INTE_TYPE cur_loop = outerloop == 0 ? 1 : loops;
		if (outerloop)
			std::printf("\nPreheat finished! Experiments have started!\n");
		else
			std::printf("Preheating...");

		for (INTE_TYPE dim_ind = 0; dim_ind < dim_end - dim_beg + 1; dim_ind++)
		{
			dim = dim_beg + dim_ind;
			if (outerloop)
				printf("Testing Matrices of dimension\t %lld \tx\t %lld...\r", dim, dim);

			// Initialize the Matrices.
			auto MatA = SkewSymmMat(dim);
			auto MatM = SkewSymmMat(dim);
			MatA.init_low_vec();
			MatM.init_low_vec();

			MatA.Rand(rng, -1.0, 1.0);
			MatA.low2upp();

			auto MatQ = ColMat<REAL_TYPE>(dim, dim);
			auto MatN = ColMat<REAL_TYPE>(dim, dim);

			auto A_ssf = SkewSchurFactor(dim);
			auto Q_ssf = SkewSchurFactor(dim);

			A_ssf.Factor_SkewSymm(MatA.v, dim);
			A_ssf.Explict_Vector();
			A_ssf.Exponential(MatQ.v, dim);

			record_ind = 0;

			TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++),
						 {},
						 {
							 Q_ssf.H.Skew_Assign(MatQ.v, dim);
							 Q_ssf.SchurAngular_SkewSymm();
						 },
						 {});

			TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++),
						 {}, {
							 Q_ssf.Explict_Vector();
							 Q_ssf._principal_angles(MatQ.v, dim);
						 },
						 {});

			TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++),
						 {}, {
							 Q_ssf.GetSkewSymm(MatM.v, dim);
							 MatM.low2upp();
						 },
						 {});

			// Error accessment :

			if (!outerloop)
			{
				A_ssf.Factor_SkewSymm(MatM.v, dim);
				A_ssf.Explict_Vector();
				A_ssf.Exponential(MatN.v, dim);
				cblas_daxpy(dim * dim, -1, MatQ.v, 1, MatN.v, 1);
				Error[dim_ind] = cblas_dnrm2(dim * dim, MatN.v, 1);

				if (dim == 10)
				{
					printf("\nExample at %lld x %lld:\n", dim, dim);

					MatQ.printf("MatQ:\n", "%1.8f\t");
					Q_ssf.H.Skew_Assign(MatQ.v, dim);
					Q_ssf.SchurAngular_SkewSymm();
					Q_ssf.Explict_Vector();

					Q_ssf.R.printf("Schur Vectors:\n", "%1.8f\t");
					Q_ssf.A.printf("Sine Values of the Principal Angles in Q:\n");

					Q_ssf._principal_angles(MatQ.v, dim);
					Q_ssf.A.printf("Principal Angles in Q:\n");

					Q_ssf.GetSkewSymm(MatM.v, dim);
					MatM.low2upp();
					MatM.printf("Principal Logarithm logQ:\n", "%1.8f\t");
				}
			}
		}
	}

	char filename[100];

	sprintf(filename, "%s/log_SpecOrth_s%s.txt", paras["-file"].c_str(), paras["-seed"].c_str());

	std::ofstream fout;
	fout.open(filename, std::ios_base::binary);

	printf("\rAll tests are finished! Writing to file %s.\n", filename);

	REAL_TYPE dimension;
	REAL_TYPE writebuff;

	for (int j = 0; j < dim_end - dim_beg + 1; j++)
	{
		dimension = dim_beg + j;
		fout.write(reinterpret_cast<char *>(&dimension), sizeof(REAL_TYPE));
		for (int i = 0; i < record_cnt; i++)
		{
			writebuff = Record(j, i);
			fout.write(reinterpret_cast<char *>(&writebuff), sizeof(REAL_TYPE));
		}
		writebuff = Error[j];
		fout.write(reinterpret_cast<char *>(&writebuff), sizeof(REAL_TYPE));
	}

	fout.close();

	printf("Sucessfully written to the file %s\nRoutine exits.\n", filename);
};