#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <chrono>
// #include <string>
#include <map>

#include <fstream>
#include <iostream>

#include "dexpSkew.hpp"
#include "dexpPade.hpp"
#include "dexpSkewEig.hpp"
#include "dlogSkewPade.hpp"

#include "basics.hpp"

#define num_main_option 5

int main(int argc, char *argv[])
{
	string def_val[num_main_option] = {"forward", "100", "100", "9527", "../figures/dexp"};
	string options[num_main_option] = {"-task", "-dim", "-loop", "-seed", "-file"};

	auto paras = read_paras(argc, argv, num_main_option, def_val, options);

	INTE_TYPE loops = stoi(paras["-loop"]);
	INTE_TYPE seed = stoi(paras["-seed"]);
	INTE_TYPE dim_end = stoi(paras["-dim"]);
	INTE_TYPE dim_beg = 3;

	INTE_TYPE record_cnt = 0;

	if (paras["-task"] == "forward")
		record_cnt = 14;
	else
		record_cnt = 16;

	INTE_TYPE record_ind = 0;
	INTE_TYPE dim;
	INTE_TYPE dlog_pade_scale = 0;

	ColMat<REAL_TYPE> Record(dim_end - dim_beg + 1, record_cnt);
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
			auto MatN = SkewSymmMat(dim);

			MatA.init_low_vec();
			MatM.init_low_vec();
			MatN.init_low_vec();

			MatA.Rand(rng, -1.0, 1.0);
			MatA.low2upp();
			MatM.Rand(rng, -1.0, 1.0);
			MatM.low2upp();

			auto MatQ = ColMat<REAL_TYPE>(dim, dim);
			auto MatY = ColMat<REAL_TYPE>(dim, dim);

			MatQ.Zero();
			MatY.Zero();

			auto ssf = SkewSchurFactor(dim);
			auto dexp_skew = dexpSkewSymmPara(dim);
			auto dexp_pade = dexpPadeApprox(dim);
			auto dexp_cmpx = dexpSkewEigenPara(dim);
			auto dlog_pade = dlogSkewPadeApprox(dim);

			record_ind = 0;

			if (paras["-task"] == "forward")
			{

				// SkewSymm

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { ssf.Factor(MatA.v, dim); ssf.Explict_Vector(); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { dexp_skew.Parameter(ssf); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { dexp_skew.Forward(MatN, MatM); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { ssf.Exponential(MatQ.v, dim); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, MatQ.v, dim, MatN.v, dim, 0.0, MatY.v, dim); }, {});

				// Pade-order(3/3)

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { dexp_pade.Expm(MatQ.v, dim, MatA.v, dim, 3); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { dexp_pade.Dexp(MatY.v, dim, MatM.v, dim); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, dim, dim, dim, 1.0, MatQ.v, dim, MatY.v, dim, 0.0, MatN.v, dim); }, {});

				// Pade-order(13/13)

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { dexp_pade.Expm(MatQ.v, dim, MatA.v, dim, 13); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { dexp_pade.Dexp(MatY.v, dim, MatM.v, dim); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, dim, dim, dim, 1.0, MatQ.v, dim, MatY.v, dim, 0.0, MatN.v, dim); }, {});

				// Semisimple

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { ssf.Factor(MatA.v, dim); ssf.Explict_Vector(); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { dexp_cmpx.Assign(ssf); dexp_cmpx.Parameter(); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { dexp_cmpx.Dexp(MatN.v, dim, MatM.v, dim); }, {});
			}
			else
			{
				ssf.Factor(MatA.v, dim);
				ssf.Explict_Vector();
				ssf.Exponential(MatQ.v, dim);
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, MatQ.v, dim, MatM.v, dim, 0.0, MatY.v, dim);

				MatQ.Zero();
				ssf.R.Zero();
				ssf.A.Zero();

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { ssf.Factor(MatA.v, dim); ssf.Explict_Vector(); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { ssf.Exponential(MatQ.v, dim); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { dexp_skew.Parameter(ssf); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, {
					// cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, dim, dim, dim, 1.0, MatQ.v, dim, MatY.v, dim, 0.0, MatM.v, dim);
					skewblas_mmlt(CblasTrans, CblasNoTrans, dim, dim, dim, 1.0, MatQ.v, dim, MatY.v, dim, 0.0, MatM.v, dim);
					MatM.low2upp(); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { dexp_skew.Inverse(MatN, MatM); }, {});

				// Pade-order(1/1) Formula

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { 
					ssf.Factor(MatA.v, dim); 
					ssf.Explict_Vector(); 
					dlog_pade.Parameter(ssf);
					dlog_pade_scale = ceil(log2(MatA.Norm1() / _EXPM_PADE_APPROX_BOUNDS[12]));
					dlog_pade_scale = dlog_pade_scale < 0 ? 0 : dlog_pade_scale;
					dlog_pade._set_orders(1, dlog_pade_scale); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { ssf.Exponential(MatQ.v, dim); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, MatQ.v, dim, MatN.v, dim, 0.0, MatY.v, dim); }, {});

				// std::printf("dlog Scaling Order: %lld\n\n", dlog_pade_scale);

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { dlog_pade.Dlog(MatN.v, dim, MatY.v, dim); }, {});

				// Pade-order(7/7), Formula

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { 
					ssf.Factor(MatA.v, dim); 
					ssf.Explict_Vector(); 
					dlog_pade.Parameter(ssf);
					dlog_pade_scale = ceil(log2(MatA.Norm1() / _EXPM_PADE_APPROX_BOUNDS[12]));
					dlog_pade_scale = dlog_pade_scale < 0 ? 0 : dlog_pade_scale;
					dlog_pade._set_orders(7, dlog_pade_scale); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { ssf.Exponential(MatQ.v, dim); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, MatQ.v, dim, MatN.v, dim, 0.0, MatY.v, dim); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { dlog_pade.Dlog(MatN.v, dim, MatY.v, dim); }, {});

				// Semisimple, Formula

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { ssf.Factor(MatA.v, dim); ssf.Explict_Vector(); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { dexp_cmpx.Assign(ssf); dexp_cmpx.Parameter(); }, {});

				TIME_AND_AVG(tlv, cur_loop, Record(dim_ind, record_ind++), {}, { dexp_cmpx.DexpInv(MatN.v, dim, MatM.v, dim); }, {});
			}
		}
	}

	char filename[100];
	if (paras["-task"] == "forward")
		sprintf(filename, "%s/dexp_elapsed_time_s%s.txt", paras["-file"].c_str(), paras["-seed"].c_str());
	else
		sprintf(filename, "%s/dlog_elapsed_time_s%s.txt", paras["-file"].c_str(), paras["-seed"].c_str());

	std::ofstream fout;
	fout.open(filename, std::ios_base::binary);

	printf("\rAll tests are finished! Writing to file %s.\n", filename);

	REAL_TYPE dimension;
	REAL_TYPE elapsed_t;

	for (int j = 0; j < dim_end - dim_beg + 1; j++)
	{
		dimension = dim_beg + j;
		fout.write(reinterpret_cast<char *>(&dimension), sizeof(REAL_TYPE));
		for (int i = 0; i < record_cnt; i++)
		{
			elapsed_t = Record(j, i);
			fout.write(reinterpret_cast<char *>(&elapsed_t), sizeof(REAL_TYPE));
		}
	}

	fout.close();

	printf("Sucessfully written to the file %s\nRoutine exits.\n", filename);
};