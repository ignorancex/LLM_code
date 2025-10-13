#include <cstdio>
#include <cstdlib>
#include <random>
#include <chrono>
#include <map>

#include "colMajMat.hpp"
#include "skewSymmMat.hpp"
#include "dgemm.hpp"
#include "hhQrFac.hpp"
#include "stiefelGeoBCH.hpp"
#include "stiefelGeoNewton.hpp"

#define num_main_option 11

using std::map;
using std::stof;
using std::stoi;
using std::string;

typedef std::chrono::steady_clock::time_point my_clock;

map<string, string>
read_paras(int argc, char *argv[], int key_size, string *def_val, string *options);

int main(int argc, char *argv[])
{

	string def_val[num_main_option] = {"20", "10", "100", "100", "100", "0.1", "4.0", "500", "10", "../figures/dexp", "9527"};
	string options[num_main_option] = {"-row", "-col", "-iter", "-gmres-restart", "-gmres-maxiter", "-scale-beg", "-scale-end", "-scale-num", "-loop", "-file", "-seed"};

	auto paras = read_paras(argc, argv, num_main_option, def_val, options);

	INTE_TYPE seed = stoi(paras["-seed"]);
	INTE_TYPE n = stoi(paras["-row"]);
	INTE_TYPE p = stoi(paras["-col"]);
	INTE_TYPE MaxIter = stoi(paras["-iter"]);
	INTE_TYPE m = n - p;
	INTE_TYPE loop = stoi(paras["-loop"]);

	INTE_TYPE GMRES_MaxIter = stoi(paras["-gmres-maxiter"]);
	INTE_TYPE GMRES_Restart = stoi(paras["-gmres-restart"]);
	INTE_TYPE scale_num = stoi(paras["-scale-num"]);
	REAL_TYPE scale_beg = stof(paras["-scale-beg"]);
	REAL_TYPE scale_end = stof(paras["-scale-end"]);
	REAL_TYPE scale_inc = (scale_end - scale_beg) / (scale_num - 1);
	REAL_TYPE scale = 0.0;

	ColMat<REAL_TYPE> Record_BCH(scale_num, 4);
	ColMat<REAL_TYPE> Record_Newton(scale_num, 4);
	ColMat<INTE_TYPE> Record(loop + 1, 2);

	REAL_TYPE AbsTol = 1e-7;

	std::default_random_engine rng;
	rng.seed(seed);
	std::uniform_real_distribution<> interval(-1.0, 1.0);

	my_clock beg, end;
	INTE_TYPE symsyl_elapsed, gmres_elapsed;

	printf("Testing QR factorization of a matrix in size of \t %lld \tx\t %lld\n", n, n);

	auto MatS = SkewSymmMat(n);
	auto sl_col_tra = LowerTraversal(n, true, strict_lower_col_traversal);
	auto workspace_saf = Generate_Workspace(&MatS);
	MatS.set_low_vec(&sl_col_tra);
	MatS.initial_saf(workspace_saf);

	for (INTE_TYPE col_ind = 0; col_ind < p; col_ind++)
		for (INTE_TYPE row_ind = 0; row_ind < n; row_ind++)
			MatS(row_ind, col_ind) = interval(rng);

	auto MateS = SpecOrthMat(n);
	auto HHF = HouseholderQrFactor(n, p);
	auto MatQ = SpecOrthMat(n);

	INTE_TYPE flip_cnt = p;

	auto point_BCH = Stiefel_Point_BCH(n, p);
	INTE_TYPE BCH_iter = 0;
	REAL_TYPE err_BCH = 0.0;
	REAL_TYPE avg_elapsed_BCH = 0;

	auto point_Newton = Stiefel_Point_Newton(n, p);
	INTE_TYPE Newton_iter = 0;
	INTE_TYPE gmres_dim = m * (m - 1) / 2;
	INTE_TYPE gmres_MaxIter = gmres_dim < GMRES_MaxIter ? gmres_dim : GMRES_MaxIter;
	INTE_TYPE gmres_restart = gmres_dim < GMRES_Restart ? gmres_dim : GMRES_Restart;
	INTE_TYPE gmres_iter = 0;
	REAL_TYPE gmres_RelTol = 1e-7;
	REAL_TYPE gmres_RelErr = 1;
	REAL_TYPE gmres_MaxNorm = M_PI / 2;
	REAL_TYPE armijo_scale = 1.0;
	REAL_TYPE armijo_slope = 1.0;
	REAL_TYPE armijo_upper = 0.8;
	REAL_TYPE armijo_lower = 0.5;
	REAL_TYPE curr_err = 0.0;
	REAL_TYPE next_err = 0.0;
	auto gmres_Hessenberg = ColMat<REAL_TYPE>(gmres_restart + 1, gmres_restart);
	auto gmres_KrylovBase = ColMat<REAL_TYPE>(gmres_dim, gmres_restart + 1);
	auto gmres_Workspace = _GMRES_Work(gmres_dim, gmres_restart);
	gmres_Hessenberg.fast_col_access();
	gmres_KrylovBase.fast_col_access();
	INTE_TYPE newton_accumulate_nlog_call = 0;
	INTE_TYPE newton_accumulate_mexp_call = 0;
	INTE_TYPE gmres_accumulate_system_call = 0;
	INTE_TYPE gmres_maximum_krylov_dim = 0;
	REAL_TYPE avg_elapsed_Newton = 0;

	for (INTE_TYPE scale_ind = 0; scale_ind < scale_num; scale_ind++)
	{
		scale = scale_beg + scale_inc * scale_ind;
		Record_Newton(scale_ind, 0) = scale;
		MatS.Skew();
		MatS.SchurAngular_SkewSymm();
		scal(scale / abs(MatS.saf.a[0]), MatS);
		MatS.mat2vec();
		MatS.vec2mat();

		// MatS.printf("Matrix S:\n");

		MatS.SchurAngular_SkewSymm();
		MatS.saf.compute_SpecOrthMat(MateS);
		// MateS.printf("Matrix exp(S):\n");

		HHF.QR(MateS.v);
		HHF.Q(MatQ);
		flip_cnt = p;
		// n x p QR requries p Householder reflectors, which produce p flip;

		for (INTE_TYPE col_ind = 0; col_ind < p; col_ind++)
			if (HHF.H(col_ind, col_ind) < 0) // R[i, i] in QR = exp(S) is negative, then it is -1.
			{
				my_dscal(n, -1.0, MatQ.col(col_ind)); // To recover exp(S) from Q, flip the respective column
				flip_cnt++;
			}

		if (flip_cnt % 2) // odd number of flips, the resulting Q is not in SO_n, flip the p+1 column.
			my_dscal(n, -1.0, MatQ.col(p), 1);

		// MatQ.printf("Matrix Q with Q I_{n, p} = exp(S) I_{n,p}:\n");

		// printf("BCH method starts.\n");
		BCH_iter = 0;
		err_BCH = 0.0;

		for (INTE_TYPE loop_ind = 0; loop_ind < loop + 1; loop_ind++)
		{
			beg = std::chrono::steady_clock::now();
			point_BCH.Q.copy(MatQ);
			err_BCH = 0.0;
			BCH_iter = 0;
			for (INTE_TYPE iter = 0; iter < MaxIter; iter++)
			{
				BCH_iter += 1;
				point_BCH.Q2S();
				err_BCH = point_BCH.objval();
				// printf("Iteration:\t %lld,\t AbsErr:\t %1.8f\n", iter, err_BCH);
				if (err_BCH < AbsTol)
					break;
				point_BCH.S2Z();
				point_BCH.Z2Q();
			}
			end = std::chrono::steady_clock::now();
			Record(loop_ind, 0) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count();
		}

		avg_elapsed_BCH = 0;
		for (INTE_TYPE loop_ind = 1; loop_ind < loop + 1; loop_ind++)
			avg_elapsed_BCH += Record(loop_ind, 0);
		avg_elapsed_BCH /= loop * 1e6;

		printf("BCH method \t\t\t\t\t\texits at iteration:\t %lld\t in \t%2.5f(ms).\n", BCH_iter, avg_elapsed_BCH);
		printf("Total call to the matrix logarithm on n x n matrix (dominated by schur):\t %lld\n", BCH_iter);
		printf("Total call to the matrix exponental on m x m matrix (Scaling and Squaring):\t %lld\n", BCH_iter);
		printf("Total call symmetric Sylvester solver on m x m system (dominated by schur):\t %lld\n", BCH_iter);

		Record_BCH(scale_ind, 0) = scale;
		Record_BCH(scale_ind, 1) = BCH_iter;
		Record_BCH(scale_ind, 2) = avg_elapsed_BCH;
		Record_BCH(scale_ind, 3) = sqrt(0.5 * normsqFro(point_BCH.S));

		printf("\n----------------------------------\n");
		// printf("\nNewton method starts.\n");

		gmres_dim = m * (m - 1) / 2;
		gmres_MaxIter = gmres_dim < GMRES_MaxIter ? gmres_dim : GMRES_MaxIter;
		gmres_restart = gmres_dim < GMRES_Restart ? gmres_dim : GMRES_Restart;
		gmres_iter = 0;
		gmres_RelTol = 1e-5;
		gmres_RelErr = 1;
		gmres_MaxNorm = 0.0;

		armijo_scale = 1.0;
		armijo_slope = 1.0;
		armijo_upper = 0.8;
		armijo_lower = 0.5;

		curr_err = 0.0;
		next_err = 0.0;

		newton_accumulate_nlog_call = 0;
		newton_accumulate_mexp_call = 0;
		gmres_accumulate_system_call = 0;
		gmres_maximum_krylov_dim = 0;

		for (INTE_TYPE loop_ind = 0; loop_ind < loop + 1; loop_ind++)
		{
			beg = std::chrono::steady_clock::now();

			newton_accumulate_nlog_call = 0;
			newton_accumulate_mexp_call = 0;
			gmres_accumulate_system_call = 0;
			gmres_maximum_krylov_dim = 0;

			point_Newton.init(MatQ);
			point_Newton.Q2Saf(point_Newton.Q.v);
			point_Newton.Saf2S();
			curr_err = point_Newton.objval();
			armijo_slope = -2 * sqrt(curr_err);

			for (Newton_iter = 0; Newton_iter < MaxIter; Newton_iter++)
			{
				// printf("Iteration:\t %lld,\t AbsErr:\t %1.8f\t", iter, curr_err);
				if (curr_err < AbsTol)
					break;

				point_Newton.Saf2Dexp();

				// gmres_RelErr = gmres_RelTol;
				gmres_RelErr = (curr_err < 1e-1) ? curr_err / 10.0 : 1e-2;
				gmres_RelErr = (gmres_RelErr < gmres_RelTol) ? gmres_RelTol : gmres_RelErr;
				gmres_iter = gmres_MaxIter;
				armijo_scale = 1.0;
				point_Newton.S2Z(gmres_iter, gmres_restart, gmres_RelErr, gmres_MaxNorm, gmres_Hessenberg, gmres_KrylovBase, gmres_Workspace);
				gmres_accumulate_system_call += gmres_iter;
				if (gmres_iter > gmres_maximum_krylov_dim)
					gmres_maximum_krylov_dim = gmres_iter > gmres_restart ? gmres_restart : gmres_iter;

				point_Newton.Z2QZ();

				for (INTE_TYPE z_ind = _StGeoNewton_ExpZ_Scale_Order; z_ind >= 0; z_ind--, armijo_scale *= 0.5)
				{
					newton_accumulate_nlog_call += 1;
					newton_accumulate_mexp_call += 1;

					point_Newton.QZ2M(z_ind);
					point_Newton.Q2Saf(point_Newton.Work_MatMul);
					point_Newton.Saf2S();
					next_err = point_Newton.objval();
					if (next_err < curr_err + armijo_slope * armijo_scale * armijo_upper || next_err < curr_err * armijo_lower)
						break;
				}
				curr_err = next_err;

				point_Newton.M2Q();
				armijo_slope = -2 * sqrt(curr_err);
			}
			end = std::chrono::steady_clock::now();
			Record(loop_ind, 1) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count();
		}

		avg_elapsed_Newton = 0;
		for (INTE_TYPE loop_ind = 1; loop_ind < loop + 1; loop_ind++)
			avg_elapsed_Newton += Record(loop_ind, 1);
		avg_elapsed_Newton /= loop * 1e6;

		printf("Newton method \t\t\t\t\t\t exits at iteration:\t %lld\t in \t%2.5f(ms).\n", Newton_iter, avg_elapsed_Newton);

		printf("Total call to the matrix logarithm on n x n matrix (dominated by schur):\t %lld\n", newton_accumulate_nlog_call);
		printf("Total call to the matrix exponental on m x m matrix (Scaling and Squaring):\t %lld\n", newton_accumulate_mexp_call);
		printf("Total call to the linear system (dominated by Dexp_S^{-1} on n x n):\t\t %lld\n", gmres_accumulate_system_call);
		printf("Maximum dimension of the Krylov space generated in GMRES in R ^ {n}:\t\t %lld\n", gmres_maximum_krylov_dim);

		Record_Newton(scale_ind, 0) = scale;
		Record_Newton(scale_ind, 1) = Newton_iter;
		Record_Newton(scale_ind, 2) = avg_elapsed_Newton;
		Record_Newton(scale_ind, 3) = sqrt(0.5 * normsqFro(point_Newton.S));

		printf("\n----------------------------------\n");

		printf("Method\t Length\t Elapsed Time\t with scale %2.5f\n", scale);
		printf("BCH\t %2.2f\t %2.2f\n", sqrt(0.5 * normsqFro(point_BCH.S)), avg_elapsed_BCH);
		printf("Newton\t %2.2f\t %2.2f\n", sqrt(0.5 * normsqFro(point_Newton.S)), avg_elapsed_Newton);
		printf("Initial\t %2.2f\t", sqrt(0.5 * normsqFro(MatS)));
		printf("\n----------------------------------\n");
	}

	char filename[100];
	// string filename;
	sprintf(filename, "%s/stGeo_elasped_time_n%lld_p%lld.bin", paras["-file"].c_str(), n, p);

	std::ofstream fout;
	fout.open(filename, std::ios_base::binary);
	printf("\rAll tests are finished! Writing to file %s.\n", filename);

	for (INTE_TYPE Record_Col_Ind = 0; Record_Col_Ind < 4; Record_Col_Ind++)
		for (INTE_TYPE scale_ind = 0; scale_ind < scale_num; scale_ind++)
			fout.write(reinterpret_cast<char *>(&Record_BCH(scale_ind, Record_Col_Ind)), sizeof(REAL_TYPE));

	for (INTE_TYPE Record_Col_Ind = 0; Record_Col_Ind < 4; Record_Col_Ind++)
		for (INTE_TYPE scale_ind = 0; scale_ind < scale_num; scale_ind++)
			fout.write(reinterpret_cast<char *>(&Record_Newton(scale_ind, Record_Col_Ind)), sizeof(REAL_TYPE));

	fout.close();

	printf("Sucessfully written to the file %s\nRoutine exits.\n", filename);

	delete[] workspace_saf;
	delete[] gmres_Workspace;
};

map<string, string> read_paras(int argc, char *argv[], int key_size, string *def_val, string *options)
{
	std::cout << "\nLoading parameters\n";
	for (int i = 1; i < argc; i++)
	{
		for (int j = 0; j < key_size; j++)
		{
			if (options[j].compare(argv[i]) == 0)
			// if (argv[i] == options[j] && i + 1 < argc && argv[i + 1][0] != '-')
			{
				def_val[j] = (string)argv[++i];
				printf("%s\t is set to value\t %s\n", options[j].c_str(), def_val[j].c_str());
				// std::cout << options[j] << "\t is set to value\t " << def_val[j] << "\n";
				break;
			}
		}
	}
	printf("\nAll parameters initialized as:\n");
	printf("------------------------------------------------------------\n");
	for (int j = 0; j < key_size; j++)
		printf("%s:\t %s\n", options[j].c_str(), def_val[j].c_str());
	printf("------------------------------------------------------------\n");

	map<string, string> paras;
	for (int i = 0; i < key_size; i++)
		paras[options[i]] = def_val[i];
	std::cout << "Parameters loaded!\n";
	return paras;
}