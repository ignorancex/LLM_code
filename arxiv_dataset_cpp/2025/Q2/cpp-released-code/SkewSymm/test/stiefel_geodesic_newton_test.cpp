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

#define num_main_option 6

using std::map;
using std::stoi;
using std::string;

typedef std::chrono::steady_clock::time_point my_clock;

map<string, string>
read_paras(int argc, char *argv[], int key_size, string *def_val, string *options);

int main(int argc, char *argv[])
{
	// std::random_device dev;
	// INTE_TYPE random_seed = 9527;
	// std::seed_seq seed(random_seed);
	// std::mt19937 rng(seed);

	string def_val[num_main_option] = {"20", "10", "100", "100", "100", "../figures/dexp"};
	string options[num_main_option] = {"-row", "-col", "-iter", "-gmres-restart", "-gmres-maxiter", "-file"};

	auto paras = read_paras(argc, argv, num_main_option, def_val, options);

	// INTE_TYPE seed = stoi(paras["-seed"]);
	INTE_TYPE n = stoi(paras["-row"]);
	INTE_TYPE p = stoi(paras["-col"]);
	INTE_TYPE MaxIter = stoi(paras["-iter"]);
	INTE_TYPE GMRES_MaxIter = stoi(paras["-gmres-maxiter"]);
	INTE_TYPE GMRES_Restart = stoi(paras["-gmres-restart"]);

	INTE_TYPE m = n - p;

	INTE_TYPE display = 5;

	std::default_random_engine rng;
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
	MatS.Skew();
	MatS.SchurAngular_SkewSymm();
	scal(1.1 * M_PI / abs(MatS.saf.a[0]), MatS);
	MatS.mat2vec();
	MatS.vec2mat();

	printf("Matrix S:\n");
	MatS.printf();

	auto MateS = SpecOrthMat(n);

	MatS.SchurAngular_SkewSymm();
	MatS.saf.compute_SpecOrthMat(MateS);
	MateS.printf("Matrix exp(S):\n");

	auto HHF = HouseholderQrFactor(n, p);
	auto MatQ = SpecOrthMat(n);

	HHF.QR(MateS.v);
	HHF.Q(MatQ);

	INTE_TYPE flip_cnt = p;
	// n x p QR requries p Householder reflectors, which produce p flip;

	for (INTE_TYPE col_ind = 0; col_ind < p; col_ind++)
		if (HHF.H(col_ind, col_ind) < 0) // R[i, i] in QR = exp(S) is negative, then it is -1.
		{
			my_dscal(n, -1.0, MatQ.col(col_ind)); // To recover exp(S) from Q, flip the respective column
			flip_cnt++;
		}

	if (flip_cnt % 2) // odd number of flips, the resulting Q is not in SO_n, flip the p+1 column.
		my_dscal(n, -1.0, MatQ.col(p), 1);

	MatQ.printf("Matrix Q with Q I_{n, p} = exp(S) I_{n,p}:\n");

	auto point = Stiefel_Point_Newton(n, p);
	INTE_TYPE gmres_dim = m * (m - 1) / 2;
	INTE_TYPE gmres_MaxIter = gmres_dim < GMRES_MaxIter ? gmres_dim : GMRES_MaxIter;
	INTE_TYPE gmres_restart = gmres_dim < GMRES_Restart ? gmres_dim : GMRES_Restart;
	INTE_TYPE gmres_iter = 0;
	REAL_TYPE gmres_RelTol = 1e-5;
	REAL_TYPE gmres_RelErr = 1;
	REAL_TYPE gmres_MaxNorm = 0.0;

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

	point.init(MatQ);
	point.Q2Saf(point.Q.v);
	point.Saf2S();
	curr_err = point.objval();
	armijo_slope = -2 * sqrt(curr_err);

	point.S.printf("Matrix S:\n");
	auto Delta = ColMat<REAL_TYPE>(n, n);

	for (INTE_TYPE iter = 0; iter < MaxIter; iter++)
	{
		printf("Iteration:\t %lld,\t AbsErr:\t %1.8f\t", iter, curr_err);
		if (curr_err < 1e-7)
			break;

		point.Saf2Dexp();

		// gmres_RelErr = gmres_RelTol;
		gmres_RelErr = (curr_err < 1e-1) ? curr_err / 10.0 : 1e-2;
		gmres_RelErr = (gmres_RelErr < gmres_RelTol) ? gmres_RelTol : gmres_RelErr;
		gmres_iter = gmres_MaxIter;
		armijo_scale = 1.0;
		point.S2Z(gmres_iter, gmres_restart, gmres_RelErr, gmres_MaxNorm, gmres_Hessenberg, gmres_KrylovBase, gmres_Workspace);
		point.Z2QZ();

		point.S.printf("Matrix S:\n");
		Delta.assign(point.S);

		point.QZ2M(0);
		point.Q2Saf(point.Work_MatMul);
		point.Saf2S();
		rsub(Delta, point.S);
		my_dscal(n * n, -pow(2.0, _StGeoNewton_ExpZ_Scale_Order), Delta.v);
		Delta.printf("(Dexp_S^{-1})[Z]:\n");

		for (INTE_TYPE z_ind = _StGeoNewton_ExpZ_Scale_Order; z_ind >= 0; z_ind--, armijo_scale *= 0.5)
		{
			point.QZ2M(z_ind);
			point.Q2Saf(point.Work_MatMul);
			point.Saf2S();

			next_err = point.objval();
			if (next_err < curr_err + armijo_slope * armijo_scale * armijo_upper || next_err < curr_err * armijo_lower)
				break;
		}
		curr_err = next_err;

		point.M2Q();
		armijo_slope = -2 * sqrt(curr_err);
	}
	printf("\n----------------------------------\n");

	MatS.printf("Generating MatS:\n");
	point.S.printf("Returning MatS:\n");
	MatQ.printf("Generating Q:\n");
	point.Q.printf("Returning MatS:\n");

	printf("Generating length:\t %2.2f,\t Returning length:\t %2.2f\n", sqrt(0.5 * normsqFro(MatS)), sqrt(0.5 * normsqFro(point.S)));

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