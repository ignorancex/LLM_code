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

#define num_main_option 5

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

	string def_val[num_main_option] = {"20", "10", "100", "9527", "../figures/dexp"};
	string options[num_main_option] = {"-row", "-col", "-loop", "-seed", "-file"};

	auto paras = read_paras(argc, argv, num_main_option, def_val, options);

	INTE_TYPE loops = stoi(paras["-loop"]);
	// INTE_TYPE seed = stoi(paras["-seed"]);
	INTE_TYPE n = stoi(paras["-row"]);
	INTE_TYPE p = stoi(paras["-col"]);
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
	scal(0.1 * M_PI / abs(MatS.saf.a[0]), MatS);
	MatS.mat2vec();
	MatS.vec2mat();

	printf("Matrix S:\n");
	MatS.printf();

	auto MateS = SpecOrthMat(n);

	MatS.SchurAngular_SkewSymm();
	MatS.saf.compute_SpecOrthMat(MateS);

	printf("Matrix exp(S):\n");
	MateS.printf();

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
	MatQ.printf();

	if (flip_cnt % 2) // odd number of flips, the resulting Q is not in SO_n, flip the p+1 column.
		my_dscal(n, -1.0, MatQ.col(p), 1);

	printf("Matrix Q with Q I_{n, p} = exp(S) I_{n,p}:\n");
	MatQ.printf();

	auto point = Stiefel_Point_BCH(n, p);

	point.Q.copy(MatQ);
	REAL_TYPE err = 0.0;

	for (INTE_TYPE iter = 0; iter < 2; iter++)
	{
		point.Q2S();

		printf("MatS:\n");
		point.S.printf();

		printf("MatA:\n");
		point.A.printf();

		printf("MatB:\n");
		point.B.printf();

		printf("MatC:\n");
		point.C.printf();

		err = point.objval();
		printf("Iteration:\t %lld,\t AbsErr:\t %1.8f\n", iter, err);
		if (err < 1e-7)
			break;
		point.S2Z();

		printf("MatR:\n");
		point.R.printf();

		printf("MatZ:\n");
		point.Z.printf();

		point.Z2Q();
	}

	delete[] workspace_saf;
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