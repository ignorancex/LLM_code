#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <chrono>
// #include <string>
#include <map>

#include <fstream>
#include <iostream>

#include "dgemm.hpp"

#include "matOp.hpp"
#include "skewSymmMat.hpp"
#include "specOrthMat.hpp"
#include "expmpa.hpp"
#include "dexpSkewSymm.hpp"
#include "dexpSemiSimple.hpp"
#include "dexpPadeApprox.hpp"

#define num_main_option 2

using std::map;
using std::stoi;
using std::string;

map<string, string>
read_paras(int argc, char *argv[], int key_size, string *def_val, string *options);

int main(int argc, char *argv[])
{
    // std::random_device dev;
    // INTE_TYPE random_seed = 9527;
    // std::seed_seq seed(random_seed);
    // std::mt19937 rng(seed);

    string def_val[num_main_option] = {"20", "9527"};
    string options[num_main_option] = {"-dim", "-seed"};

    auto paras = read_paras(argc, argv, num_main_option, def_val, options);

    INTE_TYPE dim = stoi(paras["-dim"]);
    INTE_TYPE seed = stoi(paras["-seed"]);

    std::default_random_engine rng;
    rng.seed(seed);
    std::uniform_real_distribution<> interval(-1.0, 1.0);

    printf("Testing Matrices of dimension\t %lld \tx\t %lld...\n", dim, dim);

    // Initialize the Matrices.
    auto MatA = SkewSymmMat(dim);
    auto MatM = SkewSymmMat(dim);
    auto MatN = SkewSymmMat(dim);
    auto MatQ = ColMat<REAL_TYPE>(dim, dim);
    auto MatDelta = ColMat<REAL_TYPE>(dim, dim);

    auto work_saf = new REAL_TYPE[(MatA.d + 2) * MatA.d];

    auto low_blk_traversal = LowerTraversal(dim, true, strict_lower_blk_traversal);
    MatA.set_low_vec(&low_blk_traversal);
    MatM.set_low_vec(&low_blk_traversal);
    MatN.set_low_vec(&low_blk_traversal);

    // auto MateleA = SpecOrthMat(dim);

    // Assign values to the matrices.
    for (INTE_TYPE i = 0; i < MatA.lsize; i++)
    {
        MatA.lv[i] = interval(rng);
        // MatM.lv[i] = i + 1;
        MatM.lv[i] = interval(rng);
    }
    MatA.vec2mat();
    MatM.vec2mat();

    auto A_saf_real = SchurAngularFactor(dim);
    auto A_saf_cmpx = SchurAngularFactor(dim);
    auto A_spf_cmpx = SpectralFactor(dim);
    auto A_pade = matExpPadeApproximant(dim);

    A_saf_real.set_workspace(work_saf);
    A_saf_cmpx.set_workspace(work_saf);

    auto dexp_real = dexpSkewSymmPara(dim);
    auto dexp_cmpx = dexpSemiSimplePara(dim);
    auto dexp_pade = dexpPadeApproximant(dim);

    printf("Random skew symmetric matrix A:\n");
    for (INTE_TYPE row_ind = 0; row_ind < dim; row_ind++, printf("\n"))
        for (INTE_TYPE col_ind = 0; col_ind < dim; col_ind++)
            printf("%2.2f\t", MatA.ele(row_ind, col_ind));

    printf("Random perturbation M:\n");
    for (INTE_TYPE row_ind = 0; row_ind < dim; row_ind++, printf("\n"))
        for (INTE_TYPE col_ind = 0; col_ind < dim; col_ind++)
            printf("%2.2f\t", MatM.ele(row_ind, col_ind));

    SchurAngular_SkewSymm(A_saf_real, MatA);
    dexp_real.setupSAF(&A_saf_real);
    dexp_real.setupPara();
    dexpSkewSymm_forward(MatN, MatM, dexp_real);

    printf("Skew symmetric N in exp(A)N = Dexp_A[M], computed by the real formula:\n");
    MatN.printf();

    SchurAngular_SkewSymm(A_saf_cmpx, MatA);
    A_spf_cmpx.setSPF(A_saf_cmpx);
    dexp_cmpx.setSPF(&A_spf_cmpx);
    dexp_cmpx.setupPara();
    dexpSemiSimple(MatN, MatM, dexp_cmpx, true);

    printf("Skew symmetric N in exp(A)N = Dexp_A[M], computed by the complex formula:\n");
    MatN.printf();

    A_pade.setup_perturb(MatA);
    A_pade.expPadeApprox(2);
    A_pade.exp(MatQ);

    dexp_pade.setup_empa(&A_pade);
    dexp_pade.setup_rs();
    dexp_pade.setup_perturb(MatM);
    dexp_pade.dexp(MatDelta);
    my_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, dim, dim, dim, 1.0, MatQ.v, dim, MatDelta.v, dim, 0.0, MatN.v, dim);
    // my_dgemm(CblasTrans, CblasNoTrans, 1.0, MatQ, MatDelta, 0.0, MatN);

    printf("Skew symmetric N in exp(A)N = Dexp_A[M], computed by the Pade approximant:\n");
    MatN.printf();

    REAL_TYPE scale = 1e-7;
    cblas_daxpy(MatA.lsize, scale, MatM.lv, 1, MatA.lv, 1);
    cblas_daxpy(MatA.msize, scale, MatM.v, 1, MatA.v, 1);

    A_pade.setup_perturb(MatA);
    A_pade.expPadeApprox(2);
    A_pade.exp(MatDelta);

    cblas_daxpy(MatA.msize, -1.0, MatQ.v, 1, MatDelta.v, 1);
    cblas_dscal(MatA.msize, 1.0 / scale, MatDelta.v, 1);
    my_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, dim, dim, dim, 1.0, MatQ.v, dim, MatDelta.v, dim, 0.0, MatN.v, dim);

    printf("Difference operator N_e from (exp(A + e * S) - exp(A)) / e = exp(A)N_e:\n");
    MatN.printf();

    delete[] work_saf;
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