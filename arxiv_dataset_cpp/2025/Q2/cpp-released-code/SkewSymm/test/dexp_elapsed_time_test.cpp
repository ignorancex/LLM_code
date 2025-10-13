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

#define num_main_option 5

using std::map;
using std::stoi;
using std::string;

typedef std::chrono::steady_clock::time_point my_clock;

map<string, string>
read_paras(int argc, char *argv[], int key_size, string *def_val, string *options);

REAL_TYPE avg_time(REAL_TYPE *time, INTE_TYPE size);
REAL_TYPE min_time(REAL_TYPE *time, INTE_TYPE size);

int main(int argc, char *argv[])
{
    // std::random_device dev;
    // INTE_TYPE random_seed = 9527;
    // std::seed_seq seed(random_seed);
    // std::mt19937 rng(seed);

    string def_val[num_main_option] = {"forward", "100", "100", "9527", "../figures/dexp"};
    string options[num_main_option] = {"-task", "-dim", "-loop", "-seed", "-file"};

    auto paras = read_paras(argc, argv, num_main_option, def_val, options);

    INTE_TYPE loops = stoi(paras["-loop"]);
    INTE_TYPE seed = stoi(paras["-seed"]);
    INTE_TYPE dim_end = stoi(paras["-dim"]);

    INTE_TYPE methods = 3;
    INTE_TYPE dim_beg = 4;
    INTE_TYPE dim;
    INTE_TYPE elapsed;

    my_clock beg, end;

    REAL_TYPE ***Record_Time = new REAL_TYPE **[methods];
    for (int i = 0; i < methods; i++)
    {
        Record_Time[i] = new REAL_TYPE *[dim_end - dim_beg + 1];
        for (int j = 0; j < dim_end - dim_beg + 1; j++)
            Record_Time[i][j] = new REAL_TYPE[3];
    }
    REAL_TYPE *loop_time = new REAL_TYPE[loops];

    std::default_random_engine rng;
    rng.seed(seed);
    std::uniform_real_distribution<> interval(-1.0, 1.0);

    for (INTE_TYPE outerloop = 0; outerloop < 2; outerloop++)
    {
        for (INTE_TYPE dim_ind = 0; dim_ind < dim_end - dim_beg + 1; dim_ind++)
        {
            dim = dim_beg + dim_ind;
            printf("Testing Matrices of dimension\t %lld \tx\t %lld...\n", dim, dim);

            // Initialize the Matrices.
            auto MatA = SkewSymmMat(dim);
            auto MatM = SkewSymmMat(dim);
            auto MatN = SkewSymmMat(dim);
            auto MatQ = ColMat<REAL_TYPE>(dim, dim);
            auto MatDelta = ColMat<REAL_TYPE>(dim, dim);

            auto MatIA = ColMat<CMPX_TYPE>(dim, dim);
            auto MatZ = ColMat<CMPX_TYPE>(dim, dim);
            auto VecD = new REAL_TYPE[dim];
            auto VecE = new REAL_TYPE[dim - 1];
            auto VecTau = new CMPX_TYPE[dim - 1];

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
                MatM.lv[i] = i + 1;
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

            // +---------------------------------------------------------------+
            // |Real formula                                                   |
            // +---------------------------------------------------------------+
            // |************************ Decomposition ************************|
            // +---------------------------------------------------------------+

            for (INTE_TYPE loop = 0; loop < loops; loop++)
            {
                beg = std::chrono::steady_clock::now();

                SchurAngular_SkewSymm_Spectral(A_saf_real, MatA); // Computation being timed.

                end = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count();
                loop_time[loop] = elapsed;
            }
            Record_Time[0][dim_ind][0] = min_time(loop_time, loops);

            // +---------------------------------------------------------------+
            // |************************ Preprocessing ************************|
            // +---------------------------------------------------------------+
            Record_Time[0][dim_ind][1] = 0;
            for (INTE_TYPE loop = 0; loop < loops; loop++)
            {
                beg = std::chrono::steady_clock::now();

                dexp_real.setupSAF(&A_saf_real); // Computation being timed.
                dexp_real.setupPara();

                end = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count();
                loop_time[loop] = elapsed;
            }
            Record_Time[0][dim_ind][1] = min_time(loop_time, loops);

            // +---------------------------------------------------------------+
            // |************************** Evaluation *************************|
            // +---------------------------------------------------------------+
            Record_Time[0][dim_ind][2] = 0;
            for (INTE_TYPE loop = 0; loop < loops; loop++)
            {
                beg = std::chrono::steady_clock::now();

                dexpSkewSymm_forward(MatN, MatM, dexp_real);

                end = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count();
                loop_time[loop] = elapsed;
                // memcpy(MatM.mv, MatN.mv, sizeof(REAL_TYPE) * dim * dim);
                // MatM.mat2vec();
            }
            Record_Time[0][dim_ind][2] = min_time(loop_time, loops);

            // +---------------------------------------------------------------+
            // |                                                   Real formula|
            // +---------------------------------------------------------------+

            // +---------------------------------------------------------------+
            // |Complex formula                                                |
            // +---------------------------------------------------------------+
            // |************************ Decomposition ************************|
            // +---------------------------------------------------------------+
            Record_Time[1][dim_ind][0] = 0;
            for (INTE_TYPE loop = 0; loop < loops; loop++)
            {
                beg = std::chrono::steady_clock::now();

                Spectral_SkewSymm(A_spf_cmpx, MatA);

                end = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count();
                loop_time[loop] = elapsed;
            }
            Record_Time[1][dim_ind][0] = min_time(loop_time, loops);

            // SchurAngular_SkewSymm(A_saf_cmpx, MatA);
            // A_spf_cmpx.setSPF(A_saf_cmpx);

            // +---------------------------------------------------------------+
            // |************************ Preprocessing ************************|
            // +---------------------------------------------------------------+
            Record_Time[1][dim_ind][1] = 0;

            for (INTE_TYPE loop = 0; loop < loops; loop++)
            {
                beg = std::chrono::steady_clock::now();

                dexp_cmpx.setSPF(&A_spf_cmpx);
                dexp_cmpx.setupPara();

                end = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count();
                loop_time[loop] = elapsed;
            }
            Record_Time[1][dim_ind][1] = min_time(loop_time, loops);

            // +---------------------------------------------------------------+
            // |************************** Evaluation *************************|
            // +---------------------------------------------------------------+
            Record_Time[1][dim_ind][2] = 0;
            for (INTE_TYPE loop = 0; loop < loops; loop++)
            {
                beg = std::chrono::steady_clock::now();

                dexpSemiSimple(MatN, MatM, dexp_cmpx, true);

                end = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count();
                loop_time[loop] = elapsed;
                // memcpy(MatM.mv, MatN.mv, sizeof(REAL_TYPE) * dim * dim);
                // MatM.mat2vec();
            }
            Record_Time[1][dim_ind][2] = min_time(loop_time, loops);

            // +---------------------------------------------------------------+
            // |                                                Complex formula|
            // +---------------------------------------------------------------+

            // +---------------------------------------------------------------+
            // |Pade Algorithm                                                 |
            // +---------------------------------------------------------------+
            // |************************ Decomposition ************************|
            // +---------------------------------------------------------------+
            for (INTE_TYPE loop = 0; loop < loops; loop++)
            {
                beg = std::chrono::steady_clock::now();

                A_pade.setup_perturb(MatA);
                A_pade.expPadeApprox(3);
                dexp_pade.setup_empa(&A_pade);
                dexp_pade.setup_rs();

                // A_pade.exp(MatQ);

                end = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count();
                loop_time[loop] = elapsed;
            }
            Record_Time[2][dim_ind][0] = min_time(loop_time, loops);
            // +---------------------------------------------------------------+
            // |************************ Preprocessing ************************|
            // +---------------------------------------------------------------+
            for (INTE_TYPE loop = 0; loop < loops; loop++)
            {
                beg = std::chrono::steady_clock::now();

                dexp_pade.setup_empa(&A_pade);
                // dexp_pade.setup_rs();

                end = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count();
                loop_time[loop] = elapsed;
            }
            Record_Time[2][dim_ind][1] = min_time(loop_time, loops);

            // +---------------------------------------------------------------+
            // |************************** Evaluation *************************|
            // +---------------------------------------------------------------+
            Record_Time[2][dim_ind][2] = 0;
            for (INTE_TYPE loop = 0; loop < loops; loop++)
            {
                beg = std::chrono::steady_clock::now();

                dexp_pade.setup_perturb(MatM);
                dexp_pade.dexp(MatDelta);
                my_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, dim, dim, dim, 1.0, MatQ.v, dim, MatDelta.v, dim, 0.0, MatN.v, dim);

                end = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count();
                loop_time[loop] = elapsed;
                // memcpy(MatM.mv, MatN.mv, sizeof(REAL_TYPE) * dim * dim);
                // MatM.mat2vec();
            }
            Record_Time[2][dim_ind][2] = min_time(loop_time, loops);

            // +---------------------------------------------------------------+
            // |                                                Complex formula|
            // +---------------------------------------------------------------+

            delete[] work_saf;
            delete[] VecD;
            delete[] VecE;
            delete[] VecTau;
        }
    }

    // FILE *pFile;
    char filename[100];
    // string filename;
    sprintf(filename, "%s/dexp_elapsed_time_s%s.txt", paras["-file"].c_str(), paras["-seed"].c_str());
    // pFile = fopen(filename, "w");

    std::ofstream fout;
    fout.open(filename, std::ios_base::binary);

    printf("\rAll tests are finished! Writing to file %s.\n", filename);

    REAL_TYPE dimension;
    REAL_TYPE elapsed_t;
    for (int i = 0; i < methods; i++)
    {
        for (int j = 0; j < dim_end - dim_beg + 1; j++)
        {
            // fprintf(pFile, "%lld\t%lld\t%lld\t%lld\t%lld\t%lld\n",
            //         dim_beg + j,
            //         Record_Time[i][j][0],
            //         Record_Time[i][j][1],
            //         Record_Time[i][j][2],
            //         Record_Time[i][j][1] + Record_Time[i][j][2],
            //         Record_Time[i][j][0] + Record_Time[i][j][1] + Record_Time[i][j][2]);
            dimension = dim_beg + j;
            fout.write(reinterpret_cast<char *>(&dimension), sizeof(REAL_TYPE));
            elapsed_t = Record_Time[i][j][0];
            fout.write(reinterpret_cast<char *>(&elapsed_t), sizeof(REAL_TYPE));
            elapsed_t = Record_Time[i][j][1];
            fout.write(reinterpret_cast<char *>(&elapsed_t), sizeof(REAL_TYPE));
            elapsed_t = Record_Time[i][j][2];
            fout.write(reinterpret_cast<char *>(&elapsed_t), sizeof(REAL_TYPE));
            elapsed_t = Record_Time[i][j][1] + Record_Time[i][j][2];
            fout.write(reinterpret_cast<char *>(&elapsed_t), sizeof(REAL_TYPE));
            elapsed_t = Record_Time[i][j][0] + Record_Time[i][j][1] + Record_Time[i][j][2];
            fout.write(reinterpret_cast<char *>(&elapsed_t), sizeof(REAL_TYPE));
        }
    }

    fout.close();

    printf("Sucessfully written to the file %s\nRoutine exits.\n", filename);

    for (int i = 0; i < methods; i++)
    {
        for (int j = 0; j < dim_end - dim_beg + 1; j++)
            delete[] Record_Time[i][j];
        delete[] Record_Time[i];
    }
    delete[] Record_Time;
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

REAL_TYPE avg_time(REAL_TYPE *time, INTE_TYPE size)
{
    REAL_TYPE t = 0;
    for (INTE_TYPE i = 0; i < size; i++)
        t += time[i];
    t /= size;
    return t;
}
REAL_TYPE min_time(REAL_TYPE *time, INTE_TYPE size)
{
    REAL_TYPE t = time[0];
    for (INTE_TYPE i = 1; i < size; i++)
        if (time[i] < t)
            t = time[i];
    return t;
}