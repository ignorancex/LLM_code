#include <cstdio>
#include <cstdlib>
#include <random>
#include <chrono>
#include <map>

#include "basics.hpp"
#include "arrVec.hpp"
#include "skewMat.hpp"
#include "skmv.hpp"
#include "skmm.hpp"
#include "mmlt.hpp"
#include "hhMat.hpp"

#define num_main_option 7

using std::map;
using std::stof;
using std::stoi;
using std::string;

int main(int argc, char *argv[])
{

    string def_val[num_main_option] = {"20", "10", "10", "3", "20", "9527", "../figures/dexp"};
    string options[num_main_option] = {"-row", "-col", "-inn", "-blk", "-loop", "-seed", "-file"};

    auto paras = read_paras(argc, argv, num_main_option, def_val, options);

    std::cout << "Test routine for ArrVec.hpp\n";

    INTE_TYPE r = stoi(paras["-row"]);
    INTE_TYPE c = stoi(paras["-col"]);
    INTE_TYPE k = stoi(paras["-inn"]);
    INTE_TYPE blk_max = stoi(paras["-blk"]);
    INTE_TYPE l = stoi(paras["-loop"]);
    INTE_TYPE seed = stoi(paras["-seed"]);

    ArrVec<REAL_TYPE> record(l);
    ArrVec<REAL_TYPE> time(blk_max + 1);
    REAL_TYPE gemm_time = 0, mmslt_time = 0;

    std::default_random_engine rng;
    rng.seed(seed);

    ColMat<REAL_TYPE> A(r, k);
    ColMat<REAL_TYPE> B(k, c);
    ColMat<REAL_TYPE> C(r, c);

    A.Rand(rng, -1.0, 1.0);
    B.Rand(rng, -1.0, 1.0);

    for (INTE_TYPE preheat = 1; preheat >= 0; preheat--)
    {
        if (preheat)
            printf("Preheat the routines...\n");
        TIME_AND_AVG(record.v, (!preheat) * (l - 1) + 1, time[0], { C.Zero(); }, { cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, r, c, k, 1.0, A.v, r, B.v, k, 0.0, C.v, r); }, {});

        for (INTE_TYPE blk = 1; blk <= blk_max; blk++)
        {
            TIME_AND_AVG(record.v, (!preheat) * (l - 1) + 1, time[blk], { C.Zero(); }, { skewblas_mmlt(CblasNoTrans, CblasNoTrans, r, c, k, 1.0, A.v, r, B.v, k, 0.0, C.v, r, blk); }, {});
        }
    }

    printf("All tests are finished!\n");

    char filename[100];
    std::ofstream fout;

    // string filename;

    sprintf(filename, "%s/mmlt_m%lld_n%lld_k%lld_s%lld.out", paras["-file"].c_str(), r, c, k, seed);

    fout.open(filename, std::ios_base::binary);
    printf("Writing to the file %s.\n", filename);

    for (INTE_TYPE ind = 0; ind <= blk_max; ind++)
        fout.write(reinterpret_cast<char *>(&time[ind]), sizeof(REAL_TYPE));
    fout.close();

    printf("Routine exits.\n");
};
