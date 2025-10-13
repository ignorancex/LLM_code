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
#include "mmslt.hpp"
#include "hhMat.hpp"

#define num_main_option 4

using std::map;
using std::stof;
using std::stoi;
using std::string;

int main(int argc, char *argv[])
{

    string def_val[num_main_option] = {"20", "10", "20", "9527"};
    string options[num_main_option] = {"-dim", "-col", "-loop", "-seed"};

    auto paras = read_paras(argc, argv, num_main_option, def_val, options);

    std::cout << "Test routine for ArrVec.hpp\n";

    INTE_TYPE d = stoi(paras["-dim"]);
    INTE_TYPE n = stoi(paras["-col"]);
    INTE_TYPE l = stoi(paras["-loop"]);
    INTE_TYPE seed = stoi(paras["-seed"]);

    ArrVec<REAL_TYPE> record(l);
    REAL_TYPE dgemm_time = 0, skmm_time = 0;

    std::default_random_engine rng;
    rng.seed(seed);

    SkewSymmMat A(d);
    ColMat<REAL_TYPE> FullA(d, d);
    A.ColMat<REAL_TYPE>::Zero();
    A.init_low_vec();

    View_ArrVec<REAL_TYPE> ViewLA(A.lv, A.lsize);
    ViewLA.Rand(rng, -1.0, 1.0);
    A.vec2mat();
    FullA.Assign(A.v, d);

    A.ColMat<REAL_TYPE>::Zero();
    A.vec2low();
    A.printf("Skew symmetric A in lower triangular storage:\n");

    FullA.printf("Skew symmetric A in full storage:\n");

    ColMat<REAL_TYPE> B(d, n);
    B.Rand(rng, 0.0, 1.0);
    ColMat<REAL_TYPE> C(n, d);
    C.Rand(rng, 0.0, 1.0);

    ColMat<REAL_TYPE> CA(n, d);
    ColMat<REAL_TYPE> AB(d, n);

    for (INTE_TYPE preheat = 1; preheat >= 0; preheat--)
    {
        if (preheat)
            printf("Preheat the routines...\n");
        TIME_AND_AVG(record.v, (!preheat) * (l - 1) + 1, dgemm_time, { AB.Zero(); }, { cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, n, d, 1.0, FullA.v, d, B.v, d, 0.0, AB.v, d); }, {});
        if (!preheat)
        {
            printf("The product matrix Y = A * B computed by cblas_dgemm in %.10e(ms).\n", dgemm_time);
            AB.printf();
        }

        TIME_AND_AVG(record.v, (!preheat) * (l - 1) + 1, skmm_time, { AB.Zero(); }, { skewblas_skmm('L', d, n, 1.0, A.v, d, B.v, d, 0.0, AB.v, d); }, {});
        if (!preheat)
        {
            printf("The product matrix Y = A * B computed by skewblas_skmm in %.10e(ms).\n", skmm_time);
            AB.printf();
            printf("Computed time(ms): \t(gemm)%.10e,\t(skmm)%.10e.\n", dgemm_time, skmm_time);
        }

        TIME_AND_AVG(record.v, (!preheat) * (l - 1) + 1, dgemm_time, { CA.Zero(); }, { cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, d, d, 1.0, C.v, n, FullA.v, d, 0.0, CA.v, n); }, {});
        if (!preheat)
        {
            CA.printf("The product matrix Y = C * A computed by cblas_dgemm:\n");
        }

        TIME_AND_AVG(record.v, (!preheat) * (l - 1) + 1, skmm_time, { CA.Zero(); }, { skewblas_skmm('R', n, d, 1.0, A.v, d, C.v, n, 0.0, CA.v, n); }, {});
        if (!preheat)
        {
            CA.printf("The product matrix Y = C * A computed by skewblas_skmm:\n");
            printf("Computed time(ms): \t(gemm)%.10e,\t(skmm)%.10e.\n", dgemm_time, skmm_time);
        }
    }

    // CA.Zero();
    // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, d, d, 1.0, C.v, n, FullA.v, d, 0.0, CA.v, n);
    // CA.printf("The product matrix Y = C * A computed by cblas_dgemm:\n");

    // CA.Zero();
    // skewblas_skmm('R', n, d, 1.0, A.v, d, C.v, n, 0.0, CA.v, n);
    // CA.printf("The product matrix Y = C * A computed by skewblas_skmm:\n");
};
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
#include "mmslt.hpp"
#include "hhMat.hpp"

#define num_main_option 4

using std::map;
using std::stof;
using std::stoi;
using std::string;

int main(int argc, char *argv[])
{

    string def_val[num_main_option] = {"20", "10", "20", "9527"};
    string options[num_main_option] = {"-dim", "-col", "-loop", "-seed"};

    auto paras = read_paras(argc, argv, num_main_option, def_val, options);

    std::cout << "Test routine for ArrVec.hpp\n";

    INTE_TYPE d = stoi(paras["-dim"]);
    INTE_TYPE n = stoi(paras["-col"]);
    INTE_TYPE l = stoi(paras["-loop"]);
    INTE_TYPE seed = stoi(paras["-seed"]);

    ArrVec<REAL_TYPE> record(l);
    REAL_TYPE dgemm_time = 0, skmm_time = 0;

    std::default_random_engine rng;
    rng.seed(seed);

    SkewSymmMat A(d);
    ColMat<REAL_TYPE> FullA(d, d);
    A.ColMat<REAL_TYPE>::Zero();
    A.init_low_vec();

    View_ArrVec<REAL_TYPE> ViewLA(A.lv, A.lsize);
    ViewLA.Rand(rng, -1.0, 1.0);
    A.vec2mat();
    FullA.Assign(A.v, d);

    A.ColMat<REAL_TYPE>::Zero();
    A.vec2low();
    A.printf("Skew symmetric A in lower triangular storage:\n");

    FullA.printf("Skew symmetric A in full storage:\n");

    ColMat<REAL_TYPE> B(d, n);
    B.Rand(rng, 0.0, 1.0);
    ColMat<REAL_TYPE> C(n, d);
    C.Rand(rng, 0.0, 1.0);

    ColMat<REAL_TYPE> CA(n, d);
    ColMat<REAL_TYPE> AB(d, n);

    for (INTE_TYPE preheat = 1; preheat >= 0; preheat--)
    {
        if (preheat)
            printf("Preheat the routines...\n");
        TIME_AND_AVG(record.v, (!preheat) * (l - 1) + 1, dgemm_time, { AB.Zero(); }, { cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, n, d, 1.0, FullA.v, d, B.v, d, 0.0, AB.v, d); }, {});
        if (!preheat)
        {
            printf("The product matrix Y = A * B computed by cblas_dgemm in %.10e(ms).\n", dgemm_time);
            AB.printf();
        }

        TIME_AND_AVG(record.v, (!preheat) * (l - 1) + 1, skmm_time, { AB.Zero(); }, { skewblas_skmm('L', d, n, 1.0, A.v, d, B.v, d, 0.0, AB.v, d); }, {});
        if (!preheat)
        {
            printf("The product matrix Y = A * B computed by skewblas_skmm in %.10e(ms).\n", skmm_time);
            AB.printf();
            printf("Computed time(ms): \t(gemm)%.10e,\t(skmm)%.10e.\n", dgemm_time, skmm_time);
        }

        TIME_AND_AVG(record.v, (!preheat) * (l - 1) + 1, dgemm_time, { CA.Zero(); }, { cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, d, d, 1.0, C.v, n, FullA.v, d, 0.0, CA.v, n); }, {});
        if (!preheat)
        {
            CA.printf("The product matrix Y = C * A computed by cblas_dgemm:\n");
        }

        TIME_AND_AVG(record.v, (!preheat) * (l - 1) + 1, skmm_time, { CA.Zero(); }, { skewblas_skmm('R', n, d, 1.0, A.v, d, C.v, n, 0.0, CA.v, n); }, {});
        if (!preheat)
        {
            CA.printf("The product matrix Y = C * A computed by skewblas_skmm:\n");
            printf("Computed time(ms): \t(gemm)%.10e,\t(skmm)%.10e.\n", dgemm_time, skmm_time);
        }
    }
};
