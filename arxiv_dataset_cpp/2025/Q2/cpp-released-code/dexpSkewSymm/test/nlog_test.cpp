#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <chrono>
// #include <string>
#include <map>

#include <fstream>
#include <iostream>

#include "matOp.hpp"
#include "logmNear.hpp"
#include "skewSchFac.hpp"

#include "basics.hpp"

#define num_main_option 7

int main(int argc, char *argv[])
{
    string def_val[num_main_option] = {"10", "10", "10", "200", "1.0", "9527", "../figures/dexp"};
    string options[num_main_option] = {"-dim", "-stack", "-minscale", "-maxiter", "-length", "-seed", "-file"};
    char char_buff[100];

    auto paras = read_paras(argc, argv, num_main_option, def_val, options);

    INTE_TYPE seed = stoi(paras["-seed"]);
    INTE_TYPE dim = stoi(paras["-dim"]);
    INTE_TYPE stack = stoi(paras["-stack"]);
    INTE_TYPE scale = stoi(paras["-minscale"]);
    INTE_TYPE record = stoi(paras["-maxiter"]);

    REAL_TYPE length_scale = stoi(paras["-length"]);

    INTE_TYPE stack_ind = 0;
    INTE_TYPE scale_ind = 0;
    INTE_TYPE record_ind = 0;

    std::default_random_engine rng;
    rng.seed(seed);
    std::uniform_real_distribution<> interval(-1.0, 1.0);

    INTE_TYPE n = dim;
    INTE_TYPE m = dim / 2;
    REAL_TYPE dist_bound = 0;

    auto ssf = SkewSchurFactor(n);
    auto nearlog = LogmSkewSymm(n);

    auto MatA = SkewSymmMat(n);
    auto MatM = SkewSymmMat(n);

    MatA.init_low_vec();
    MatM.init_low_vec();

    MatA.Rand(rng, -1.0, 1.0);
    MatA.low2upp();
    MatM.Rand(rng, -length_scale, length_scale);
    // cblas_dscal(MatM.lsize, 10.0, MatM.lv, 1);
    MatM.low2upp();

    auto MateA = ColMat<REAL_TYPE>(n, n);
    auto MateM = ColMat<REAL_TYPE>(n, n);

    auto MatQ = ColMat<REAL_TYPE>(n, n);

    auto StackScalar = ArrVec<INTE_TYPE>(record);
    auto StackSOMatQ = ArrVec<ColMat<REAL_TYPE> *>(stack);
    auto StackSchVec = ArrVec<ColMat<REAL_TYPE> *>(stack);
    auto StackAngles = ArrVec<ArrVec<REAL_TYPE> *>(stack);

    for (auto ind = 0; ind < stack; ind++)
    {
        StackSOMatQ.v[ind] = new ColMat<REAL_TYPE>(n, n);
        StackSchVec.v[ind] = new ColMat<REAL_TYPE>(n, n);
        StackAngles.v[ind] = new ArrVec<REAL_TYPE>(m);
    }

    auto VecMateM = ArrVec<ColMat<REAL_TYPE> *>(scale);

    auto RecordBoundA = ArrVec<REAL_TYPE>(record);
    auto RecordSSMatA = ArrVec<ColMat<REAL_TYPE> *>(record);
    auto RecordSOMatQ = ArrVec<ColMat<REAL_TYPE> *>(record);
    auto RecordSchVec = ArrVec<ColMat<REAL_TYPE> *>(record);
    auto RecordAngles = ArrVec<ArrVec<REAL_TYPE> *>(record);

    for (auto ind = 0; ind < record; ind++)
    {
        RecordSSMatA.v[ind] = new ColMat<REAL_TYPE>(n, n);
        RecordSOMatQ.v[ind] = new ColMat<REAL_TYPE>(n, n);
        RecordSchVec.v[ind] = new ColMat<REAL_TYPE>(n, n);
        RecordAngles.v[ind] = new ArrVec<REAL_TYPE>(m);
    }

    nearlog.SSF.Factor_SkewSymm(MatA.v, n);
    nearlog.SSF.Explict_Vector();
    nearlog.SSF.Exponential(MateA);

    RecordBoundA[record_ind] = nearlog.SSF.Dist2ConjugateLocus();
    RecordSSMatA[record_ind]->Assign(MatA);
    RecordSOMatQ[record_ind]->Assign(MateA);
    RecordSchVec[record_ind]->Assign(nearlog.SSF.R);
    RecordAngles[record_ind]->Assign(nearlog.SSF.A);

    ssf.Factor_SkewSymm(MatM.v, n);
    ssf.Explict_Vector();
    for (auto ind = 0; ind < scale; ind++)
    {
        VecMateM.v[ind] = new ColMat<REAL_TYPE>(n, n);
        ssf.Exponential(*VecMateM.v[ind]);
        for (auto angle_ind = 0; angle_ind < m; angle_ind++)
            ssf.A.v[angle_ind] *= 0.5;
    }

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, MateA.v, n, VecMateM.v[0]->v, n, 0.0, MatQ.v, n);
    StackSOMatQ[stack_ind]->Assign(MatQ.v, n);
    StackScalar[stack_ind] = 0;
    nearlog.SSF.Factor_SpecOrth(StackSOMatQ[stack_ind]->v, n);
    nearlog.SSF.Explict_Vector();
    StackSchVec[stack_ind]->Assign(nearlog.SSF.R);
    StackAngles[stack_ind]->Assign(nearlog.SSF.A);

    // MateA.printf();
    // MatQ.printf();

    while (stack_ind >= 0)
    {
        nearlog._diff(RecordSSMatA[record_ind]->v, n, StackSchVec[stack_ind]->v, n, StackAngles[stack_ind]->v);
        // RecordSSMatA[record_ind]->printf("Skew Symmetric A:\n");
        // StackSOMatQ[stack_ind]->printf();
        // nearlog.SSF.printf();

        if (nearlog._norm(nearlog.SSF) < RecordBoundA[record_ind])
        {
            record_ind++;
            if (record_ind >= record)
            {
                std::cout << "Error: Nearlog not found within the given number of iterations!!!\n";
                for (auto ind = 0; ind < record; ind++)
                    RecordSSMatA[ind]->printf((std::snprintf(char_buff, sizeof(char_buff), "Iteration %d:\n", ind), char_buff));
                throw(1);
            }
            nearlog.SSF.R.Assign(*StackSchVec[stack_ind]);
            nearlog.SSF.A.Assign(*StackAngles[stack_ind]);
            RecordSOMatQ[record_ind]->Assign(*StackSOMatQ[stack_ind]);
            RecordSchVec[record_ind]->Assign(nearlog.SSF.R);
            RecordAngles[record_ind]->Assign(nearlog.SSF.A);
            RecordBoundA[record_ind] = nearlog.SSF.Dist2ConjugateLocus();
            nearlog.SSF.GetSkewSymm(RecordSSMatA[record_ind]->v, n);
            skewl2m(*RecordSSMatA[record_ind]);
            stack_ind--;
        }
        else
        {
            StackScalar[stack_ind] += 1;
            scale_ind = StackScalar[stack_ind];
            stack_ind++;
            if (stack_ind >= stack || scale_ind > scale)
            {
                std::cout << "Error: Stack Overflow in nearlog!!!\n";
                throw(1);
            }
            StackScalar[stack_ind] = scale_ind;
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, RecordSOMatQ[record_ind]->v, n, VecMateM[scale_ind]->v, n, 0.0, StackSOMatQ[stack_ind]->v, n);

            nearlog.SSF.Factor_SpecOrth(StackSOMatQ[stack_ind]->v, n);
            nearlog.SSF.Explict_Vector();
            StackSchVec[stack_ind]->Assign(nearlog.SSF.R);
            StackAngles[stack_ind]->Assign(nearlog.SSF.A);
        }
    }

    for (auto ind = 0; ind < record_ind + 1; ind++)
    {
        RecordSSMatA[ind]->printf((std::snprintf(char_buff, sizeof(char_buff), "Iteration %d,\tSkew Symmetric A with distance bound\t%f:\n", ind, RecordBoundA[ind]), char_buff));
        // RecordSOMatQ[ind]->printf((std::snprintf(char_buff, sizeof(char_buff), "Iteration %d,\tSpectial Orthogonal Q:\n", ind), char_buff));
    }

    // Release array
    for (auto ind = 0; ind < stack; ind++)
    {
        delete StackSOMatQ.v[ind];
        delete StackSchVec.v[ind];
        delete StackAngles.v[ind];
    }

    for (auto ind = 0; ind < scale; ind++)
        delete VecMateM.v[ind];

    for (auto ind = 0; ind < record; ind++)
    {
        delete RecordSSMatA.v[ind];
        delete RecordSOMatQ.v[ind];
        delete RecordSchVec.v[ind];
        delete RecordAngles.v[ind];
    }

    return 1;

    // auto VecMatR = DynamicArr<ColMat<REAL_TYPE> *>(0, 20);
    // auto VecVecT = DynamicArr<ArrVec<REAL_TYPE> *>(0, 20);

    // // SkewSymm

    // MatQ.Zero();
    // MatE.Assign(MatR);
    // ssf.Factor(MatA.v, dim);
    // ssf.Explict_Vector();
    // ssf.Exponential(MatQ.v, dim);

    // cblas_daxpy(dim * dim, -1, MatQ.v, 1, MatE.v, 1);

    // for (auto ind = 0; ind < dim * dim; ind++)
    // 	MatE.v[ind] = std::ldexp(MatE.v[ind], 50);
    // err_schur = std::ldexp(cblas_dnrm2(dim * dim, MatE.v, 1), -50);

    // // Pade-order(3/3)

    // MatQ.Zero();
    // MatE.Assign(MatR);
    // dexp_pade.Expm(MatQ.v, dim, MatA.v, dim, 3);

    // cblas_daxpy(dim * dim, -1, MatQ.v, 1, MatE.v, 1);
    // err_p3 = cblas_dnrm2(dim * dim, MatE.v, 1);

    // // Pade-order(13/13)

    // MatQ.Zero();
    // MatE.Assign(MatR);
    // dexp_pade.Expm(MatQ.v, dim, MatA.v, dim, 13);

    // cblas_daxpy(dim * dim, -1, MatQ.v, 1, MatE.v, 1);
    // for (auto ind = 0; ind < dim * dim; ind++)
    // 	MatE.v[ind] = std::ldexp(MatE.v[ind], 50);
    // err_p13 = std::ldexp(cblas_dnrm2(dim * dim, MatE.v, 1), -50);
};