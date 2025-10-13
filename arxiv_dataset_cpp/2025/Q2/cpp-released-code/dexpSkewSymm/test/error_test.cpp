#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <chrono>
// #include <string>
#include <map>

#include <fstream>
#include <iostream>

#include "matOp.hpp"
#include "dexpSkew.hpp"
#include "dexpPade.hpp"
#include "dexpSkewEig.hpp"
#include "dlogSkewPade.hpp"

#include "basics.hpp"

#define num_main_option 3

int main(int argc, char *argv[])
{
    string def_val[num_main_option] = {"100", "9527", "/mnt/d/figures/dexp"};
    string options[num_main_option] = {"-max-size", "-seed", "-file"};

    auto paras = read_paras(argc, argv, num_main_option, def_val, options);

    INTE_TYPE smax = stoi(paras["-max-size"]);
    INTE_TYPE seed = stoi(paras["-seed"]);

    INTE_TYPE record_cnt = 12;
    auto Record = ColMat<REAL_TYPE>(smax, record_cnt);

    std::cout << "Error Experiments:\n";

    char fname_skew[100];
    char fname_spor[100];
    char fname_forw[100];
    char fname_inve[100];
    sprintf(fname_skew, "%s/expm_accurate_skewsymm_s%lld.out", paras["-file"].c_str(), seed);
    sprintf(fname_spor, "%s/expm_accurate_specorth_s%lld.out", paras["-file"].c_str(), seed);
    sprintf(fname_forw, "%s/expm_accurate_xforward_s%lld.out", paras["-file"].c_str(), seed);
    sprintf(fname_inve, "%s/expm_accurate_yinverse_s%lld.out", paras["-file"].c_str(), seed);

    std::ifstream fin_skew(fname_skew, std::ios::binary);
    std::ifstream fin_spor(fname_spor, std::ios::binary);
    std::ifstream fin_forw(fname_forw, std::ios::binary);
    std::ifstream fin_inve(fname_inve, std::ios::binary);

    if (!fin_skew || !fin_spor || !fin_forw || !fin_inve)
    {
        std::cerr << "Failed to open the file.\n";
        return 1;
    }

    ArrVec<INTE_TYPE> Record_Dimension(smax);
    INTE_TYPE size = 0;

    while (size <= smax && fin_skew.peek() != EOF)
    {
        REAL_TYPE n_file;
        fin_skew.read(reinterpret_cast<char *>(&n_file), sizeof(REAL_TYPE));
        fin_spor.read(reinterpret_cast<char *>(&n_file), sizeof(REAL_TYPE));
        fin_forw.read(reinterpret_cast<char *>(&n_file), sizeof(REAL_TYPE));
        fin_inve.read(reinterpret_cast<char *>(&n_file), sizeof(REAL_TYPE));
        if (fin_skew.eof())
            break;
        if (!fin_skew)
        {
            std::cerr << "Failed to read matrix size.\n";
            break;
        }

        INTE_TYPE n = static_cast<INTE_TYPE>(n_file);
        INTE_TYPE nsq = n * n;
        INTE_TYPE dlog_pade_scale = 0;

        Record(size, 0) = n;

        auto MatAexact = SkewSymmMat(n);
        auto MatXexact = SkewSymmMat(n);
        auto MatQexact = ColMat<REAL_TYPE>(n, n);
        auto MatYexact = ColMat<REAL_TYPE>(n, n);

        fin_skew.read(reinterpret_cast<char *>(MatAexact.v), sizeof(double) * nsq);
        fin_spor.read(reinterpret_cast<char *>(MatQexact.v), sizeof(double) * nsq);
        fin_forw.read(reinterpret_cast<char *>(MatXexact.v), sizeof(double) * nsq);
        fin_inve.read(reinterpret_cast<char *>(MatYexact.v), sizeof(double) * nsq);

        MatAexact.init_low_vec();
        MatAexact.mat2vec();
        MatXexact.init_low_vec();
        MatXexact.mat2vec();

        auto MatLY = SkewSymmMat(n);
        auto MatLX = SkewSymmMat(n);
        auto MateA = ColMat<REAL_TYPE>(n, n);
        auto MatDX = ColMat<REAL_TYPE>(n, n);

        MatLY.init_low_vec();
        MatLX.init_low_vec();

        auto ssf = SkewSchurFactor(n);
        auto dexp_skew = dexpSkewSymmPara(n);
        auto dexp_pade = dexpPadeApprox(n);
        auto dexp_cmpx = dexpSkewEigenPara(n);
        auto dlog_pade = dlogSkewPadeApprox(n);

        // Error in Dexp(A)[X]

        ssf.Factor(MatAexact.v, n);
        ssf.Explict_Vector();
        ssf.Exponential(MateA.v, n);
        dexp_skew.Parameter(ssf);
        dexp_skew.Forward(MatLX, MatXexact);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, MateA.v, n, MatLX.v, n, 0.0, MatDX.v, n); // Computation done.
        cblas_daxpy(nsq, -1.0, MatYexact.v, 1, MatDX.v, 1);
        Record(size, 1) = cblas_dnrm2(nsq, MatDX.v, 1) / nsq;
        MatDX.Zero();

        ssf.Factor(MatAexact.v, n);
        ssf.Explict_Vector();
        ssf.Exponential(MateA.v, n);
        dexp_cmpx.Assign(ssf);
        // dexp_cmpx.Parameter_Skew();
        // dexp_cmpx.Dexp(MatLX.v, n, MatXexact.v, n);
        // skewl2m(MatLX);
        // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, MateA.v, n, MatLX.v, n, 0.0, MatDX.v, n); // Computation done.
        dexp_cmpx.Parameter_Full();
        dexp_cmpx.Dexp(MatDX.v, n, MatXexact.v, n); // Computation done.
        cblas_daxpy(nsq, -1.0, MatYexact.v, 1, MatDX.v, 1);
        Record(size, 2) = cblas_dnrm2(nsq, MatDX.v, 1) / nsq;
        MatDX.Zero();

        dexp_pade.Expm(MateA.v, n, MatAexact.v, n, 3);
        dexp_pade.Dexp(MatDX.v, n, MatXexact.v, n); // Computation done.
        cblas_daxpy(nsq, -1.0, MatYexact.v, 1, MatDX.v, 1);
        Record(size, 3) = cblas_dnrm2(nsq, MatDX.v, 1) / nsq;
        MatDX.Zero();

        dexp_pade.Expm(MateA.v, n, MatAexact.v, n, 13);
        dexp_pade.Dexp(MatDX.v, n, MatXexact.v, n); // Computation done.
        cblas_daxpy(nsq, -1.0, MatYexact.v, 1, MatDX.v, 1);
        Record(size, 4) = cblas_dnrm2(nsq, MatDX.v, 1) / nsq;
        MatDX.Zero();

        // // Error in Dexp(A)^{-1}[X]

        ssf.Factor(MatAexact.v, n);
        ssf.Explict_Vector();
        ssf.Exponential(MateA.v, n);
        dexp_skew.Parameter(ssf);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, n, n, 1.0, MateA.v, n, MatYexact.v, n, 0.0, MatLX.v, n);
        skewl2m(MatLX);
        MatLX.mat2vec();
        dexp_skew.Inverse(MatLY, MatLX); // Computation done.
        cblas_daxpy(nsq, -1.0, MatXexact.v, 1, MatLY.v, 1);
        Record(size, 5) = cblas_dnrm2(nsq, MatLY.v, 1) / nsq;
        MatLY.Zero();

        ssf.Factor(MatAexact.v, n);
        ssf.Explict_Vector();
        ssf.Exponential(MateA.v, n);
        dexp_cmpx.Assign(ssf);
        // dexp_cmpx.Parameter_Skew();
        // cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, n, n, 1.0, MateA.v, n, MatYexact.v, n, 0.0, MatLX.v, n);
        // dexp_cmpx.DexpInv(MatLY.v, n, MatLX.v, n);
        dexp_cmpx.Parameter_Full();
        dexp_cmpx.DexpInv(MatLY.v, n, MatYexact.v, n); // Computation done.
        cblas_daxpy(nsq, -1.0, MatXexact.v, 1, MatLY.v, 1);
        Record(size, 6) = cblas_dnrm2(nsq, MatLY.v, 1) / nsq;
        MatLY.Zero();

        ssf.Factor(MatAexact.v, n);
        ssf.Explict_Vector();
        dlog_pade.Parameter(ssf);
        dlog_pade_scale = ceil(log2(MatAexact.Norm1() / _EXPM_PADE_APPROX_BOUNDS[12]));
        dlog_pade_scale = dlog_pade_scale < 0 ? 0 : dlog_pade_scale;
        dlog_pade._set_orders(1, dlog_pade_scale);
        dlog_pade.Dlog(MatLY.v, n, MatYexact.v, n);
        cblas_daxpy(nsq, -1.0, MatXexact.v, 1, MatLY.v, 1);
        Record(size, 7) = cblas_dnrm2(nsq, MatLY.v, 1) / nsq;
        MatLY.Zero();

        ssf.Factor(MatAexact.v, n);
        ssf.Explict_Vector();
        dlog_pade.Parameter(ssf);
        dlog_pade_scale = ceil(log2(MatAexact.Norm1() / _EXPM_PADE_APPROX_BOUNDS[12]));
        dlog_pade_scale = dlog_pade_scale < 0 ? 0 : dlog_pade_scale;
        dlog_pade._set_orders(7, dlog_pade_scale);
        dlog_pade.Dlog(MatLY.v, n, MatYexact.v, n);
        cblas_daxpy(nsq, -1.0, MatXexact.v, 1, MatLY.v, 1);
        Record(size, 8) = cblas_dnrm2(nsq, MatLY.v, 1) / nsq;
        MatLY.Zero();

        // Error in exponential

        ssf.Factor(MatAexact.v, n);
        ssf.Explict_Vector();
        ssf.Exponential(MateA.v, n);
        cblas_daxpy(nsq, -1.0, MatQexact.v, 1, MateA.v, 1);
        Record(size, 9) = cblas_dnrm2(nsq, MateA.v, 1) / nsq;
        MateA.Zero();

        dexp_pade.Expm(MateA.v, n, MatAexact.v, n, 3);
        cblas_daxpy(nsq, -1.0, MatQexact.v, 1, MateA.v, 1);
        Record(size, 10) = cblas_dnrm2(nsq, MateA.v, 1) / nsq;
        MateA.Zero();

        dexp_pade.Expm(MateA.v, n, MatAexact.v, n, 13);
        cblas_daxpy(nsq, -1.0, MatQexact.v, 1, MateA.v, 1);
        Record(size, 11) = cblas_dnrm2(nsq, MateA.v, 1) / nsq;
        MateA.Zero();

        size++;
    }

    fin_skew.close();
    fin_spor.close();
    fin_forw.close();
    fin_inve.close();

    char filename[100];

    sprintf(filename, "%s/dexp_error_s%s.txt", paras["-file"].c_str(), paras["-seed"].c_str());

    std::ofstream fout;
    fout.open(filename, std::ios_base::binary);

    printf("\rAll tests are finished! Writing to file %s.\n", filename);

    for (int i = 0; i < size; i++)
        for (int j = 0; j < record_cnt; j++)
            fout.write(reinterpret_cast<char *>(Record.cstptr(i, j)), sizeof(REAL_TYPE));

    fout.close();

    return 1;
};