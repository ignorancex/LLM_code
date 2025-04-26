#include <cmath>
#include <iostream>

#include "utility.h"
#include "matrix_elements.h"


//=============Baisc function for Q+q --> Q+q==================================

double dqhat_Qq2Qq_dPS(double *PS, size_t n_dims, void *params)
{
    (void) n_dims;
    double t = PS[0];
    double* p = static_cast<double*>(params);
    // for 2->2 process p = {s, temp, M, index};
    double s = p[0], M2=p[2]*p[2];
    int index = int(p[3]); // floor double p[3] into integer

    //double p10_CoM = (s + M2)/(2*std::sqrt(s));
    double p20_CoM = (s - M2)/(2*std::sqrt(s));
    double cos_theta13 = 1. + t/(2*p20_CoM*p20_CoM);
    double sin_theta13 = std::sqrt(1-cos_theta13*cos_theta13);

    double q1_CoM = -p20_CoM * sin_theta13;		 // Delta px
    double q3_CoM = p20_CoM * (1 - cos_theta13); // Delta pz
    double q11_CoM = q1_CoM*q1_CoM;				 // Delta px ^2
    double q33_CoM = q3_CoM*q3_CoM;				 // Delta pz ^2
    double q13_CoM = q1_CoM*q3_CoM;				 // Detta px pz

    double vecq[] = {q1_CoM, q3_CoM, q11_CoM, q33_CoM, q13_CoM, 1.};

    return M2_Qq2Qq(t, params)/c16pi/std::pow(s-M2,2) * vecq[index] ;
}




//============Basic function for Q+g->Q+g
double dqhat_Qg2Qg_dPS(double * PS, size_t n_dims, void * params)
{
    (void)n_dims;
    double t = PS[0];
    double * p = static_cast<double *>(params);
    double s = p[0], M2 = p[2]*p[2];
    int index = int(p[3]); // floor double p[3] into integer

    //double p10_CoM = (s + M2)/(2*std::sqrt(s));
    double p20_CoM = (s - M2)/(2*std::sqrt(s));
    double cos_theta13 = 1. + t/(2*p20_CoM*p20_CoM);
    double sin_theta13 = std::sqrt(1-cos_theta13*cos_theta13);

    double q1_CoM = -p20_CoM * sin_theta13;
    double q3_CoM = p20_CoM * (1 - cos_theta13);
    double q11_CoM = q1_CoM*q1_CoM;
    double q33_CoM = q3_CoM*q3_CoM;
    double q13_CoM = q1_CoM*q3_CoM;

    double vecq[] = {q1_CoM, q3_CoM, q11_CoM, q33_CoM, q13_CoM, 1.};
    return M2_Qg2Qg(t, params)/c16pi/std::pow(s-M2, 2) * vecq[index];    
}

