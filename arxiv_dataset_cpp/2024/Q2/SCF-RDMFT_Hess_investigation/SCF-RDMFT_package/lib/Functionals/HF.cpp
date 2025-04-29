#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

using namespace std;

#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"
#include "HF.hpp"

MatrixXd HF_WK(RDM1* gamma){
    int l = gamma->size(); MatrixXd W (l,l);
    VectorXd N = gamma->n();
    MatrixXd v = v_K(gamma,&N);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            W(i,j) = gamma->n(i)* v(i,j);
        }
    }
    
    return 1./2.*W;
}

VectorXd HF_dWK(RDM1* gamma){
    int l = gamma->size(); VectorXd dW = VectorXd::Zero(l);
    VectorXd N = gamma->n();
    MatrixXd v = v_K(gamma,&N);
    for (int i = 0; i<l; i++){
        for(int j = 0; j<l; j++){
            dW(j) += gamma->dn(i,j)*v(i,i);
        }
    }
    return 1./2.*dW;
}

VectorXd HF_n_K(RDM1* gamma){
    return RSQRT_TWO*gamma->n();
}
VectorXd HF_dn_K(RDM1* gamma, int i){
    return RSQRT_TWO*gamma->dn(i).diagonal();
}
double HF_dn_Kn(RDM1* gamma, int i){
    return 2.*RSQRT_TWO*gamma->sqrtn(i);
}
VectorXd HF_ddn_K(RDM1* gamma, int i, int j){
    return RSQRT_TWO*gamma->ddn(i,j).diagonal();
}
double HF_ddn_Kn(RDM1* gamma, int i, int j){
    return 2.*RSQRT_TWO*(i==j);
}


MatrixXd H_WK(RDM1* gamma){
    int l = gamma->size();
    return MatrixXd::Zero(l,l);
}

VectorXd H_dWK(RDM1* gamma){
    int l = gamma->size();
    return VectorXd::Zero(l);
}

VectorXd H_n_K(RDM1* gamma){
    return VectorXd::Zero(gamma->size());
}
VectorXd H_dn_K(RDM1* gamma,int i){
    return VectorXd::Zero(gamma->size());
}
double H_dn_Kn(RDM1* gamma,int i){
    return 0.;
}
VectorXd H_ddn_K(RDM1* gamma,int i, int j){
    return VectorXd::Zero(gamma->size());
}
double H_ddn_Kn(RDM1* gamma,int i, int j){
    return 0.;
}

MatrixXd E1_WK(RDM1* gamma){
    int l = gamma->size();
    return MatrixXd::Zero(l,l);
}

VectorXd E1_dWK(RDM1* gamma){
    int l = gamma->size();
    return VectorXd::Zero(l);
}

VectorXd E1_n_K(RDM1* gamma){
    return VectorXd::Zero(gamma->size());
}
VectorXd E1_dn_K(RDM1* gamma,int i){
    return VectorXd::Zero(gamma->size());
}
double E1_dn_Kn(RDM1* gamma, int i){
    return 0.;
}
VectorXd E1_ddn_K(RDM1* gamma,int i, int j){
    return VectorXd::Zero(gamma->size());
}
double E1_ddn_Kn(RDM1* gamma, int i, int j){
    return 0.;
}