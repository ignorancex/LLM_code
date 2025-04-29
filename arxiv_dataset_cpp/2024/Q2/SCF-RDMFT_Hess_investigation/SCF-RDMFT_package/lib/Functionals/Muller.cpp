#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

using namespace std;

#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"
#include "Muller.hpp"


MatrixXd Muller_WK(RDM1* gamma){
    int l = gamma->size(); MatrixXd W (l,l);
    VectorXd N = gamma->sqrtn();
    MatrixXd v = v_K(gamma,&N);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            W(i,j) = gamma->sqrtn(i)* v(i,j);
        }
        
    }
    return W;
}

VectorXd Muller_dWK(RDM1* gamma){
    int l = gamma->size(); VectorXd dW = VectorXd::Zero(l);
    VectorXd N = gamma->sqrtn();
    MatrixXd v = v_K(gamma,&N);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            dW(j) += gamma->dsqrt_n(i,j)*v(i,i);
        }
            
    }
    return dW;
}

VectorXd Muller_n_K(RDM1* gamma){
    return gamma->sqrtn();
}

VectorXd Muller_dn_K(RDM1* gamma,int i){
    return gamma->dsqrt_n(i).diagonal(); 
}

double Muller_dn_Kn(RDM1* gamma, int i){
    return 1.;
}

VectorXd Muller_ddn_K(RDM1* gamma, int i,int j){
    return gamma->ddsqrt_n(i,j).diagonal();
}

double Muller_ddn_Kn(RDM1* gamma, int i, int j){
    return 0.;
}