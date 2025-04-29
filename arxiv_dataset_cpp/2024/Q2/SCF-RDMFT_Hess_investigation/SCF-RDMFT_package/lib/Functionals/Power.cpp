#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

using namespace std;

#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"
#include "Power.hpp"


MatrixXd Power_WK(RDM1* gamma){
    int l = gamma->size(); MatrixXd W (l,l);
    VectorXd N = gamma->n().pow(ALPHA);
    MatrixXd v = v_K(gamma,&N);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            W(i,j) = pow(gamma->n(i),ALPHA)* v(i,j);
        }
        
    }
    return W;
}

VectorXd Power_dWK(RDM1* gamma){
    int l = gamma->size(); VectorXd dW = VectorXd::Zero(l);
    VectorXd N = gamma->n().pow(ALPHA);
    MatrixXd v = v_K(gamma,&N);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            dW(j) += ALPHA*pow(gamma->n(i),ALPHA-1.)*gamma->dn(i,j)*v(i,i);
        }
            
    }
    return dW;
}

VectorXd Power_n_K(RDM1* gamma){
    return gamma->n().pow(ALPHA);
}

VectorXd Power_dn_K(RDM1* gamma,int i){
    return ALPHA*(gamma->n().pow(ALPHA)*gamma->dn(i)).diagonal(); 
}

double Power_dn_Kn(RDM1* gamma, int i){
    return 2.*ALPHA*pow(gamma->sqrtn(i),2.*ALPHA-1.);
}

VectorXd Power_ddn_K(RDM1* gamma, int i,int j){
    return ALPHA*(ALPHA-1.)*(gamma->n().pow(ALPHA-2.)*gamma->ddn(i,j)).diagonal();
}

double Power_ddn_Kn(RDM1* gamma, int i, int j){
    return 2.*ALPHA*(2.*ALPHA-1.)*(i==j)*pow(gamma->n(i),2.*ALPHA-2.);
}