#include <stdio.h>
#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <float.h>
#include <iostream>
#include <time.h>

using namespace std;
using namespace Eigen;

#include "1RDM_class.hpp"
#include "Functional_class.hpp"
#include "Matrix_Tensor_converter.cpp"
#include "EBI_add.hpp"

/*Constructor for the Functional class
\param args 8 functions defining the functional F, assumed to be of the form:
        F = WJ(fJ,gJ) -WK(fK,gK) where WJ(fJ,gJ) = sum_{mu nu kappa lambda}fJ(n)_{mu nu}gJ(n)_{lambda kappa}[mu nu|kappa lambda]
                                       WK(fK,gK) = sum-{mu nu kappa lambda}fK(n)_{lambda mu}gK(n)_{nu kappa}[mu nu|kappa lambda]
                                       (n the occupations, J Coulomb, K exchange)
        see K.J.H. "Giesbertz, Avoiding the 4-index transformation in one-body reduced density matrix functional calculations for 
            separable functionals", Phys. Chem. Chem. Phys. 18, 21024-21031 (2016)
*/

Functional::Functional(MatrixXd(*W_K)(RDM1*), VectorXd(*dW_K)(RDM1*), bool is_J_func, VectorXd(*dW_K_subspace)(RDM1*,int), VectorXd (*n_K)(RDM1*), 
                        VectorXd (*dn_K)(RDM1*,int), double (*dn_Kn)(RDM1*,int), VectorXd (*ddn_K)(RDM1*,int,int),double (*ddn_Kn)(RDM1*,int,int)){
    W_K_ = W_K; dW_K_ = dW_K; dW_K_subspace_ = dW_K_subspace; is_J_func_ = is_J_func; n_K_ = n_K; dn_K_ = dn_K; dn_Kn_ = dn_Kn; ddn_K_ = ddn_K; ddn_Kn_ = ddn_Kn;
}

Functional::Functional(MatrixXd(*W_K)(RDM1*), VectorXd(*dW_K)(RDM1*), VectorXd (*n_K)(RDM1*), VectorXd (*dn_K)(RDM1*,int), double (*dn_Kn)(RDM1*,int), 
                        VectorXd (*ddn_K)(RDM1*,int,int), double (*ddn_Kn)(RDM1*,int,int), bool is_J_func){
    W_K_ = W_K; dW_K_ = dW_K; n_K_ = n_K; dn_K_ = dn_K; dn_Kn_ = dn_Kn; ddn_K_ = ddn_K; ddn_Kn_ = ddn_Kn; is_J_func_ = is_J_func;
}

//Check wether functional needs to build the subspaces (i.e. if functionals is PNOF7 for now)
bool Functional::needs_subspace() const{
    return dW_K_subspace_ != nullptr;
}

//Computes the energy of gamma for the functional
double Functional::E(RDM1* gamma) const {
    int l = gamma->size();
    MatrixXd W_J(l,l); W_J = compute_WJ(gamma); MatrixXd W_K(l,l); W_K = compute_WK(gamma);
    return E1(gamma) + E_Hxc(&W_J,&W_K); 
}

double Functional::E(RDM1* gamma, MatrixXd* W_J, MatrixXd* W_K) const {
    return E1(gamma) + E_Hxc(W_J,W_K); 
}

//Computes the gradient of the energy of gamma for the functional
//if only_n = False return the derivatives respect to the NOs
//if only_no = False return the derivatives respect to the occupations
VectorXd Functional::grad_E(RDM1* gamma, bool only_n, bool only_no) const {
    MatrixXd W_J = compute_WJ(gamma); MatrixXd W_K = compute_WK(gamma);
    return dE1(gamma, only_n, only_no) + dE_Hxc(gamma, &W_J,&W_K, only_n, only_no); 
}

VectorXd Functional::grad_E(RDM1* gamma, MatrixXd* W_J, MatrixXd* W_K, bool only_n, bool only_no) const {
    return dE1(gamma, only_n, only_no) + dE_Hxc(gamma, W_J, W_K, only_n, only_no); 
}

VectorXd Functional::grad_E_subspace(RDM1* gamma, int g) const{
    //Derivative of the energy resp to the occ, for only one subspace of PNOF Omega, assuming J fuctional.
    MatrixXd W_K = compute_WK(gamma); int l = gamma->size();
    VectorXd dE1_bis = VectorXd::Zero(l); VectorXd temp = dE1(gamma, true, false);
    for (int i: gamma->omega[g]){
        dE1_bis(i) = temp(i);
    }
    return dE1_bis+dE_Hxc_subspace(gamma, g);
}
//Compute Hessian of the energy of gamma for the functional
// if auxhess is a not provided use the exact Hessian
MatrixXd Functional::hess_E_exa(RDM1* gamma, bool only_n, bool only_no, bool only_coupled) const{
    return ddE1(gamma,only_n,only_no) + ddE_Hxc(gamma,only_n,only_no,only_coupled);
}

MatrixXd Functional::hess_E(RDM1* gamma, bool only_n, bool only_no, bool only_coupled) const{
    return ddE1(gamma,only_n,only_no) + ddE_Hxc_k(gamma, only_n, only_no, only_coupled); 
}

MatrixXd Functional::hess_E_cheap(RDM1* gamma, bool only_n, bool only_no, bool only_coupled) const{
    //Mathematically equivalent to hess_E but numerical error is enought to alter the convergence
    MatrixXd J = Jac(gamma);
    return ddE1(gamma,only_n,only_no) + J.transpose()*ddE_Hxc_aux(gamma, only_n, only_no, only_coupled)*J + ddJac(gamma,only_n,only_no); 
}

//Computes the functional independant part of the energy (1 electron and nuclei)
double E1(RDM1* gamma){
    MatrixXd g = gamma->mat();
    return gamma->E_nuc + compute_E1(&gamma->int1e,&g);
}

//Computes the gradient of the 1 electron part of the energy
VectorXd dE1(RDM1* gamma, bool only_n, bool only_no){
    int l = gamma->size(); int ll = l*(l+1)/2;
    MatrixXd* H1 = &gamma->int1e ; MatrixXd N = gamma->n().asDiagonal(); 
    VectorXd dE1 (ll); MatrixXd g = gamma->mat(); MatrixXd* C = &gamma->no; 
    MatrixXd Ct = C->transpose(); MatrixXd NC = N*Ct;
    if (not only_no){
        for (int i =0; i<l;i++){
            MatrixXd dg = (*C) *gamma->dn(i)* Ct;
            dE1(i) = compute_E1(H1,&dg);
        } 
    }
    if (not only_n){
        int index = l;
        for (int a =0; a<l;a++){
            for (int b =0; b<a;b++){
                MatrixXd dCNC = dU(C,a,b)*NC;
                MatrixXd dg = dCNC+dCNC.transpose();
                dE1(index) =  compute_E1(H1,&dg);
                index++;
            }
        }
    }
    if (only_n) {return dE1.segment(0,l);}
    else{ if(only_no){return dE1.segment(l, l*(l-1)/2);}
    else{ return dE1;}
    }
    
}
//Computes the Hessian of the 1 electron part of the energy
MatrixXd ddE1(RDM1* gamma, bool only_n, bool only_no){
    int l = gamma->size(); int ll = l*(l-1)/2; 
    MatrixXd N = gamma->n().asDiagonal(); 
    MatrixXd ddE1 = MatrixXd::Zero(l+ll,l+ll); MatrixXd g = gamma->mat(); MatrixXd* C = &gamma->no; 
    MatrixXd Ct = C->transpose(); MatrixXd NC = N*C->transpose(); 
    MatrixXd H = Ct*gamma->int1e*(*C);
    if (not only_no){
        for (int i=0;i<l;i++){
            for (int j=0;j<i+1;j++){
                ddE1(i,j) = 0;
                for (int k=0;k<l;k++){
                    ddE1(i,j) += H(k,k) * gamma->ddn(k,i,j);
                }
                ddE1(j,i) = ddE1(i,j);
            }
        }
    }
    if (not only_n){
        for (int i=0;i<l;i++){
            for (int j=0;j<i;j++){
                for (int p=0;p<=i;p++){
                    if(p<i){
                    double temp = 2.*H(j,p)*(gamma->n(i)-gamma->n(p));
                    ddE1(i*(i-1)/2+j+l,i*(i-1)/2+p+l) = temp;
                        if(p!=j){ ddE1(i*(i-1)/2+p+l,i*(i-1)/2+j+l) = temp; }
                    }
                    if(j<p){
                    double temp = 2.*H(i,p)*(gamma->n(j)-gamma->n(p));
                    ddE1(i*(i-1)/2+j+l,p*(p-1)/2+j+l) += temp;
                        if(p!=i){ ddE1(p*(p-1)/2+j+l,i*(i-1)/2+j+l) += temp; }
                    }
                    if(p<j){
                    double temp = -2.*H(i,p)*(gamma->n(j)-gamma->n(p));
                    ddE1(i*(i-1)/2+j+l,j*(j-1)/2+p+l) += temp;
                    ddE1(j*(j-1)/2+p+l,i*(i-1)/2+j+l) += temp;
                    }
                    
                }
            }
        }
    }
    if((not only_n) && (not only_no)){
        for (int i=0;i<l;i++){
            for (int j=0;j<i;j++){
                for (int k=0;k<l;k++){
                    double temp = 2.*(gamma->dn(j,k) - gamma->dn(i,k))*H(i,j);
                    ddE1(i*(i-1)/2+j+l,k) = temp;
                    ddE1(k,i*(i-1)/2+j+l) = temp;
                }
            }
        }
    }
    if (only_n){return ddE1.block(0,0,l,l);}
    else{ if(only_no){return ddE1.block(l,l,ll,ll);}
    else{ return ddE1; }
    }
}


//Computes the part of the Hessian of the 1 electron requireing only 1st derivative respect to E
MatrixXd ddE1_k(RDM1* gamma, bool only_n, bool only_no){
    int l = gamma->size(); int ll = l*(l+1)/2; 
    MatrixXd C = gamma->no; MatrixXd Ct = C.transpose();
    MatrixXd H1 = Ct*gamma->int1e*C; MatrixXd N = gamma->n().asDiagonal(); 
    MatrixXd ddE1 = MatrixXd::Zero(ll,ll); 
    if(not only_no){
        VectorXd dnE1 (l); 
        for(int i=0;i<l;i++){
            dnE1(i) = 2.*gamma->sqrtn(i)*H1(i,i);
        }
        for (int i=0;i<l;i++){
            for (int j=0;j<=i;j++){
                ddE1(i,j) = gamma->ddsqrt_n(i,j).diagonal().dot(dnE1); 
                ddE1(j,i) = ddE1(i,j);                                
                
            }
        }
    }
    if (not only_n){
        MatrixXd dE1 = 2.*H1*gamma->n().asDiagonal(); dE1 = dE1.reshaped(l*l,1);
        MatrixXd I = MatrixXd::Identity(l,l);
        int id1 = l; int id2 = l;
        for (int i=0;i<l;i++){
            for (int j=0;j<i;j++){
                for (int p=0;p<=i;p++){
                    for (int q=0;q<p;q++){
                        MatrixXd d2U = ddU(&I,i,j,p,q).reshaped(1,l*l);
                        ddE1(id1,id2) = (d2U*dE1)(0,0);
                        ddE1(id2,id1) = ddE1(id1,id2);
                        id2++;
                    }
                }
                id1++; id2=l;
            }
        } 
    }
    if (only_n){return ddE1.block(0,0,l,l);}
    else{ if(only_no){return ddE1.block(l,l,ll,ll);}
    else{ return ddE1; }
    }
}


// Compute the 1 electron part of the energy
double compute_E1(MatrixXd* H, MatrixXd* g){
    
    int l = g->rows(); int ll = pow(l,2);
    MatrixXd res (1,1); res =  H->reshaped(1,ll) * g->reshaped(ll,1);
    return res(0,0);

}

// Compute the Jacobian of the transformation between (n,U) space and the space of parameters
MatrixXd Functional::Jac(RDM1* gamma, bool only_n, bool only_no) const{
    int l = gamma->size(); int ll = l*(l+1)/2; int l2 =l*l;
    MatrixXd J = MatrixXd::Zero(l2+l,ll);
    if (not only_no){
        for(int i=0;i<l;i++){
            J.block(0,i,l,1) = gamma->dsqrt_n(i).diagonal();
        }
    }
    if(not only_n){
        int index=l; MatrixXd I = MatrixXd::Identity(l,l);
        for(int i=0;i<l;i++){
            for (int j=0;j<i;j++){
                J.block(l,index,l2,1) = dU(&I,i,j).reshaped(l2,1);
                index++;
            }
        }
    }
    if(only_n){return J.block(0,0,l,l);}
    else if(only_no){return J.block(l,l,l2,ll-l);}
    else {return J;}
}

// Compute the Inverse of the Jacobian of the transformation between (n,U) space and the space of parameters
MatrixXd Functional::InvJac(RDM1* gamma, bool only_n, bool only_no) const{
    int l = gamma->size(); int ll = l*(l+1)/2; int l2 =l*l;
    MatrixXd J = MatrixXd::Zero(ll,l2+l);
    
    if (not only_no){
        //True pseudo-inverse but more expensive
        if (l<30){
            J.block(0,0,l,l) = Jac(gamma,true).completeOrthogonalDecomposition().pseudoInverse();
        }
        //Approximate pseudo-inverse (occ block of J is not invertible because has a 0 eigenvalue comming from mu)
        //since only one 0 eigenvalue, becomes exact in the limit l->infinity 
        else{
            for(int i=0;i<l;i++){
                J.block(0,i,l,1) = gamma->dx_sqrtn(i).diagonal();
            }
        }
        
    }
    if(not only_n){
        int index=l; MatrixXd I = MatrixXd::Identity(l,l);
        for(int i=0;i<l;i++){
            for (int j=0;j<i;j++){
                J.block(index,l,1,l2) = dU(&I,i,j).reshaped(1,l2)/2.;
                index++;
            }
        }
    }
    if(only_n){return J.block(0,0,l,l);}
    else if(only_no){return J.block(l,l,ll-l,l2);}
    else {return J;}
}

MatrixXd Functional::x_space_hess(RDM1* gamma,MatrixXd* hess) const{
    MatrixXd invJ = InvJac(gamma);
    if (invJ.rows()!=hess->cols()){
        throw invalid_argument("Dimensions of the Jacobian and Hessian are not consistantes.");
    }
    return invJ.transpose()*(*hess)*invJ;
}

// Compute the potential in NO basis for J and K (using f and g, see Constructor operator)
MatrixXd v_J(RDM1* gamma, VectorXd* n, bool is_J_func) {
    int l = gamma->size(); int ll = pow(l,2);
    if(is_J_func){
        return MatrixXd::Zero(l,l);
    }
    MatrixXd f = gamma->no * n->asDiagonal() *gamma->no.transpose();
    MatrixXd res = ( gamma->int2e *f.reshaped(ll,1) ).reshaped(l,l);
    return gamma->no.transpose() *res *gamma->no;
}

MatrixXd v_K(RDM1* gamma, VectorXd* n) { 
    int l = gamma->size(); int ll = pow(l,2);
    MatrixXd f = gamma->no * n->asDiagonal() * gamma->no.transpose();
    MatrixXd res = (gamma->int2e_x * f.reshaped(ll,1)).reshaped(l,l);
    return gamma->no.transpose() * res * gamma->no;
}

// Compute the derivative of v_J, v_K respect to the occupation parameters 
MatrixXd Functional::dv_J(RDM1* gamma,int i) const{
    int l = gamma->size(); int ll = pow(l,2);
    if (is_J_func_){
        return MatrixXd::Zero(l,l);
    }
    MatrixXd df = gamma->no * gamma->dn(i) * gamma->no.transpose();
    MatrixXd res(l,l); res = (gamma->int2e * df.reshaped(ll,1)).reshaped(l,l);
    return gamma->no.transpose() * res * gamma->no;
}

MatrixXd Functional::dv_K(RDM1* gamma, int i) const{ 
    int l = gamma->size(); int ll = pow(l,2);
    MatrixXd df = gamma->no * dn_K_(gamma,i).asDiagonal() * gamma->no.transpose();
    MatrixXd res = (gamma->int2e_x * df.reshaped(ll,1)).reshaped(l,l);
    return gamma->no.transpose() * res * gamma->no;
}

// Compute the W_J W_K matrices in NO basis
MatrixXd Functional::compute_WJ(RDM1* gamma) const{
    int l = gamma->size(); 
    if (is_J_func_){ 
        return MatrixXd::Zero(l,l);
    }
    MatrixXd W (l,l);
    VectorXd N = gamma->n(); MatrixXd v = v_J(gamma,&N,is_J_func_);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            W(i,j) = N(i)* v(i,j);
        }
    }
    return W;
}

MatrixXd Functional::compute_WK(RDM1* gamma) const{
    return W_K_(gamma);
}

Tensor<double,4> Tensor_prod(MatrixXd a, Tensor<double,4> b, int i, int j){
    int l=-1;
    if (i==1){l=a.rows();}
    if (i==0){l=a.cols();}
    if (j>3 || j<0){throw std::invalid_argument( "Invalide index for b" );}
    if (l!=b.dimension(j)){throw std::invalid_argument( "Indexes of a and b must have same dimension." );}
    Eigen::array<int,4> index;
    if (j==0){ index = {3,1,2,0};}
    if (j==1){ index = {0,3,2,1};}
    if (j==2){ index = {0,1,3,2};}
    if (j==3){ index = {0,1,2,3};}
    Tensor<double,4> B = b.shuffle(index);
    
    Tensor<double,2> A = TensorCast(a,l,l);
    Eigen::array<Eigen::IndexPair<int>,1> prod = {Eigen::IndexPair<int>(3,i)};
    Tensor<double,4> temp = B.contract(A,prod);

    Tensor<double,4> res = temp.shuffle(index);
    return res;
}

// Compute Wbar_J Wbar_K (used in Hessian respect to rotations)
Tensor<double,4> Functional::compute_Wbar_J(RDM1* gamma) const{
    int l = gamma->size(); 
    if (is_J_func_){ 
        Tensor<double,4> W(l,l,l,l);
        W.setZero();
        return W;
    }
    VectorXd N = gamma->n();
    MatrixXd no = gamma->no;
    Tensor<double,4> I2 = TensorCast(gamma->int2e,l,l,l,l);
    I2 = Tensor_prod(no,I2,0,0); I2 = Tensor_prod(no,I2,0,1); I2 = Tensor_prod(no,I2,0,2); I2 = Tensor_prod(no,I2,0,3);
    Tensor<double,4> W (l,l,l,l);
    for (int i=0;i<l;i++){
        for(int j=0;j<l;j++){
            for (int p=0;p<l;p++){
                for(int q=0;q<l;q++){
                    W(i,j,p,q) = 2.*N(i)*N(q)*I2(i,j,p,q);
                }
            }
        }
    }
    return W;
}
Tensor<double,4> Functional::compute_Wbar_K(RDM1* gamma) const{
    int l = gamma->size(); 
    VectorXd N = n_K_(gamma);
    MatrixXd no = gamma->no;
    Tensor<double,4> I2 = TensorCast(gamma->int2e,l,l,l,l);
    I2 = Tensor_prod(no,I2,0,0); I2 = Tensor_prod(no,I2,0,1); I2 = Tensor_prod(no,I2,0,2); I2 = Tensor_prod(no,I2,0,3);
    Tensor<double,4> W (l,l,l,l);
    for (int i=0;i<l;i++){
        for(int j=0;j<l;j++){
            for (int p=0;p<l;p++){
                for(int q=0;q<l;q++){
                    W(i,j,p,q) = N(i)*N(q)* (I2(i,j,p,q) + I2(i,q,j,p));
                }
            }
        }
    }
    return W;
}
// Compute the derivative of W_J W_K respect to the occupations
VectorXd Functional::compute_dW_J(RDM1* gamma) const{
    int l = gamma->size();
    if (is_J_func_){
        return VectorXd::Zero(l);
    }
    else{
        VectorXd dW_J = VectorXd::Zero(l);
        VectorXd N = gamma->n();
        MatrixXd v = v_J(gamma,&N,is_J_func_);
        for (int i=0;i<l;i++){
            for (int j=0;j<l;j++){
                dW_J(j) += v(i,i)*gamma->dn(i,j); 
            }
        } 
        return dW_J;
    }
}
VectorXd Functional::compute_dW_K(RDM1* gamma) const{
    return dW_K_(gamma);
}

// Compute the Hessian of W_J W_K respect to the occupations
MatrixXd Functional::compute_ddW_J(RDM1* gamma) const{
    int l = gamma->size();
    if (is_J_func_){
        return MatrixXd::Zero(l,l);
    }
    else{
        MatrixXd ddW_J (l,l);
        VectorXd N = gamma->n();
        VectorXd v = v_J(gamma,&N,is_J_func_).diagonal();
        for (int i=0;i<l;i++){
            VectorXd dv = dv_J(gamma,i).diagonal();
            for (int j=0;j<i+1;j++){
                VectorXd dn = gamma->dn(j).diagonal(); VectorXd ddn = gamma->ddn(i,j).diagonal();
                ddW_J(i,j) = (ddn.cwiseProduct(v) + dn.cwiseProduct(dv)).sum();
                ddW_J(j,i) = ddW_J(i,j);
            }
        }
        return ddW_J;
    }
    
}

MatrixXd Functional::compute_ddW_K(RDM1* gamma) const{
    int l = gamma->size();
    MatrixXd ddW_K(l,l);
    VectorXd N = n_K_(gamma);
    VectorXd v = v_K(gamma,&N).diagonal();
    for (int i=0;i<l;i++){
        VectorXd dv = dv_K(gamma,i).diagonal();
        for (int j=0;j<i+1;j++){
            VectorXd dn = dn_K_(gamma,j); VectorXd ddn = ddn_K_(gamma,i,j);
            ddW_K(i,j) = (ddn.cwiseProduct(v) + dn.cwiseProduct(dv)).sum();
            ddW_K(j,i) = ddW_K(i,j);
        }
    }
    return ddW_K;
}

// Compute the Hatree exchange correlation energy  
double Functional::E_Hxc(MatrixXd* W_J, MatrixXd* W_K) const{
    return 1./2.*( (*W_J).trace()-(*W_K).trace() );
}

// Compute the gradiant of the Hatree exchange correlation energy gradient
VectorXd Functional::dE_Hxc(RDM1* gamma, bool only_n, bool only_no) const{
    int l = gamma->size(); int ll = l*(l+1)/2; VectorXd dE2 (ll);  
    if (not only_no){
        dE2.segment(0,l) = compute_dW_J(gamma) - compute_dW_K(gamma);
    }
    if (not only_n){
        MatrixXd W_J = compute_WJ(gamma); MatrixXd W_K = compute_WK(gamma);
        int index = l;
        for (int i=0;i<l;i++){
            for (int j=0;j<i;j++){
                dE2(index) = 2.*(W_J(j,i) - W_J(i,j) - W_K(j,i) + W_K(i,j) ); 
                index ++;
            }
        }
    }
    if(only_n){ return dE2.segment(0,l);}
    else{if(only_no){return dE2.segment(l,l*(l-1)/2);}
    else{return dE2;}}
}

VectorXd Functional::dE_Hxc(RDM1* gamma, MatrixXd* W_J, MatrixXd* W_K, bool only_n, bool only_no) const{
    int l = gamma->size(); int ll = l*(l+1)/2; VectorXd dE2 (ll);  
    if (not only_no){
        dE2.segment(0,l) = compute_dW_J(gamma) - compute_dW_K(gamma);
    }
    if (not only_n){
        int index = l;
        for (int i=0;i<l;i++){
            for (int j=0;j<i;j++){
                dE2(index) = 2.*(W_J->coeff(j,i) - W_J->coeff(i,j) - W_K->coeff(j,i) + W_K->coeff(i,j) ); 
                index ++;
            }
        }
    }
    if(only_n){ return dE2.segment(0,l);}
    else{if(only_no){return dE2.segment(l,ll-l);}
    else{return dE2;}}
}

VectorXd Functional::dE_Hxc_subspace(RDM1* gamma, int g) const{
    return - dW_K_subspace_(gamma,g);
}

// Compute the Hessian of the Hatree exchange correlation energy 
MatrixXd Functional::ddE_Hxc(RDM1* gamma, bool only_n, bool only_no, bool only_coupled) const{
    int l = gamma->size(); int ll = l*(l+1)/2; MatrixXd ddE2 = MatrixXd::Zero(ll,ll);
    VectorXd NJ = gamma->n(); VectorXd NK = n_K_(gamma);
    MatrixXd vJ = v_J(gamma,&NJ,is_J_func_); MatrixXd vK = v_K(gamma,&NK);

    if ((not only_no) && (not only_coupled)){
        //ddE2.block(0,0,l,l) = compute_ddW_J(gamma) - compute_ddW_K(gamma);
        for(int i=0;i<l;i++){
           MatrixXd dvJ = dv_J(gamma,i); MatrixXd dvK = dv_K(gamma,i); 
           for (int j=0;j<=i;j++){
                ddE2(i,j) = gamma->ddn(i,j).diagonal().dot(vJ.diagonal()) + gamma->dn(j).diagonal().dot(dvJ.diagonal()) 
                            -ddn_K_(gamma,i,j).dot(vK.diagonal()) - dn_K_(gamma,j).dot(dvK.diagonal());
                ddE2(j,i) = ddE2(i,j);
           } 
        }

    }
    if((not only_n) && (not only_coupled)){ 
        Tensor<double,4> I2 ;
        I2 = TensorCast(gamma->int2e,l,l,l,l); MatrixXd no = gamma->no; 
        I2 = Tensor_prod(no,I2,0,0); I2 = Tensor_prod(no,I2,0,1); I2 = Tensor_prod(no,I2,0,2); I2 = Tensor_prod(no,I2,0,3);
        for (int i=0;i<l;i++){
            for (int j=0;j<i;j++){
                for (int p=0;p<=i;p++){
                    if (p<=j){
                    double temp = -2.*(vJ(j,p)*(NJ(p) - NJ(i)) -vK(j,p)*(NK(p) - NK(i)));
                              ddE2(i*(i-1)/2 +j +l, i*(i-1)/2 +p +l) = temp;
                    if (j!=p){ddE2(i*(i-1)/2 +p +l, i*(i-1)/2 +j +l) = temp; }
                    }
                    if (p<j){
                    double temp = 2.*(vJ(i,p)*(NJ(p) - NJ(j)) -vK(i,p)*(NK(p) - NK(j)));
                    ddE2(i*(i-1)/2 +j +l, j*(j-1)/2 +p +l) += temp;
                    ddE2(j*(j-1)/2 +p +l, i*(i-1)/2 +j +l) += temp;
                    }
                    if(j<p){
                    double temp = -2.*(vJ(i,p)*(NJ(p) - NJ(j)) -vK(i,p)*(NK(p) - NK(j)));  
                              ddE2(i*(i-1)/2 +j +l, p*(p-1)/2 +j +l) += temp;
                    if (p!=i){ddE2(p*(p-1)/2 +j +l, i*(i-1)/2 +j +l) += temp; }
                    }
                    for (int q=0;q<p;q++){  //expensive part of the Hessian
                        double temp = 4.*(NJ(i)*NJ(p) + NJ(j)*NJ(q) - NJ(i)*NJ(q) - NJ(j)*NJ(p))*I2(i,j,p,q)*(!is_J_func_)
                                                                 -2.*(NK(i)*NK(p) + NK(j)*NK(q) - NK(i)*NK(q) - NK(j)*NK(p))*(I2(i,p,j,q)+I2(i,q,j,p));
                                         ddE2(i*(i-1)/2 +j +l, p*(p-1)/2 +q +l) += temp;
                        if(i!=p || j!=q){ddE2(p*(p-1)/2 +q +l, i*(i-1)/2 +j +l) += temp;}
                    }
                }
            }
        }
    }
    if((not only_n) && (not only_no)){    
        
        VectorXd NJ = gamma->n(); VectorXd NK = n_K_(gamma);
        MatrixXd vJ = v_J(gamma,&NJ,is_J_func_); MatrixXd vK = v_K(gamma,&NK);
        int index = l;
        for (int i=0;i<l;i++){
            MatrixXd dvJ = dv_J(gamma,i); MatrixXd dvK = dv_K(gamma,i); VectorXd dNK = dn_K_(gamma,i);
            for (int p=0;p<l;p++){
                for (int q=0;q<p;q++){
                    ddE2(i,index) =2*(gamma->dn(q,i)*vJ(q,p) + NJ(q)*dvJ(q,p) - gamma->dn(p,i)*vJ(p,q) - NJ(p)*dvJ(p,q) 
                                     -dNK(q)*vK(q,p)         - NK(q)*dvK(q,p) + dNK(p)*vK(p,q)         + NK(p)*dvK(p,q));
                    ddE2(index,i) = ddE2(i,index);
                    index++;
                }
            }
            index =l;
        }
    }
    if(only_n){ return ddE2.block(0,0,l,l);}
    else{if(only_no){return ddE2.block(l,l,ll-l,ll-l);}
    else{ return ddE2;}}
}

//Compute the part of the cheap Hessian of E_Hxc
MatrixXd Functional::ddE_Hxc_k(RDM1* gamma, bool only_n, bool only_no, bool only_coupled) const{
    int l = gamma->size(); int ll = l*(l+1)/2; MatrixXd ddE2 = MatrixXd::Zero(ll,ll);
    MatrixXd C = gamma->no; MatrixXd Ct = C.transpose(); 
    VectorXd NJ = gamma->n(); VectorXd NK = n_K_(gamma);
    MatrixXd vJ = v_J(gamma,&NJ,is_J_func_); MatrixXd vK = v_K(gamma,&NK);
    if ((not only_no) && (not only_coupled)){
        //ddE2.block(0,0,l,l) = compute_ddW_J(gamma) - compute_ddW_K(gamma);
        for(int i=0;i<l;i++){
           for (int j=0;j<=i;j++){
                ddE2(i,j) = gamma->ddn(i,j).diagonal().dot(vJ.diagonal()) -ddn_K_(gamma,i,j).dot(vK.diagonal());
                ddE2(j,i) = ddE2(i,j);
           } 
        }

    }
    if((not only_n) && (not only_coupled)){ 
        for (int i=0;i<l;i++){
            for (int j=0;j<i;j++){
                for (int p=0;p<=i;p++){
                    if (p<i){
                    ddE2(i*(i-1)/2 +j +l, i*(i-1)/2 +p +l) = -2.*(vJ(j,p)*(NJ(p) - NJ(i)) -vK(j,p)*(NK(p) - NK(i)));
                        if(j!=p){    
                        ddE2(i*(i-1)/2 +p +l, i*(i-1)/2 +j +l) = ddE2(i*(i-1)/2 +j +l, i*(i-1)/2 +p +l);
                        }
                    }
                    if (p<j){
                    ddE2(i*(i-1)/2 +j +l, j*(j-1)/2 +p +l) += 2.*(vJ(i,p)*(NJ(p) - NJ(j)) -vK(i,p)*(NK(p) - NK(j)));
                    ddE2(j*(j-1)/2 +p +l, i*(i-1)/2 +j +l) += ddE2(i*(i-1)/2 +j +l, j*(j-1)/2 +p +l);
                    }
                    if (j<p){
                    ddE2(i*(i-1)/2 +j +l, p*(p-1)/2 +j +l) -= 2.*(vJ(i,p)*(NJ(p) - NJ(j)) -vK(i,p)*(NK(p) - NK(j)));    
                        if (p!=i){
                        ddE2(p*(p-1)/2 +j +l, i*(i-1)/2 +j +l) += ddE2(i*(i-1)/2 +j +l, p*(p-1)/2 +j +l);
                        }
                    }
                }
            }
        }

    }
    if((not only_n) && (not only_no)){    
        int index = l;
        for (int i=0;i<l;i++){
            VectorXd dNK = dn_K_(gamma,i);
            for (int p=0;p<l;p++){
                for (int q=0;q<p;q++){
                    ddE2(i,index) =2.*(gamma->dn(q,i)*vJ(q,p) - gamma->dn(p,i)*vJ(p,q)  
                                     -dNK(q)*vK(q,p)         + dNK(p)*vK(p,q)       );
                    ddE2(index,i) = ddE2(i,index);
                    index++;
                }
            }
            index =l;
        }
    }
    if(only_n){ return ddE2.block(0,0,l,l);}
    else{if(only_no){return ddE2.block(l,l,ll-l,ll-l);}
    else{ return ddE2;}}
}

//Compute the part of the cheap Hessian of E_Hxc in auxiliary space 
MatrixXd Functional::ddE_Hxc_aux(RDM1* gamma, bool only_n, bool only_no, bool only_coupled) const{
    int l = gamma->size(); int ll = l*l; MatrixXd ddE2 = MatrixXd::Zero(l+ll,l+ll);
    VectorXd NJ = gamma->n(); VectorXd NK = n_K_(gamma);
    MatrixXd vJ = v_J(gamma,&NJ,is_J_func_); MatrixXd vK = v_K(gamma,&NK);
    Tensor<double,4> I2 = TensorCast(gamma->int2e,l,l,l,l); MatrixXd no = gamma->no; 
    I2 = Tensor_prod(no,I2,0,0); I2 = Tensor_prod(no,I2,0,1); I2 = Tensor_prod(no,I2,0,2); I2 = Tensor_prod(no,I2,0,3);

    if ((not only_no) && (not only_coupled)){
        for(int i=0;i<l;i++){
            ddE2(i,i) = 2.*vJ(i,i)  -ddn_Kn_(gamma,i,i)*vK(i,i);
            //expensive part
            /*ddE2(i,i) += 4.*NJ(i)*I2(i,i,i,i) - pow(dn_Kn_(gamma,i),2)*I2(i,i,i,i);
            for(int j=0;j<i;j++){
                ddE2(i,j) += 4.*sqrt(NJ(i)*NJ(j))*I2(i,i,j,j) - dn_Kn_(gamma,i)*dn_Kn_(gamma,j)*I2(i,j,i,j);
                ddE2(j,i) += ddE2(i,j);
            } */        
        }
    }
    if((not only_n) && (not only_coupled)){ 
        for (int i=0;i<l;i++){
            for (int j=0;j<l;j++){
                for (int p=0;p<l;p++){
                    ddE2(l*(i+1) +j, l*(i+1) +p) += 2.* vJ(j,p)*NJ(i) - 2.* vK(j,p)*NK(i);
                    //expensive part
                    /*for (int q=0;q<l;q++){
                        ddE2(l*(i+1) +j, l*(p+1) +q) += 4.*NJ(i)*NJ(p)*(I2(i,j,p,q)) - 2.*NK(i)*NK(p)*(I2(i,p,j,q)+I2(i,q,j,p));
                    }*/
                }
            }
        }
    }
    if((not only_n) && (not only_no)){    
        for (int i=0;i<l;i++){
            for (int p=0;p<l;p++){
                ddE2(i,l*(i+1) +p) += 4.*sqrt(NJ(i))*vJ(i,p) - 2.*dn_Kn_(gamma,i)*vK(i,p);
                ddE2(l*(i+1) +p,i) += 4.*sqrt(NJ(i))*vJ(i,p) - 2.*dn_Kn_(gamma,i)*vK(i,p);
                //expensive part
                /*for (int q=0;q<l;q++){
                    ddE2(i,l*(p+1) +q) += 4.*NJ(p)*sqrt(NJ(i))*I2(i,i,p,q) - 2.*NK(p)*dn_Kn_(gamma,i)*I2(i,p,i,q);
                    ddE2(l*(p+1) +q,i) += 4.*NJ(p)*sqrt(NJ(i))*I2(i,i,p,q) - 2.*NK(p)*dn_Kn_(gamma,i)*I2(i,p,i,q);
                }*/
            }
        }
    }
    if(only_n){ return ddE2.block(0,0,l,l);}
    else{if(only_no){return ddE2.block(l,l,ll,ll);}
    else{ return ddE2;}}
}

//Compute the part of the Hessian of E_Hxc rquiering 1st order derivatives in E
MatrixXd Functional::ddJac(RDM1* gamma, bool only_n, bool only_no) const{
    int l = gamma->size(); int ll = l*(l+1)/2; MatrixXd ddE2 = MatrixXd::Zero(ll,ll);
    MatrixXd C = gamma->no; MatrixXd Ct = C.transpose(); 
    VectorXd NJ = gamma->n(); VectorXd NK = n_K_(gamma);
    MatrixXd vJ = v_J(gamma,&NJ,is_J_func_); MatrixXd vK = v_K(gamma,&NK);
    if (not only_no){
        VectorXd dnvJ (l); VectorXd dnvK (l);
        for(int i=0;i<l;i++){
            dnvJ(i) = 2.*gamma->sqrtn(i)*vJ(i,i);
            dnvK(i) = dn_Kn_(gamma,i)*vK(i,i);
        }
        for(int i=0;i<l;i++){
            for(int j=0;j<=i;j++){
                ddE2(i,j) = gamma->ddsqrt_n(i,j).diagonal().dot(dnvJ - dnvK);
                ddE2(j,i) = ddE2(i,j);
            }
        }
    }
    if(not only_n){
        MatrixXd dE2_J = 2.*vJ*gamma->n().asDiagonal();
        MatrixXd dE2_K = 2.*vK*n_K_(gamma).asDiagonal(); 
        MatrixXd dE2 = (dE2_J-dE2_K).reshaped(l*l,1); MatrixXd I = MatrixXd::Identity(l,l);
        int id1 = l; int id2 = l;
        for(int i=0;i<l;i++){
            for(int j=0;j<i;j++){
                for (int p=0;p<=i;p++){
                    for (int q=0;q<p;q++){
                        MatrixXd d2U = ddU(&I,i,j,p,q).reshaped(1,l*l);
                        ddE2(id1,id2) = (d2U*dE2)(0,0);
                        ddE2(id2,id1) = ddE2(id1,id2); id2++;
                    }
                }
                id1++; id2 = l;
            }
        }
    }
    if(only_n){ return ddE2.block(0,0,l,l);}
    else{if(only_no){return ddE2.block(l,l,ll-l,ll-l);}
    else{ return ddE2;}}
}

// Auxiliary functions for the 1 electron part
MatrixXd dU(MatrixXd* C,int i,int j){
    int l = C->rows();
    MatrixXd res = MatrixXd::Zero(l,l);
    res.col(i) = -C->col(j); res.col(j) = C->col(i);
    return res;
}

MatrixXd ddU(MatrixXd* C,int i,int j,int k,int l){
    int n = C->rows(); MatrixXd res = MatrixXd::Zero(n,n);
    if (i==k && j==l){ res.col(i) = -C->col(i); res.col(j) = -C->col(j); }
    else{
    if (i==l && j==k){ res.col(i) =  C->col(i); res.col(j) =  C->col(i);}
    else{
        if (i==l){res.col(k) =  C->col(j);}
        if (i==k){res.col(l) = -C->col(j);}
        if (j==l){res.col(k) = -C->col(i);}
        if (j==k){res.col(l) =  C->col(i);}
    }}
    return res;
}

MatrixXd outer(VectorXd v1, VectorXd v2){
    int l = v1.size();
    MatrixXd res; res = MatrixXd::Zero(l,l);
    for(int i=0;i<l;i++){
        for(int j=0;j<l;j++){
            res(i,j)= v1(i)* v2(j);
        }
    }
    return res;
}

VectorXd pow(const VectorXd* v, double p){
    int l = v->size(); VectorXd res (l);
    for (int i=0; i<l;i++){
        res(i) = pow(v->coeff(i),2);
    }
    return res;
}

