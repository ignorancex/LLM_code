#include <math.h>
#include <eigen3/Eigen/Core>
#include <iostream>

#include "numerical_deriv.hpp"
#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"

using namespace std;
using namespace Eigen;

#include<iomanip>
//Compute a numerical approximation of the gradiaent of the functional func at 1RDM gamma
//(for testing purpose)
VectorXd grad_func(Functional* func, RDM1* gamma, bool do_n, bool do_no, double epsi){
    int l = gamma->size(); int ll = l*(l+1)/2; VectorXd res (ll);
    if (do_n){
        for (int i=0;i<l;i++){
            RDM1 gamma_p (gamma); RDM1 gamma_m (gamma);
            gamma_p.x(i, gamma->x(i)+epsi) ; gamma_m.x(i, gamma->x(i)-epsi);
            gamma_p.solve_mu(); gamma_m.solve_mu();
            res(i) = (func->E(&gamma_p) - func->E(&gamma_m))/(2*epsi);
        }
    }
    if (do_no){
        for (int i=l;i<ll;i++){
            RDM1 gamma_p (gamma); RDM1 gamma_m(gamma);
            VectorXd dtheta_p = VectorXd::Zero(ll-l); dtheta_p(i-l) =  epsi;
            VectorXd dtheta_m = VectorXd::Zero(ll-l); dtheta_m(i-l) = -epsi;
            gamma_p.set_no( gamma->no*exp_unit(&dtheta_p)); gamma_m.set_no( gamma->no*exp_unit(&dtheta_m));
            res(i) = (func->E(&gamma_p) - func->E(&gamma_m))/(2.*epsi);
        }
    }
    if (do_n &&(not do_no)){return res.segment(0,l);}
    if (do_no &&(not do_n)){return res.segment(l,ll-l);}
    else{return res;}
}

const int K = 8; //Number of Richardson iterations
const double T =2.; //Numuber to wich the step is divided at each iteration

//Compute a numerical approximation of the Hessian of the functional func at 1RDM gamma
//(for testing purpose, requires analytical gradient)
MatrixXd hess_func(Functional* func, RDM1* gamma, bool do_n, bool do_no, double epsi){
    int l = gamma->size(); int ll = l*(l+1)/2; MatrixXd res =MatrixXd::Zero(ll,ll);
    if (do_n){
        for(int i=0;i<l;i++){
            RDM1 gamma_p (gamma); RDM1 gamma_m (gamma);
            gamma_p.x(i, gamma->x(i)+epsi) ; gamma_m.x(i, gamma->x(i)-epsi);
            gamma_p.solve_mu(); gamma_m.solve_mu();
            VectorXd grad_p = func->grad_E(&gamma_p,true,false); VectorXd grad_m = func->grad_E(&gamma_m,true,false);
            res.block(i,0,1,l) = (grad_p-grad_m).transpose()/(2.*epsi); 
            
             //8th order
            RDM1 gamma_p4 (gamma); RDM1 gamma_p3 (gamma); RDM1 gamma_p2 (gamma); RDM1 gamma_p1 (gamma);
            RDM1 gamma_m4 (gamma); RDM1 gamma_m3 (gamma); RDM1 gamma_m2 (gamma); RDM1 gamma_m1 (gamma);
            gamma_p4.x(i, gamma->x(i)+4.*epsi); gamma_p3.x(i, gamma->x(i)+3.*epsi); gamma_p2.x(i, gamma->x(i)+2.*epsi); gamma_p1.x(i, gamma->x(i)+epsi);
            gamma_m4.x(i, gamma->x(i)-4.*epsi); gamma_m3.x(i, gamma->x(i)-3.*epsi); gamma_m2.x(i, gamma->x(i)-2.*epsi); gamma_m1.x(i, gamma->x(i)-epsi);
            gamma_p4.solve_mu(); gamma_p3.solve_mu(); gamma_p2.solve_mu(); gamma_p1.solve_mu();
            gamma_m4.solve_mu(); gamma_m3.solve_mu(); gamma_m2.solve_mu(); gamma_m1.solve_mu();
            VectorXd grad_p4 = func->grad_E(&gamma_p4,true,false); VectorXd grad_p3 = func->grad_E(&gamma_p3,true,false); 
            VectorXd grad_p2 = func->grad_E(&gamma_p2,true,false); VectorXd grad_p1 = func->grad_E(&gamma_p1,true,false); 
            VectorXd grad_m4 = func->grad_E(&gamma_m4,true,false); VectorXd grad_m3 = func->grad_E(&gamma_m3,true,false); 
            VectorXd grad_m2 = func->grad_E(&gamma_m2,true,false); VectorXd grad_m1 = func->grad_E(&gamma_m1,true,false); 
            res.block(i,0,1,l) = (-1./280.*grad_p4+4./105.*grad_p3-1./5.*grad_p2+4./5.*grad_p1-4./5.*grad_m1+1./5.*grad_m2-4./105.*grad_m3+1./280.*grad_m4).transpose()/(epsi); 
        }
    }
    if (do_no){
        for(int i=l;i<ll;i++){  
            RDM1 gamma_p (gamma); RDM1 gamma_m (gamma);
            VectorXd dtheta_p = VectorXd::Zero(ll-l); dtheta_p(i-l) += epsi;
            VectorXd dtheta_m = VectorXd::Zero(ll-l); dtheta_m(i-l) -= epsi;
            gamma_p.set_no( gamma->no*exp_unit(&dtheta_p)); gamma_m.set_no( gamma->no*exp_unit(&dtheta_m));
            VectorXd grad_p = func->grad_E(&gamma_p,false,true); VectorXd grad_m = func->grad_E(&gamma_m,false,true);
            for (int j=l;j<ll;j++){
                res(i,j) = (grad_p(j-l)-grad_m(j-l))/(2.*epsi);
                res(j,i) = res(i,j);
            }
            //res.block(i,l,1,ll-l) = (grad_p-grad_m).transpose()/(2.*epsi); 
            
            


        }
    }
    if(do_n && do_no){
        for(int i=0;i<l;i++){
            
            RDM1 gamma_p (gamma); RDM1 gamma_m (gamma);
            gamma_p.x(i, gamma->x(i)+epsi) ; gamma_m.x(i, gamma->x(i)-epsi);
            gamma_p.solve_mu(); gamma_m.solve_mu();
            VectorXd grad_p = func->grad_E(&gamma_p,false,true); VectorXd grad_m = func->grad_E(&gamma_m,false,true);
            res.block(i,l,1,ll-l) = (grad_p-grad_m).transpose()/(2.*epsi);
            res.block(l,i,ll-l,1) = (grad_p-grad_m)/(2.*epsi);
        }
    }
    if (do_n &&(not do_no)){return res.block(0,0,l,l);}
    if (do_no &&(not do_n)){return res.block(l,l,ll-l,ll-l);}   
    else{return res;}
}

//Kronecker delta
MatrixXd delta(int l,int i, int j){
    MatrixXd res = MatrixXd::Zero(l,l);
    res(i,j) = 1.;
    return res;
}
