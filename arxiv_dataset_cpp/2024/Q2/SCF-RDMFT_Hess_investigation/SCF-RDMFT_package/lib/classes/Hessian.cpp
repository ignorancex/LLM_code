#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <limits>
#include <cmath>
#include <iostream>
#include <deque>

#include "1RDM_class.hpp"


using namespace std;
using namespace Eigen;

const double precision = 0.1*sqrt(DBL_EPSILON);

void H_init(MatrixXd* hess_, VectorXd s, VectorXd y, int l){ 
    int ls = s.size();
    (*hess_) = MatrixXd::Zero(ls,ls);
    //if (l==0){
        double sigma = y.dot(s)/y.squaredNorm();
        for (int i=0;i<ls;i++){
            (*hess_)(i,i) = sigma;
        }
    /*}
    else{
        VectorXd y_n = y.segment(0,l); VectorXd s_n = s.segment(0,l); 
        double sigma_n = y_n.dot(s_n)/s_n.squaredNorm();
        for (int i=0;i<l;i++){
            (*hess_)(i,i) = sigma_n;
        }
        VectorXd y_no = y.segment(l,ls-l); VectorXd s_no = s.segment(l,ls-l); 
        double sigma_no = y_no.dot(s_no)/s_no.squaredNorm();
        for (int i=l;i<ls;i++){
            (*hess_)(i,i) = sigma_no;
        }
    }*/
}

void SR1(void* f_data){ 
    data_struct *data = (data_struct*) f_data; 
    if(data->niter==0){ return ;}
    VectorXd step = data->x2 - data->x1; VectorXd psi = data->grad2 - data->grad1 - data->hess_cheap_*step;
    if(data->niter==1 && data->do_1st_iter){H_init(&data->hess_exp_,step,psi,data->gamma->size());}
    double sNorm = step.norm(); 
    VectorXd u = psi -data->hess_exp_*step; double uNorm = u.norm();
    double sigma = u.dot(step); 
    if (abs(sigma) >precision*sNorm*uNorm){
        data->hess_exp_ += u*u.transpose()/sigma; 
    }
}

void BFGS(void* f_data){
    data_struct *data = (data_struct*) f_data; 
    if(data->niter==0){ return ;}
    VectorXd step = data->x2 - data->x1; VectorXd psi = data->grad2 - data->grad1 - data->hess_cheap_*step;
    if(data->niter==1 && data->do_1st_iter){ H_init(&data->hess_exp_,step,psi,data->gamma->size());}
    double sNorm = step.norm(); double sTpsi = step.dot(psi);
    VectorXd Hs = data->hess_exp_*step;
    double sHs = step.dot(Hs);
    data->hess_exp_ += psi*psi.transpose()/sTpsi - Hs*Hs.transpose()/sHs;
}

void DFP(void* f_data){
    data_struct *data = (data_struct*) f_data; 
    if(data->niter==0){ return ;}
    VectorXd step = data->x2 - data->x1; VectorXd psi = data->grad2 - data->grad1 - data->hess_cheap_*step;
    if(data->niter==1 && data->do_1st_iter){H_init(&data->hess_exp_,step,psi,data->gamma->size());}
    double sNorm = step.norm(); double sTpsi = step.dot(psi);
    MatrixXd u = MatrixXd::Identity(step.size(),step.size()) - psi*step.transpose()/sTpsi;
    data->hess_exp_ = u*data->hess_*u.transpose() + psi*psi.transpose()/sTpsi;
}

void Broyden(void* f_data){
    data_struct *data = (data_struct*) f_data; 
    if(data->niter==0){ return ;}
    VectorXd step = data->x2 - data->x1; VectorXd psi = data->grad2 - data->grad1 - data->hess_cheap_*step;
    if(data->niter==1 && data->do_1st_iter){ H_init(&data->hess_exp_,step,psi,data->gamma->size());}
    double sNorm = step.norm(); double sTpsi = step.dot(psi);
    
    //BFGS part
    VectorXd Hs = data->hess_exp_*step;
    double sHs = step.dot(Hs);
    MatrixXd H_BFGS = data->hess_exp_ + psi*psi.transpose()/sTpsi - Hs*Hs.transpose()/sHs;
    
    //DFP part
    MatrixXd u = MatrixXd::Identity(step.size(),step.size()) - psi*step.transpose()/sTpsi;
    MatrixXd H_DFP = data->hess_exp_ + u*data->hess_*u.transpose() + psi*psi.transpose()/sTpsi;
    
    MatrixXd Hinv = data->hess_exp_.inverse(); // /!\ inefficiant methode 
    VectorXd Hpsi = Hinv*psi; double psiHpsi = psi.dot(Hpsi);
    double phi;
    if( sTpsi<= 2.*sHs*psiHpsi/(sHs+psiHpsi)){  phi = sTpsi*(psiHpsi-sTpsi)/(sHs*psiHpsi-sTpsi*sTpsi); }
    else{ phi = sTpsi/(sTpsi-sHs); }

    data->hess_exp_ = (1.-phi)*H_BFGS + phi*H_DFP;

}

void SR1_aux(void* f_data){
    data_struct *data = (data_struct*) f_data; 
    if(data->niter==0){ return ;}
    int l = data->gamma->size();
    VectorXd step = data->x2 - data->x1; VectorXd psi = data->grad2 - data->grad1 - data->hess_cheap_*step;
    MatrixXd J = data->func->Jac(data->gamma); MatrixXd Jt = J.transpose();
    MatrixXd Jinv = data->func->InvJac(data->gamma).transpose(); 
    VectorXd s = J*step; VectorXd y = Jinv*psi;
    if(data->niter==1 && data->do_1st_iter){ data->hess_exp_nu = 1e-6*MatrixXd::Identity(l+l*l, l+l*l);}
    double sNorm = s.norm(); 
    VectorXd u = y -data->hess_exp_nu*s; double uNorm = u.norm();
    double sigma = u.dot(s); 
    if(abs(sigma) >precision*sNorm*uNorm){
        data->lu.push_back(u); data->luTu.push_back(sigma);
        if(data->lu.size()> data->memory){
            data->lu.pop_front(); data->luTu.pop_front();
        }
        data->hess_exp_nu += u*u.transpose()/sigma;
        for (int i=0;i<data->lu.size();i++){
            VectorXd ui = data->lu[i];
            VectorXd Ju= Jt*u;
            data->hess_exp_ += Ju*Ju.transpose()/data->luTu[i];
        }
        data->update_hess = true;
        
    }
    else{
        data->update_hess = false;
    }
}

void BFGS_aux(void* f_data){//also called bBFGS or nuBFGS
    data_struct *data = (data_struct*) f_data; 
    if(data->niter==0){ return ;}
    int l = data->gamma->size(); 
    VectorXd step = data->x2 - data->x1; VectorXd psi = data->grad2 - data->grad1 - data->hess_cheap_*step;
    MatrixXd J = data->func->Jac(data->gamma); MatrixXd Jt = J.transpose();
    MatrixXd Jinv = data->func->InvJac(data->gamma).transpose();  
    VectorXd s = J*step; VectorXd y = Jinv*psi;
    if(data->niter==1 && data->do_1st_iter){ data->hess_exp_nu = 1e-6*MatrixXd::Identity(l*l+l,l*l+l);} //look for soemthing more 'subtle' but better than yTs/yTy, and 0 for some reason
    double sNorm = s.norm(); 
    VectorXd Hs = data->hess_exp_nu*s; double HsNorm = Hs.norm();
    double sTy = s.dot(y); double sHs = s.dot(Hs);
    data->hess_exp_nu += y*y.transpose()/sTy - Hs*Hs.transpose()/sHs;  
    data->lu.push_back(y); data->lv.push_back(Hs); data->luTu.push_back(sTy); data->lvTv.push_back(sHs);
    if(data->lu.size()> data->memory){
        data->lu.pop_front(); data->lv.pop_front(); data->luTu.pop_front(); data->lvTv.pop_front();
    }
    for (int i=0;i<data->lu.size();i++){ // O(N_basis^4 min(N_memory,N_iter))
            VectorXd u = data->lu[i]; VectorXd v = data->lv[i];
            VectorXd Ju= Jt*u; VectorXd Jv = Jt*v;
            data->hess_exp_ += Ju*Ju.transpose()/data->luTu[i] - Jv*Jv.transpose()/data->lvTv[i];
    }
}

void tBFGS_aux(void* f_data){//also called xBFGS
    data_struct *data = (data_struct*) f_data; 
    if(data->niter==0){ return ;}
    VectorXd step = data->x2 - data->x1; VectorXd y = data->grad2 - data->grad1;
    MatrixXd J = data->func->Jac(data->gamma); MatrixXd Jt = J.transpose();
    MatrixXd Jinv = data->func->InvJac(data->gamma).transpose();  
    MatrixXd Htilde = Jt*data->hess_exp_nu*J + data->hess_cheap_; 
    VectorXd Hs = Htilde*step;
    VectorXd u = Jinv*y; VectorXd v = Jinv*Hs;  
    if(data->niter==1 && data->do_1st_iter){ 
        int l = data->gamma->size();
        data->hess_exp_nu = MatrixXd::Zero(l+l*l,l+l*l); }
    double sHs = step.dot(Hs); double sTy = step.dot(y);
    data->hess_exp_nu += u*u.transpose()/sTy - v*v.transpose()/sHs;
    data->lu.push_back(u); data->lv.push_back(v); data->luTu.push_back(sTy); data->lvTv.push_back(sHs);
    if(data->lu.size()> data->memory){
        data->lu.pop_front(); data->lv.pop_front(); data->luTu.pop_front(); data->lvTv.pop_front();
    }
    for (int i=0;i<data->lu.size();i++){ // O(N_basis^4 min(N_memory,N_iter))
            VectorXd u = data->lu[i]; VectorXd v = data->lv[i];
            VectorXd Ju= Jt*u; VectorXd Jv = Jt*v;
            data->hess_exp_ += Ju*Ju.transpose()/data->luTu[i] - Jv*Jv.transpose()/data->lvTv[i];
    }
}

void sBFGS_aux(void* f_data){  
    /* Inefficient version */
    data_struct *data = (data_struct*) f_data; 
    if(data->niter==0){ return ;}
    MatrixXd J = data->func->Jac(data->gamma); MatrixXd Jt = J.transpose();
    MatrixXd Jinv = data->func->InvJac(data->gamma).transpose(); 
    VectorXd step = data->x2 - data->x1; VectorXd y = data->grad2 - data->grad1 -(ddE1(data->gamma) - Jt*data->func->ddE_Hxc_aux(data->gamma)*J)*step; 
    MatrixXd Htilde = Jt*data->hess_exp_nu*J  + data->func->ddJac(data->gamma); 
    VectorXd Hs = Htilde*step;
    VectorXd u = Jinv*y; VectorXd v = Jinv*Hs;  
    if(data->niter==1 && data->do_1st_iter){ 
        int l = data->gamma->size();
        data->hess_exp_nu = MatrixXd::Zero(l+l*l,l+l*l); }
    double sHs = step.dot(Hs); double sTy = step.dot(y);
    data->hess_exp_nu += u*u.transpose()/sTy - v*v.transpose()/sHs;
    data->lu.push_back(u); data->lv.push_back(v); data->luTu.push_back(sTy); data->lvTv.push_back(sHs);
    if(data->lu.size()> data->memory){
        data->lu.pop_front(); data->lv.pop_front(); data->luTu.pop_front(); data->lvTv.pop_front();
    }
    for (int i=0;i<data->lu.size();i++){ // O(N_basis^4 min(N_memory,N_iter))
            VectorXd u = data->lu[i]; VectorXd v = data->lv[i];
            VectorXd Ju= Jt*u; VectorXd Jv = Jt*v;
            data->hess_exp_ += Ju*Ju.transpose()/data->luTu[i] - Jv*Jv.transpose()/data->lvTv[i];
    }
}

void dBFGS_aux(void* f_data){  
    data_struct *data = (data_struct*) f_data; 
    if(data->niter==0){ return ;}
    MatrixXd J = data->func->Jac(data->gamma); MatrixXd Jt = J.transpose();
    MatrixXd Jinv = data->func->InvJac(data->gamma); 
    MatrixXd Jtinv = Jinv.transpose();
    VectorXd step = data->x2 - data->x1; VectorXd rho = data->grad2 - data->grad1 -Jtinv*data->hess_cheap_*Jinv*step; 
    MatrixXd Htilde = data->hess_exp_nu  + Jtinv*data->hess_cheap_*Jinv; 
    VectorXd Hs = Htilde*step; 
    if(data->niter==1 && data->do_1st_iter){ 
        int l = data->gamma->size();
        data->hess_exp_nu = MatrixXd::Zero(l+l*l,l+l*l); }
    double sHs = step.dot(Hs); double sTrho = step.dot(rho);
    data->hess_exp_nu += rho*rho.transpose()/sTrho - Hs*Hs.transpose()/sHs;
    data->lu.push_back(rho); data->lv.push_back(Hs); data->luTu.push_back(sTrho); data->lvTv.push_back(sHs);
    if(data->lu.size()> data->memory){
        data->lu.pop_front(); data->lv.pop_front(); data->luTu.pop_front(); data->lvTv.pop_front();
    }
    for (int i=0;i<data->lu.size();i++){ // O(N_basis^4 min(N_memory,N_iter))
            VectorXd u = data->lu[i]; VectorXd v = data->lv[i];
            VectorXd Ju= Jt*u; VectorXd Jv = Jt*v;
            data->hess_exp_ += Ju*Ju.transpose()/data->luTu[i] - Jv*Jv.transpose()/data->lvTv[i];
    }
}

MatrixXd vMap(deque<VectorXd> v){
    //Converts a deque<VectorXd> to MatrixXd
    //Rmk: deque<VectorXd> has non contiguous memory storage so cannot avoid the loop
    int cols = v.size(); int rows = v[0].size(); MatrixXd res (rows,cols);
    for (int i=0;i<cols;i++){
        res.col(i) = v[cols-i-1]; //consistancy with the reference paper; 
    } 
    return res;
}

void LBFGS(void* f_data){
    // https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.139.9400&rep=rep1&type=pdf Equation (2.3)
   // Version incompatible with trust-region 
   
    data_struct *data = (data_struct*) f_data; 
    if(data->niter==0){ return ;} 
    VectorXd step = data->x2 - data->x1; VectorXd psi = data->grad2 - data->grad1 - data->hess_cheap_*step;
    if(data->niter==1 && data->do_1st_iter){ H_init(&data->hess_exp_,step,psi,data->gamma->size());}
    
    double lambda = step.dot(psi)/step.squaredNorm(); //this scaling seems (numericaly) to work better than yTy/sTy

    data->lstep.push_back(step); data->ly.push_back(psi);
    if(data->lstep.size()> data->memory){
        data->lstep.pop_front(); data->ly.pop_front();
    }
    
    MatrixXd S = vMap(data->lstep); double SNorm = S.norm(); S /= SNorm;
    MatrixXd Y = vMap(data->ly)/SNorm; 
    MatrixXd STY = S.transpose()*Y;
    MatrixXd STS = S.transpose()*S;
    MatrixXd D = STY.diagonal().asDiagonal(); MatrixXd L = STY.triangularView<StrictlyLower>();
    
    MatrixXd U(S.rows(),2*S.cols()); MatrixXd Gamma (D.rows()+S.cols(),D.cols()+S.cols());
    U.block(0,0,S.rows(),S.cols()) = lambda*S;    U.block(0,S.cols(),S.rows(),S.cols()) = Y;
    Gamma.block(0,0,D.rows(),D.cols()) = -D;      Gamma.block(0,D.cols(),L.cols(),L.rows()) = L.transpose();
    Gamma.block(D.rows(),0,L.rows(),L.cols()) = L;Gamma.block(D.rows(),D.cols(),S.cols(),S.cols()) = lambda*STS;
    MatrixXd InvGamma = Gamma.inverse(); 
    if (InvGamma.norm()>precision){
        data->hess_exp_ = /*lambda* MatrixXd::Identity(ll,ll)*/ -U*InvGamma*U.transpose();
    }
    else{ 
        cout<<"Hessian not updated : too small denominator."<<endl;
    }
}


void LBFGS_aux(void* f_data){
    data_struct *data = (data_struct*) f_data; 
    if(data->niter==0){ return ;}
    VectorXd step = data->x2 - data->x1; VectorXd psi = data->grad2 - data->grad1 - data->hess_cheap_*step;
    MatrixXd J = data->func->Jac(data->gamma); MatrixXd Jt = J.transpose();
    MatrixXd Jinv = data->func->InvJac(data->gamma).transpose();    
    VectorXd s = J*step; VectorXd y = Jinv*psi;
    double lambda = s.dot(y)/s.squaredNorm();

    data->lstep.push_back(s); data->ly.push_back(y);
    if(data->lstep.size()> data->memory){
        data->lstep.pop_front(); data->ly.pop_front();
    }
    MatrixXd S = vMap(data->lstep); double SNorm = S.norm(); S /= SNorm;
    MatrixXd Y = vMap(data->ly)/SNorm; 
    MatrixXd STY = S.transpose()*Y;
    MatrixXd STS = S.transpose()*S;
    MatrixXd D = STY.diagonal().asDiagonal(); MatrixXd L = STY.triangularView<StrictlyLower>();
    
    MatrixXd U(S.rows(),2*S.cols()); MatrixXd Gamma (D.rows()+S.cols(),D.cols()+S.cols());
    U.block(0,0,S.rows(),S.cols()) = lambda*S;    U.block(0,S.cols(),S.rows(),S.cols()) = Y;
    Gamma.block(0,0,D.rows(),D.cols()) = -D;      Gamma.block(0,D.cols(),L.cols(),L.rows()) = L.transpose();
    Gamma.block(D.rows(),0,L.rows(),L.cols()) = L;Gamma.block(D.rows(),D.cols(),S.cols(),S.cols()) = lambda*STS;
    MatrixXd InvGamma = Gamma.inverse(); MatrixXd JU = Jt*U;
    if (InvGamma.norm()>precision){
        data->hess_exp_nu = /*lambda* MatrixXd::Identity(l2,l2)*/ -U*InvGamma*U.transpose();
        data->hess_exp_   = -JU*InvGamma*U.transpose();
    }
    else{ 
        cout<<"Hessian not updated : too small denominator."<<endl;
    }
}

void LbBFGS_aux(void* f_data){
    data_struct *data = (data_struct*) f_data; 
    if(data->niter==0){ return ;}
    VectorXd step = data->x2 - data->x1; VectorXd psi = data->grad2 - data->grad1 - data->hess_cheap_*step;
    MatrixXd J = data->func->Jac(data->gamma); MatrixXd Jt = J.transpose();
    MatrixXd Jinv = data->func->InvJac(data->gamma).transpose();  
    VectorXd s = J*step; VectorXd v = data->hess_exp_nu*s; 
    VectorXd u = Jinv*psi;
    double lambda = s.dot(u)/s.squaredNorm();

    data->lstep.push_back(s); data->ly.push_back(u);
    if(data->lstep.size()> data->memory){
        data->lstep.pop_front(); data->ly.pop_front();
    }
    MatrixXd S = vMap(data->lstep); double SNorm = S.norm(); S /= SNorm;
    MatrixXd Y = vMap(data->ly)/SNorm; 
    MatrixXd STY = S.transpose()*Y;
    MatrixXd STS = S.transpose()*S;
    MatrixXd D = STY.diagonal().asDiagonal(); MatrixXd L = STY.triangularView<StrictlyLower>();
    
    MatrixXd U(S.rows(),2*S.cols()); MatrixXd Gamma (D.rows()+S.cols(),D.cols()+S.cols());
    U.block(0,0,S.rows(),S.cols()) = lambda*S;    U.block(0,S.cols(),S.rows(),S.cols()) = Y;
    Gamma.block(0,0,D.rows(),D.cols()) = -D;      Gamma.block(0,D.cols(),L.cols(),L.rows()) = L.transpose();
    Gamma.block(D.rows(),0,L.rows(),L.cols()) = L;Gamma.block(D.rows(),D.cols(),S.cols(),S.cols()) = lambda*STS;
    MatrixXd InvGamma = Gamma.inverse(); MatrixXd JU = Jt*U;
    if (InvGamma.norm()>precision){
        data->hess_exp_nu = /*lambda* MatrixXd::Identity(l2,l2)*/ -U*InvGamma*U.transpose();
        data->hess_exp_   = -JU*InvGamma*U.transpose();
    }
    else{ 
        cout<<"Hessian not updated : too small denominator."<<endl;
    }
}


void ZERO (void* f_data){
    data_struct *data = (data_struct*) f_data; 
    int l = data->gamma->size(); int ll = l*(l+1)/2;     
    data->hess_exp_ = MatrixXd::Zero(ll,ll);
}

void ZERO_aux (void* f_data){
    data_struct *data = (data_struct*) f_data; 
    int l = data->gamma->size(); int l2 = l*l;    
    data->hess_exp_ = MatrixXd::Zero(l+l2,l+l2);
}

