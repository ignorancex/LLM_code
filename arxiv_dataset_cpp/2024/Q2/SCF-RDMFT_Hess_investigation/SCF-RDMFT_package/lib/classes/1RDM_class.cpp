#include <stdio.h>
#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <float.h>
#include <iostream>
#include <fstream>
#include <iomanip>      
#include <chrono>
#include <tuple>
#include <vector> 

#include "1RDM_class.hpp"
#include "Functional_class.hpp"
#include "../tools.hpp"
#include "EBI_add.hpp"

#include "../../fides-cpp-main/src/fides.cpp"
#include "../../fides-cpp-main/src/hessian_approximation.cpp"
#include "../../fides-cpp-main/src/minimize.cpp"
#include "../../fides-cpp-main/src/stepback.cpp"
#include "../../fides-cpp-main/src/steps.cpp"
#include "../../fides-cpp-main/src/subproblem.cpp"
#include "../../fides-cpp-main/src/trust_region.cpp"

using namespace std;
using namespace Eigen;

/*Constructor for the 1RDM
\param result the empty 1RDM
*/
RDM1::RDM1(){
    MatrixXd null (0,0); 
    n_elec = 0;
    E_nuc  = 0.;
    ovlp   = null;
    int1e  = null;
    int2e  = null; 
    int2e_x = null;
    vector<int> v(1,0); 
    omega.push_back(v);
    mu = VectorXd(0);
    V_ = VectorXd(0);
    W_ = VectorXd(0);
    x_ = VectorXd(0); 
    no = null;
    computed_V_ = VectorXi(0);
    computed_W_ = VectorXi(0);
}
/*Constructor for the 1RDM
\param args ne: number of electron
            Enuc: nuclei energy 
            overlap: overlap matrix
            elec1int: 1electron integrals matrix
            elec2int: 2electrons intergrals matrix
            exch2int: 2electrons intergrals matrix permutated to give the exchange energy
\param result the corresponding 1RDM
*/
RDM1::RDM1(int ne, double Enuc,MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int, MatrixXd exch2int){
    n_elec = ne;
    E_nuc = Enuc;
    ovlp = overlap;
    int1e = elec1int;
    int2e = elec2int;
    int2e_x = exch2int;
    int l= overlap.rows();
    vector<int> v(l); iota(v.begin(), v.end(), 0);
    omega.push_back(v);
    mu = VectorXd::Constant(1,0.);
    V_ = VectorXd::Constant(1,0.);
    W_ = VectorXd::Constant(1,0.);
    computed_V_ = VectorXi::Constant(1,false);
    computed_W_ = VectorXi::Constant(1,false);
    no = overlap.inverse().sqrt();
    x_.resize(overlap.rows()); 
    if (ne>l){
        for (int i= 0;i<l; i++){
            if (i>ne){ set_n(i,2.); }
            else {set_n(i,1.);} 
        }
    }
    else {
        for (int i= 0;i<ne; i++){
            set_n(i,1.);
        } 
    } 
}
/*Constructor for the 1RDM
\param args n: vector of the occupations
            no: matrix of the NOs
            ne: number of electron
            Enuc: nuclei energy 
            overlap: overlap matrix
            elec1int: 1electron integrals matrix
            elec2int: 2electrons intergrals matrix
            exch2int: 2electrons intergrals matrix permutated to give the exchange energy
\param result the corresponding 1RDM
*/
RDM1::RDM1(VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc,MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int, MatrixXd exch2int){
    
    n_elec = ne;
    E_nuc = Enuc;
    ovlp = overlap;
    int1e = elec1int;
    int2e = elec2int;
    int2e_x = exch2int;
    vector<int> v(occ.size()); iota(v.begin(), v.end(), 0);
    omega.push_back(v);
    mu = VectorXd::Constant(1,0.);
    V_ = VectorXd::Constant(1,0.);
    W_ = VectorXd::Constant(1,0.);
    computed_V_ = VectorXi::Constant(1,false);
    computed_W_ = VectorXi::Constant(1,false);
    x_.resize(occ.size());
    set_sqrtn(occ); //Initialise x
    no = orbital_mat;
}
/*Copy a instance of the RDM1 class*/
RDM1::RDM1(const RDM1* gamma){
    n_elec = gamma->n_elec;
    E_nuc  = gamma->E_nuc;
    ovlp   = gamma->ovlp;
    int1e  = gamma->int1e;
    int2e  = gamma->int2e;
    int2e_x = gamma->int2e_x;
    x_  = gamma->x_;
    no = gamma->no;
    mu = gamma->mu;
    V_ = gamma->V_;
    W_ = gamma->W_;
    computed_V_ = gamma->computed_V_;
    computed_W_ = gamma->computed_W_;
    omega = gamma->omega;   
}
RDM1::~RDM1(){}; 

/* Affectation to x and its elements 
(needs to be a method to update computed_V_) */
void RDM1::x(int i, double xi){
    x_(i) = xi;
    int g = find_subspace(i);
    computed_V_(g) = false;
    computed_W_(g) = false;
}

void RDM1::x(VectorXd x0){
    x_ = x0;
    computed_V_ = VectorXi::Constant(omega.size(),false);
    computed_W_ = VectorXi::Constant(omega.size(),false);
}

/*Get the index of the subspace containing index i*/
int RDM1::find_subspace(int i) const{
    if(omega.size()==1){ //avoid useless loop if no subspaces
        return 0;
    }
    else{
        int g=-1; int j=0;
        while(g==-1){
            if (find(omega[j].begin(), omega[j].end(), i) != omega[j].end()){
                g=j;
            }
            j++;
        }
        return g;
    }
}

/*Get the ith occupation from x and mu */
double RDM1::n(int i) const{
    return pow(sqrtn(i),2.);
}
/*Get all the occupations from x and mu */
VectorXd RDM1::n() const{
    return sqrtn().cwiseAbs2();
}
/* Get the squared of the ith occupation */
double RDM1::sqrtn(int i)const{
    int g = find_subspace(i);
    return RSQRT_TWO*(erf(x(i)+mu(g))+1.);
}
/* Get all the squaered roots of the occupations */
VectorXd RDM1::sqrtn()const{
    VectorXd res (size());
    for (int i=0;i<size();i++){
        res(i) = sqrtn(i);
    }
    return res;
}

/*Compute x_i cooresponding to the given occupation*/ 
void RDM1::set_n(int i,double ni){
    int g = find_subspace(i); 
    if (ni>2.-4e-8){ x(i,4.);}
    else if (ni<4e-8){ x(i,-4.);}
    else {x(i, erfinv(SQRT_TWO*sqrt(ni)-1.)-mu(g));}
}
/*Compute x corresponding to the given occupations*/
void RDM1::set_n(VectorXd n){
    for (int i=0;i<n.size();i++){
        set_n(i,n(i));
    }
}
/*Compute x_i cooresponding to the given square root occupation*/ 
void RDM1::set_sqrtn(int i,double ni){
    int g = find_subspace(i); 
    if (ni>SQRT_TWO -3e-8){ x(i,4.);}
    else if (ni<=3e-8){ x(i,-4.);}
    else {x(i, erfinv(SQRT_TWO*ni-1.)-mu(g));}
}
/*Compute x corresponding to the given square root occupations*/
void RDM1::set_sqrtn(VectorXd n){
    for (int i=0;i<n.size();i++){
        set_sqrtn(i,n(i));
    }
}

/*Compute V (used in computation of derivatives)*/
double RDM1::get_V(int g){
    if (computed_V_(g)){
        return V_(g);
    }
    else{
        V_(g) = 0; 
        for (int i:omega[g]){
            V_(g) += derf(x(i)+mu(g))*(erf(x(i)+mu(g))+1.);
        }
        if (V_(g)<1e-15){ //to avoid numerical issues /!\ modifies the result.
            V_(g)= 1e-9; 
        }
        computed_V_(g) = true;
        return V_(g);
    }
}
/*Compute W */
double RDM1::get_W(int g){
    if (computed_W_(g)){
        return W_(g);
    }
    else{
        W_(g) = 0;
        for (int i:omega[g]){
            W_(g) += pow(derf(x(i)+mu(g)),2.) 
                    + dderf(x(i)+mu(g))*(erf(x(i)+mu(g))+1.);
        } 
        computed_W_(g) = true;
        return W_(g);
    }
}

/*Computes the matrix form of the 1RDM*/
MatrixXd RDM1::mat() const {
    int l = size(); 
    MatrixXd N = n().asDiagonal();
    return no*N* no.transpose();
}
/* Derivative of ith occupation respect to the parameters jth parameter x_j */
double RDM1::dn(int i, int j){
    int f = find_subspace(i);
    int g = find_subspace(j);
    if (f==g){
        double dmu = -derf(x(j)+mu(f))*(erf(x(j)+mu(f))+1.) /get_V(f);
        double derf_i = derf(x(i)+mu(f));
        double  erf_i = erf(x(i)+mu(f))+1.;
        return derf_i*erf_i*((i==j)+ dmu);

    }
    else{
        return 0.;
    }
}
/* Derivative of the occupations respect to the parameters ith parameter x_i */
MatrixXd RDM1::dn(int i){
    int l = size(); VectorXd res(l);
    for (int j=0;j<l;j++){
        res(j) = dn(j,i);
    }
    return res.asDiagonal();
}
/* 2nd derivative of kth occupation respect to the ith and jth parameters */
double RDM1::ddn(int k,int i,int j){
    int f  = find_subspace(k); 
    int g1 = find_subspace(i);
    int g2 = find_subspace(j);
    if (f==g1 && f==g2){
        double erf_i  = erf(x(i)+mu(f))+1.;
        double erf_j  = erf(x(j)+mu(f))+1.;
        double erf_k  = erf(x(k)+mu(f))+1.;
        double derf_i = derf(x(i)+mu(f));
        double derf_j = derf(x(j)+mu(f));
        double derf_k = derf(x(k)+mu(f));
        double dmu_i = -derf_i*erf_i/get_V(f);
        double dmu_j = -derf_j*erf_j/get_V(f);
        double dderf_i = dderf(x(i)+mu(f));
        double dderf_j = dderf(x(j)+mu(f));
        double dderf_k = dderf(x(k)+mu(f));
        double f_i = derf_i*derf_i + erf_i*dderf_i;
        double f_j = derf_j*derf_j + erf_j*dderf_j;
        double ddmu  = -(get_W(f)*dmu_i*dmu_j+ f_i*dmu_j+ f_j*dmu_i +(i==j)*f_i)/get_V(f);
        return derf_k*((i==k)+dmu_i)*derf_k*((j==k)+dmu_j) 
                + erf_k*(dderf_k * ((i==k)+dmu_i) * ((j==k)+dmu_j) + derf_k*ddmu);
    }
    else{
        return 0;
    }
}
/* 2nd derivative of the occupations respect to the ith and jth parameter */
MatrixXd RDM1::ddn(int i, int j){
    int l = size(); VectorXd res(l);
    for (int k=0;k<l;k++){
        res(k) = ddn(k,i,j);
    }
    return res.asDiagonal();
}


/* Derivative of the square root of ith occupation respect to the parameters jth parameter x_j */
double RDM1::dsqrt_n(int i,int j){
    int f = find_subspace(i);
    int g = find_subspace(j);
    if(f==g){
        double derf_i = derf(x(i)+mu(f));
        double dmu = -derf(x(j)+mu(f))*(erf(x(j)+mu(f))+1.)/get_V(f);
        return RSQRT_TWO*((i==j)+dmu)*derf_i;
    }
    else{
        return 0;
    }
}
/* Derivative of the square root of the occupations respect to the parameters ith parameter x_i */
MatrixXd RDM1::dsqrt_n(int i){
    int l = size(); VectorXd res(l);
    int f = find_subspace(i);
    double dmu = -derf(x(i)+mu(f))*(erf(x(i)+mu(f))+1.)/get_V(f);

    for (int j=0;j<l;j++){
        int g = find_subspace(j);
        if(f==g){
            double derf_j = derf(x(j)+mu(f));
            res(j) = RSQRT_TWO*((i==j)+dmu)*derf_j;
        }
        else{ res(j) = 0; }
    }
    return res.asDiagonal();
}
/* 2nd derivative of the square root of kth occupation respect to the ith and jth parameters */
double RDM1::ddsqrt_n(int k, int i,int j){
    int f  = find_subspace(k); 
    int g1 = find_subspace(i);
    int g2 = find_subspace(j);
    if (f==g1 && f==g2){
        double erf_i  = erf(x(i)+mu(f))+1.;
        double erf_j  = erf(x(j)+mu(f))+1.;
        double erf_k  = erf(x(k)+mu(f))+1.;
        double derf_i = derf(x(i)+mu(f));
        double derf_j = derf(x(j)+mu(f));
        double derf_k = derf(x(k)+mu(f));
        double dmu_i = -derf_i*erf_i/get_V(f);
        double dmu_j = -derf_j*erf_j/get_V(f);
        double dderf_i = dderf(x(i)+mu(f));
        double dderf_j = dderf(x(j)+mu(f));
        double dderf_k = dderf(x(k)+mu(f));
        double f_i = derf_i*derf_i + erf_i*dderf_i;
        double f_j = derf_j*derf_j + erf_j*dderf_j;
        double ddmu  = -(get_W(f)*dmu_i*dmu_j+ f_i*dmu_j+ f_j*dmu_i +(i==j)*f_i)/get_V(f);
        return RSQRT_TWO*(dderf_k*((i==k)+dmu_i)*((j==k)+dmu_j)+derf_k*ddmu);
    }
    else{
        return 0;
    }
}
/* 2nd derivative of the square root of the occupations respect to the ith and jth parameters */
MatrixXd RDM1::ddsqrt_n(int i,int j){
    int l = size(); VectorXd res(l);
    for (int k=0;k<l;k++){
        res(k) = ddsqrt_n(k,i,j);
    }
    return res.asDiagonal();
}


/* Derivative of the ith parameter x_i w.r.t the square root of jth occupation */
double RDM1::dx_sqrtn(int i,int j){
    int f = find_subspace(i);
    int g = find_subspace(j);
    if(f==g){
        double sqrtni = sqrtn(i);
        double sqrtnj = sqrtn(j);
        double derfinv_i = derfinv(SQRT_TWO*sqrtni-1.);
        double derfinv_j = derfinv(SQRT_TWO*sqrtnj-1.);
        double dmu = -derf(x(j)+mu(f))*(erf(x(j)+mu(f))+1.)/get_V(f);
        return SQRT_TWO*((i==j)*derfinv_i -dmu*derfinv_j);
    }
    else{
        return 0;
    }
}
/* Derivative of the ith parameters w.r.t the square root of ith occupation */
MatrixXd RDM1::dx_sqrtn(int i){
    int l = size(); VectorXd res(l);
    int f = find_subspace(i);
    double sqrtni = sqrtn(i);
    double derfinv_i = derfinv(SQRT_TWO*sqrtni-1.);
    double dmu = -derf(x(i)+mu(f))*(erf(x(i)+mu(f))+1.)/get_V(f);

    for (int j=0;j<l;j++){
        int g = find_subspace(j);
        if(f==g){
            double sqrtnj = sqrtn(j);
            double derfinv_j = derfinv(SQRT_TWO*sqrtnj-1.);
            res(j) = SQRT_TWO*((i==j)*derfinv_j -dmu*derfinv_i);
        }
        else{ res(j) = 0; }
    }
    return res.asDiagonal();
}

/* Compute the value of mu (shared paramerter of EBI representation) from x */
void RDM1::solve_mu(){
    if(omega.size()==1){
        solve_mu_aux(this); // see EBI_add.cpp
    }
    else{
        cout<<"Error: multi-space functionals not implemented."; // see EBI_add.cpp
    }
    computed_V_ = VectorXi::Constant(omega.size(),false);
    computed_W_ = VectorXi::Constant(omega.size(),false);
}
/* Return a boolean vector of size l initialised at false */
Vector<bool,Eigen::Dynamic> init_V(int l){
    Vector<bool,Eigen::Dynamic> v(l);
    for (int i=0;i<l;i++){
        v(i) = false;
    }
    return v;
}

/*Build the ensenble of subspaces used to compute the energy by some functionals (PNOF7 for now) :
Works such that a subspace is composed of 1 occupied and any number of unoccupied natural orbitals of 
expoentioanlly decreasing occupation.*/
void RDM1::subspace() {
    // Requires 2*Nocc = N_elec (usually the cas but not for [0.66, 0.66, 0.66] for ex.)
    omega.clear();
    int l = size(); int Nocc = 0;
    for (int i = 0; i < l; i++) {
        if (x(i) > -mu(0)) { Nocc++; } 
    }
    int N_omega = l/Nocc; int N_res = l%Nocc; 
    mu = VectorXd::Constant(Nocc,0.); 
    V_ = VectorXd::Constant(Nocc,0); computed_V_ = VectorXi::Constant(Nocc,false);
    W_ = VectorXd::Constant(Nocc,0); computed_W_ = VectorXi::Constant(Nocc,false);
    if(l >= n_elec){
        for (int i = 0; i < N_res; i++) {
            vector<int> v; int p0 = i+l-Nocc;
            v.push_back(p0); double Z = 0;
            for (int j = 1; j < N_omega+1; j++) {
                Z += exp(-j);
            }
            for (int j = 1; j < N_omega; j++) {
                int p = l-j*Nocc -i-1;
                v.push_back(p);
            }
            int p = N_res-i-1;
            v.push_back(p);
            omega.push_back(v); //omega has to be set before the occupations
            set_n(p, (2 - n(p0)) * exp(-N_omega)/Z );
            for (int j = 1; j < N_omega; j++) {
                int p = l-j*Nocc -i-1;
                set_n(p,(2. - n(p0)) * exp(-j)/Z)   ;
            }
        }
        
        for (int i= N_res; i < Nocc;i++){
            vector<int> v; int p0 = i+l-Nocc;
            v.push_back(p0); double Z = 0;
            for (int j = 1; j < N_omega; j++) {
                Z += exp(-j);
            }
            for (int j = 1; j < N_omega; j++) {
                int p = l-j*Nocc -i-1;
                v.push_back(p);
            }
            omega.push_back(v);
            for (int j = 1; j < N_omega; j++) {
                int p = l-j*Nocc -i-1;
                set_n(p, (2. - n(p0)) * exp(-j)/Z );
            }
        }
    }
    else{
        for (int i=0; i<N_res; i++){
            vector<int> v; int p0 = N_res+i; int p = N_res-i-1;
            v.push_back(p0); v.push_back(p); omega.push_back(v);
            set_n(p, 2. - n(p0) );
        }
        for (int i= N_res; i < Nocc; i++){
            vector<int> v; int p0 = i+N_res;
            v.push_back(p0); omega.push_back(v);
            set_n(p0,2.);
        }
        
    }
}

VectorXd zero_eigvls(MatrixXd M,double epsi){
    VectorXd eigs  = M.selfadjointView<Upper>().eigenvalues();
    vector<double>neigvls; 
    for (int i=0;i<eigs.size();i++){
        if(eigs(i)<epsi && eigs(i)>-epsi){
            neigvls.push_back(eigs(i));
        }
    }
    VectorXd res = VectorXd::Map(neigvls.data(),neigvls.size());
    return res;
}

VectorXd negative_eigvls(MatrixXd M,double epsi){
    VectorXd eigs  = M.selfadjointView<Upper>().eigenvalues();
    vector<double>neigvls; 
    for (int i=0;i<eigs.size();i++){
        if(eigs(i)<-epsi){
            neigvls.push_back(eigs(i));
        }
    }
    VectorXd res = VectorXd::Map(neigvls.data(),neigvls.size());
    return res;
}

tuple<VectorXd,MatrixXd,VectorXd,MatrixXd> negative_eigvects(MatrixXd M,double epsi){
    SelfAdjointEigenSolver<MatrixXd> solver(M);
    VectorXd eigvls  = solver.eigenvalues();
    MatrixXd eigvects= solver.eigenvectors();
    vector<double>neigvls; vector<VectorXd>neigvects;
    for (int i=0;i<eigvls.size();i++){
        if(eigvls(i)<-epsi){
            neigvls.push_back(eigvls(i));
            neigvects.push_back(eigvects.col(i));
        }
    }
    VectorXd ret_vls = VectorXd::Map(neigvls.data(),neigvls.size());
    MatrixXd ret_vcts (M.rows(),neigvls.size());
    for (int i=0;i<neigvls.size();i++){
        ret_vcts.col(i) = neigvects[i];
    }
    return make_tuple(ret_vls,ret_vcts,eigvls,eigvects);
}

/*Optimises the occupations (n) and NOs (no) of the 1RDM with respect to the energy minimisation
\param args func: the functional to use
            disp: if >1 displais details about the computation
            epsi: relative precision required for the optimisation
            epsi_n: relative precision required for the optimisation of the occupations (default sqrt(epsi))
            epsi_no: relative precision required for the optimisation of the NOs (default sqrt(epsi))
            maxiter: maximum number of iterations for one optimisation of the NOs/occupations, and maximum number of 
                     calls to those optimisations
the occupations and NOs are optimised in-place
*/
void RDM1::opti(Functional* func, string hess_approx, string file, int disp, double epsi, double epsi_nno, int maxiter){
    ofstream ofile; string path ("/Users/nicolascartier/Documents/Programmes/Tests/Conv data/"); path.append(file); path.append(".txt");
    if(file!=""){ ofile.open(path);}
    else{ofile.setstate(ios_base::badbit);}
    cout<<setprecision(-log(epsi)+1); ofile<<setprecision(-log(epsi)+1);
    auto t_init = chrono::high_resolution_clock::now();
    int k = 0; int l = size(); int nit= 0; int ll = l*(l-1)/2;
    double E = func->E(this); double E_bis = DBL_MAX; double grad = DBL_MAX;
    
    bool detailed_disp;
    if (disp>2){detailed_disp = true;}
    else {detailed_disp = false;}
    double epsi_nno_bis = epsi_nno; 
    MatrixXd hess_nno;

    while( ( abs((E_bis-E)/E)>epsi ) && k<maxiter){ 
        k++; E_bis = E; epsi_nno_bis = epsi_nno;
        auto t0 = chrono::high_resolution_clock::now();
        auto res = opti_nno(this, func, hess_approx, &ofile, epsi_nno, detailed_disp, maxiter);
        int nit_i = get<1>(res); E = get<0>(res); 
        
        auto t1 = chrono::high_resolution_clock::now();
        ofile<<"---------"<<endl;
        
        nit += nit_i;
        if (disp>1){
            grad = (func->grad_E(this,false,false)).norm();
            cout<<"Iteration "<<k <<" E="<<E<<" |grad_E|="<<grad<<endl;
            cout<<"Opti time: "; print_t(t1,t0); cout<<" and # of iter "<< nit_i<<endl;
        }
    }
    if (k==maxiter){
        cout<<"Computation did not converge"<<endl;
    }
    auto t_fin = chrono::high_resolution_clock::now();
    if (disp>0){ 
        cout<<endl;
        cout<<"Computational time "; print_t(t_fin,t_init); cout<<" total # of iter "<<nit<<endl;
    }
}


double norm2(VectorXd* x){
    int l = x->size();
    double res = 0;
    for (int i =0; i<l;i++){
        res += pow(x->coeff(i),2);
    }
    return sqrt(res);
}

double norm1(VectorXd* x){
    int l = x->size();
    double res = 0;
    for (int i =0;i<l;i++){
        res += abs(x->coeff(i));
    }
    return res;
}

MatrixXd exp_unit(VectorXd* l_theta){
    int l = ((sqrt(8*l_theta->size())+1)+1)/2; int index = 0;
    MatrixXd res (l,l); 
    for (int i=0;i<l;i++){
        for (int j=0;j<=i;j++){
            if (i==j){res(i,i) = 0;}
            else{
                res(i,j) =  l_theta->coeff(index);
                res(j,i) = -l_theta->coeff(index);
                index++;
            }
        }
    }
    return res.exp();
}

fides::cost_fun_ret_t f_dir(DynamicVector<double> X0, DynamicVector<double> X, void* f_data, bool accepted){
    int ll = X.size(); int l = (sqrt(8*ll+1)-1)/2; 
    VectorXd x0 = VectorXd::Map(X0.data(),ll); 
    VectorXd x  = VectorXd::Map(X.data(),ll); 
    data_struct *data = (data_struct*) f_data;
    //Update gamma
    MatrixXd NO = data->gamma->no;
    data->gamma->x(x.segment(0,l));
    data->gamma->solve_mu();
    VectorXd step = x.segment(l,ll-l) - x0.segment(l,ll-l);
    data->gamma->set_no(NO*exp_unit(&step));
    if (accepted){
        //on a successfull step we have to update the reference points
        data->x1 = x; data->grad1 = data->func->grad_E(data->gamma); data->E1 = data->func->E(data->gamma);
        return make_tuple(0,DynamicVector<double>(0),DynamicMatrix<double>(0,0));
    }
    data->x2 = x;
    //Compute E grad hessian
    double E = data->func->E(data->gamma) ; data->E2 = E;
    VectorXd grad =  data->func->grad_E(data->gamma);
    data->grad2 = grad;
   
    if(data->hess_approx == "exa" ) { data->hess_ = data->func->hess_E_exa(data->gamma); }
    else if (data->niter%data->interval==0 && data->niter !=0){
        MatrixXd hess_exa = data->func->hess_E_exa(data->gamma);
        data->hess_ = hess_exa;
        data->hess_exp_ = hess_exa;
    }
    else {
        data->hess_cheap_ = data->func->hess_E(data->gamma);
        //Computes a BFGS approximation of the expensive part of the Hessian
        if(data->hess_approx == "SR1"){ SR1(f_data); }
        else if(data->hess_approx == "BFGS"){ BFGS(f_data); }
        else if(data->hess_approx == "DFP"){ DFP(f_data); }
        else if(data->hess_approx == "LBFGS"){ LBFGS(f_data);}
        else if(data->hess_approx == "Broyden"){ Broyden(f_data); }
        else if(data->hess_approx == "Zero"){ ZERO(f_data); }
        else{ throw::invalid_argument("Unknown Hessian approximation. "); }   
        data->hess_ = data->hess_cheap_; 
        if(!data->do_NONO){ //only use the occupation block of the expensive part of the Hessian
            data->hess_exp_.block(0,l,l,ll-l) = MatrixXd::Zero(l,ll-l);
            data->hess_exp_.block(l,0,ll-l,l) = MatrixXd::Zero(ll-l,l);
            data->hess_exp_.block(l,l,ll-l,ll-l) = MatrixXd::Zero(ll-l,ll-l);
        }
        if (data->mixed){ 
            double snNorm  = (x-x0).segment(0,l).lpNorm<Infinity>(); if(data->niter==0){snNorm=1;}
            data->hess_ += (1.-Activation_log(snNorm,EPSILON,BETA))*data->hess_exp_;
        }
        else {
            data->hess_ += data->hess_exp_;
        }
        
    }
    if(!data->do_nNO){
        data->hess_.block(0,l,l,ll-l) = MatrixXd::Zero(l,ll-l); //remove coupling
        data->hess_.block(l,0,ll-l,l) = MatrixXd::Zero(ll-l,l);
    }
    if(data->r_diag){
        data->hess_ = data->hess_.diagonal().asDiagonal();
    }

    DynamicVector<double> grad_ret (ll); blaze::SymmetricMatrix<DynamicMatrix<double>> hess_ret (ll);
    for (int i=0;i<ll;i++){
        grad_ret[i] = grad(i);
        for (int j =0;j<=i;j++){
            hess_ret(i,j) = data->hess_(i,j); hess_ret(j,i) = data->hess_(i,j); 
        }
    }

    //Returns/writes results
    /*MatrixXd hess_exa = data->func->hess_E_exa(data->gamma);
    MatrixXd H_error = (hess_exa - data->hess_).array()/hess_exa.array(); 
    VectorXd neigvals; MatrixXd neigvects; VectorXd eigvals; MatrixXd eigvects;
    tie(neigvals,neigvects,eigvals,eigvects) = negative_eigvects(hess_exa);
    VectorXd nvct_s_ovlp (neigvects.cols()); VectorXd vct_s_ovlp (eigvects.cols());
    for(int i=0;i<neigvects.cols();i++){
        nvct_s_ovlp(i) = neigvects.col(i).dot(x-x0);
         vct_s_ovlp(i) =  eigvects.col(i).dot(x-x0);

    }
    MatrixXd Hdiag = MatrixXd::Zero(ll,ll); for (int i=0;i<ll;i++){ Hdiag(i,i) = data->hess_(i,i);}
    VectorXd s = (x-x0);*/

    *data->ofile<<"E="<<E<<" iter="<<data->niter<<" |grad|="<<data->grad2.norm()<<" |step|="<<(x-x0).norm()\
    /*<<" sTg_l2="<<s.transpose()*data->grad1<<" sTg_H="<<s.transpose()*data->hess_*data->grad1<<" sTg_Hd="<<s.transpose()*Hdiag*data->grad1\
    <<" #0_eigval_tot="<<zero_eigvls(data->hess_).size()<<" #0_eigval_n="<<zero_eigvls(data->hess_.block(0,0,l,l)).size()\
    <<" #0_eigval_no="<<zero_eigvls(data->hess_.block(l,l,ll-l,ll-l)).size()\ 
    <<" |eigvls|="<<hess_exa.selfadjointView<Upper>().eigenvalues().norm() \
    <<" |H_er|="<<H_error.lpNorm<1>()/(ll*ll)<<" |H_er_nn|="<<H_error.block(0,0,l,l).lpNorm<1>()/(l*l)\
    <<" |H_er_NONO|="<<H_error.block(l,l,ll-l,ll-l).lpNorm<1>()/( (ll-l)*(ll-l) )<<" |H_er_nNO|="<<H_error.block(0,l,l,ll-l).lpNorm<1>()/(l*(ll-l))\
    <<" #eigvl_tot="<<neigvals.size()<<" #eigvl_n="<<negative_eigvls(hess_exa.block(0,0,l,l)).size()\
    <<" #eigvl_no="<<negative_eigvls(hess_exa.block(l,l,ll-l,ll-l)).size()<<" mean_eigvl="<<eigvals.cwiseAbs().mean()<<" max_neigvl="<<neigvals.lpNorm<Infinity>()\
    <<" mean_ovlp="<<vct_s_ovlp.cwiseAbs().mean()<<" max_novlp="<<nvct_s_ovlp.lpNorm<Infinity>()\ */
    <<endl;
    
    data->gamma->set_no(NO);
    data->niter++;
    
    return make_tuple(E,grad_ret,hess_ret);
    };

//Objective function for the occs and NOs optimistion called by the fides minimizer
fides::cost_fun_ret_t f_aux(DynamicVector<double> X0, DynamicVector<double> X, void* f_data, bool accepted){
    int ll = X.size(); int l = (sqrt(8*ll+1)-1)/2; int l2 = l*l;
    VectorXd x0 = VectorXd::Map(X0.data(),ll); 
    VectorXd x  = VectorXd::Map(X.data(),ll); 
    data_struct *data = (data_struct*) f_data;
   
    //Update gamma
    MatrixXd NO = data->gamma->no;
    data->gamma->x(x.segment(0,l));
    data->gamma->solve_mu();
    VectorXd step = x.segment(l,ll-l) - x0.segment(l,ll-l);
    data->gamma->set_no(NO*exp_unit(&step));
    if (accepted){
        //on a successfull step we have to update the reference points
        if(data->hess_approx == "dBFGS"){
            //get x and grad in nu space
            MatrixXd J = data->func->Jac(data->gamma); 
            MatrixXd Jinv = data->func->InvJac(data->gamma).transpose(); 
            data->x1 = J*x; data->grad1 = Jinv*data->func->grad_E(data->gamma); data->E1 = data->func->E(data->gamma);
        }
        else{
            data->x1 = x; data->grad1 = data->func->grad_E(data->gamma); data->E1 = data->func->E(data->gamma);
        }
        return make_tuple(0,DynamicVector<double>(0),DynamicMatrix<double>(0,0));
    }
    MatrixXd J;  MatrixXd Jinv;

    if(data->hess_approx == "dBFGS"){
        J = data->func->Jac(data->gamma);
        Jinv = data->func->InvJac(data->gamma).transpose(); 
        data->x2 = J*x;
    }
    else{data->x2 = x;}
    
    
    //Compute E grad hessian
    MatrixXd hess_approx;
    double E = data->func->E(data->gamma) ; data->E2 = E;
    VectorXd grad = data->func->grad_E(data->gamma);
    if(data->hess_approx == "dBFGS"){data->grad2 = Jinv*grad;}
    else{data->grad2 = grad;}
    
    
    if(data->hess_approx == "exa") { data->hess_ = data->func->hess_E_exa(data->gamma); }
    else if( data->niter%data->interval==0 && data->niter !=0){ 
        MatrixXd hess_exa = data->func->hess_E_exa(data->gamma);
        data->hess_ = hess_exa;
        MatrixXd hess_approx = hess_exa - data->func->hess_E(data->gamma);
        data->hess_exp_ = data->func->x_space_hess(data->gamma,&hess_approx);
    } 
    else {
        data->hess_cheap_ = data->func->hess_E(data->gamma);
        //Computes a BFGS approximation of the expensive part of the Hessian
        if(data->hess_approx == "SR1"){ SR1_aux(f_data); }
        else if(data->hess_approx == "BFGS"){ BFGS_aux(f_data); }
        else if(data->hess_approx == "tBFGS"){ tBFGS_aux(f_data); }
        else if(data->hess_approx == "sBFGS"){ sBFGS_aux(f_data); }
        else if(data->hess_approx == "dBFGS"){ dBFGS_aux(f_data); }
        else if(data->hess_approx == "LBFGS"){ LBFGS_aux(f_data); }
        else if(data->hess_approx == "Zero"){ ZERO_aux(f_data); }
        else{ throw::invalid_argument("Unknown Hessian approximation. "); }
        data->hess_ = data->hess_cheap_;  
        if(!data->do_NONO){ //only use the occupation block of the expensive part of the Hessian
            data->hess_exp_.block(0,l,l,ll-l) = MatrixXd::Zero(l,ll-l);
            data->hess_exp_.block(l,0,ll-l,l) = MatrixXd::Zero(ll-l,l);
            data->hess_exp_.block(l,l,ll-l,ll-l) = MatrixXd::Zero(ll-l,ll-l);
        }
        if (data->mixed){ 
            double snNorm  = (x-x0).segment(0,l)                                                                                                                                .lpNorm<Infinity>(); if(data->niter==0){snNorm=1;}
            data->hess_ += (1.-Activation_log(snNorm,EPSILON,BETA))*data->hess_exp_;
        }
        else {
            data->hess_ += data->hess_exp_;
        }
        data->hess_exp_ = MatrixXd::Zero(ll,ll); //have to rebuild hess_exp_, cannot accumulate in aux
    }
    if(!data->do_nNO){
        data->hess_.block(0,l,l,ll-l) = MatrixXd::Zero(l,ll-l); //remove coupling
        data->hess_.block(l,0,ll-l,l) = MatrixXd::Zero(ll-l,l);
    }
    if(data->r_diag){
        data->hess_ = data->hess_.diagonal().asDiagonal();
    }
    
    DynamicVector<double> grad_ret (ll); blaze::SymmetricMatrix<DynamicMatrix<double>> hess_ret (ll); 
    for (int i=0;i<ll;i++){
        grad_ret[i] = grad(i);
        for (int j =0;j<=i;j++){
            hess_ret(i,j) = data->hess_(i,j); hess_ret(j,i) = data->hess_(j,i); 
        }
    }

    //Returns/writes results
    /*MatrixXd hess_exa = data->func->hess_E_exa(data->gamma);
    MatrixXd H_error = hess_exa - data->hess_; double Hnorm = hess_exa.norm();
    VectorXd neigvals; MatrixXd neigvects; VectorXd eigvals; MatrixXd eigvects;
    tie(neigvals,neigvects,eigvals,eigvects) = negative_eigvects(hess_exa);
    VectorXd nvct_s_ovlp (neigvects.cols()); VectorXd vct_s_ovlp (eigvects.cols());
    for(int i=0;i<neigvects.cols();i++){
        nvct_s_ovlp(i) = neigvects.col(i).dot(x-x0);
         vct_s_ovlp(i) =  eigvects.col(i).dot(x-x0);

    }
    double snNorm = (x-x0).segment(0,l).lpNorm<Infinity>();
    MatrixXd Hdiag = MatrixXd::Zero(ll,ll); for (int i=0;i<ll;i++){ Hdiag(i,i) = data->hess_(i,i);}
    VectorXd s = (x-x0);*/
    
    *data->ofile<<"E="<<E<<" iter="<<data->niter<<" |grad|="<<grad.norm()<<" |step|="<<(x-x0).norm()\
    /*<<" sTg_l2="<<s.transpose()*data->grad1<<" sTg_H="<<s.transpose()*data->hess_*data->grad1<<" sTg_Hd="<<s.transpose()*Hdiag*grad\
    <<" |s_n|_oo="<<snNorm<<" |s_no|="<<s.segment(l,ll-l).lpNorm<Infinity>()<<" A="<<(1.-Activation_log(snNorm,EPSILON,BETA))\
    <<" #eigvl_tot="<<neigvals.size()<<" #eigvl_n="<<negative_eigvls(hess_exa.block(0,0,l,l)).size()\
    <<" #eigvl_no="<<negative_eigvls(hess_exa.block(l,l,ll-l,ll-l)).size()<<" mean_eigvl="<<eigvals.cwiseAbs().mean()<<" max_neigvl="<<neigvals.lpNorm<Infinity>()\
    <<" |eigvls|="<<hess_exa.selfadjointView<Upper>().eigenvalues().norm() \
    <<" |H_er|="<<H_error.norm()<<" |H_er_nn|="<<H_error.block(0,0,l,l).norm()\
    <<" |H_er_NONO|="<<H_error.block(l,l,ll-l,ll-l).norm()<<" |H_er_nNO|="<<sqrt(2.)*H_error.block(0,l,l,ll-l).norm()\ 
    <<" mean_ovlp="<<vct_s_ovlp.cwiseAbs().mean()<<" max_novlp="<<nvct_s_ovlp.lpNorm<Infinity>()*/
    <<endl;

    data->gamma->set_no(NO);
    data->niter++;

    return make_tuple(E,grad_ret,hess_ret);
    };


/* Optimises the NOs of the 1RDM with respect to the inimisation of the energy
\param args gamma: 1RDM
            func: functional
            epsilon: required precision
            disp: get detail on the optimisation /TO DO/
            maxiter: maximum number of iterations
the occupations are NOs in-place
\param results  corresponding energy, number of iterations
*/
tuple<double,int> opti_nno(RDM1* gamma, Functional* func, string hess_approx, ofstream* ofile, 
                            double epsilon, bool disp, int maxiter){
    int l = gamma->size(); int ll = l*(l+1)/2; int l2 = l*l; 
    DynamicVector<double>  x(ll); DynamicVector<double> lb (ll); DynamicVector<double> ub (ll);
    
    for (int i = 0; i < l; i++) {
        x[i] = gamma->x(i); lb[i] = -4.; ub[i] = 4.;
    }
    for (int i=l; i<ll;i++){
        x[i] = 0.; lb[i] = -2.*M_PI; ub[i] = 2.*M_PI;
    }
    
    data_struct f_data;
    if (hess_approx.find("exa") != string::npos && hess_approx.find("start") == string::npos){
        //Use the exact Hessian
        f_data.hess_approx = "exa";
    }
    //Updates for H_exp : the expensive part of the Hessian i.e O(N^5) term in the NO-NO block.
    else if (hess_approx.find("SR1") != string::npos){
        //Use SR1 approximation of the Hessian
        f_data.hess_approx = "SR1";
    }
    else if (hess_approx.find("LBFGS") != string::npos){
        //Use limited memory BFGS approximation of the Hessian
        f_data.hess_approx = "LBFGS";
        string::size_type p = hess_approx.find("LBFGS");
        if(p != string::npos){
            int M;
            string::size_type p2 = hess_approx.find('_',p+7);
            if (p2 == string::npos){ M = 10; }
            else{
                string m = hess_approx.substr(p+6,p2-p-1); M = stoi(m);
            }  
            f_data.memory = M; 
        }
    }
    else if (hess_approx.find("tBFGS") != string::npos){
        //Use ~BFGS approximation of the Hessian
        f_data.hess_approx = "tBFGS";
    } 
    else if (hess_approx.find("sBFGS") != string::npos){
        //Use ~BFGS approximation of the Hessian
        f_data.hess_approx = "sBFGS";
    } 
    else if (hess_approx.find("dBFGS") != string::npos){
        //Use ~BFGS approximation of the Hessian
        f_data.hess_approx = "dBFGS";

    }
    else if (hess_approx.find("BFGS") != string::npos){//also called by bBFGS in nu-space
        //Use BFGS approximation of the Hessian
        f_data.hess_approx = "BFGS";
    } 
    else if (hess_approx.find("DFP") != string::npos){
        //Use DFP approximation of the Hessian
        f_data.hess_approx = "DFP";
    }
    else if (hess_approx.find("Broyden") != string::npos){
        //Use a element of the Broyden class approximation of the Hessian
        f_data.hess_approx = "Broyden";
    }
    else if (hess_approx.find("ZERO") != string::npos || hess_approx.find("Zero") != string::npos){
        //Set the expensive part of the Hessian to 0
        f_data.hess_approx = "Zero";
    }
    else{
        throw::invalid_argument("Unknown Hessian approximation. ");
    }

    fides::cost_function_t f_;    
    if(hess_approx.find("_aux") != string::npos){
        //Use an approximation of the auxiliary Hessian
        f_ = f_aux;
        if(hess_approx.find("_exastart") != string::npos){
            //Use the exact Hessian at 1st iteration
            MatrixXd hess_exa = func->hess_E(gamma); 
            f_data.hess_ = hess_exa; f_data.do_1st_iter = false;
            f_data.hess_exp_ = MatrixXd::Identity(ll,ll);
            f_data.hess_exp_nu = func->x_space_hess(gamma,&hess_exa);
            
        }
        else{
            f_data.hess_ = MatrixXd::Identity(ll,ll); 
            f_data.hess_exp_ = MatrixXd::Identity(ll,ll);
            f_data.hess_exp_nu = MatrixXd::Identity(l2+l,l2+l);
        }
        string::size_type p = hess_approx.find("_aux");
        if(p != string::npos){
            int M;
            string::size_type p2 = hess_approx.find('_',p+6);
            if (p2 == string::npos){ M = 500; }
            else{
                string m = hess_approx.substr(p+5,p2-p-1); M = stoi(m);
            }  
            f_data.memory = M; 
        }
    }
    else{
        f_ = f_dir; 
        if(hess_approx.find("_exastart") != string::npos){
            //Use the exact Hessian at 1st iteration
            f_data.hess_ = func->hess_E_exa(gamma); f_data.do_1st_iter = false;
            f_data.hess_exp_ = (f_data.hess_ - func->hess_E(gamma));
        } else{ 
            f_data.hess_ = MatrixXd::Identity(ll,ll); 
            f_data.hess_exp_ = MatrixXd::Identity(ll,ll); 
        }
    }
    f_data.gamma = gamma; f_data.func = func; f_data.disp = disp;
    if (f_data.hess_approx == "dBFGS"){
        MatrixXd Jinv = func->InvJac(gamma).transpose(); 
        f_data.x1 = VectorXd::Zero(l+l*l); f_data.grad1 = Jinv*func->grad_E(gamma); 
    }
    else{f_data.x1 = VectorXd::Zero(ll); f_data.grad1 = func->grad_E(gamma); }
    f_data.E1 = func->E(gamma);
    f_data.niter=0; f_data.ofile = ofile;  
    f_data.do_1st_iter = true; f_data.interval = INT_MAX; 
    f_data.do_nn = true; f_data.do_NONO = true; f_data.do_nNO = true; 
    
    fides::Options options; options.maxiter = maxiter; options.refine_stepback = false; 
    options.fatol = 1e-17; options.frtol = 1e-17; options.xtol = epsilon; options.gatol = 100*epsilon; 
    //options.mu = 1e-2; options.eta = 0.4;
    //options.delta_init = 1.; options.gamma1 = 1.; options.gamma2 = 1.; //test without trust-region

    if ((hess_approx.find("no_coupling") != string::npos) || (hess_approx.find("no_nNO") != string::npos)){
        //don't use the coupling of the total Hessian
        f_data.do_nNO = false; 
    }
    if (hess_approx.find("no_NONO") != string::npos){
        //don't use the orbital block of the expensive approximate Hessian
        f_data.do_NONO = false; 
    }
    if (hess_approx.find("diag") != string::npos){
        //restric to diagonal of the Hessian
        f_data.r_diag=true;
    }
    else { f_data.r_diag=false; }

    string::size_type p = hess_approx.find("hybrid");
    if(p != string::npos){
        //Use exact Hessian every M iterations
        string::size_type p2 = hess_approx.find('_',p+8);
        string m = hess_approx.substr(p+7,p2-p-1); int M;
        if (m.empty()){ M = 10; }
        else{ M = stoi(m); }
        f_data.interval = M; 
    }
    if(hess_approx.find("mixed") != string::npos || hess_approx.find("Mixed") != string::npos ){
        //Start with the Zero approximation of the expensive Hessian and then used the approximation specified above
        f_data.mixed = true; }
    else{ f_data.mixed = false;}

    
    fides::Optimizer opti (f_,lb,ub,options,nullptr,&f_data);
    opti.minimize(x);
    return make_tuple(opti.fval_,f_data.niter);
}

