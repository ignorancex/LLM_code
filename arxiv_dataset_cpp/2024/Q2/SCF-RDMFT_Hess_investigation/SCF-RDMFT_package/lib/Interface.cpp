#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
//Eigen3 & NLopt
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <nlopt.hpp>
//Types
#include <float.h>
#include <vector>
#include <string>
#include <tuple>
//Print 
#include <iostream>
#include <fstream>
#include <iomanip>      
#include <chrono>

using namespace std;
using namespace Eigen;

#include "classes/1RDM_class.hpp"
#include "classes/Functional_class.hpp"
#include "classes/Matrix_Tensor_converter.cpp"
#include "numerical_deriv/numerical_deriv.hpp"
#include "classes/EBI_add.hpp"
#include "tools.hpp"
#include "Interface.hpp"

#include "Functionals/HF.hpp"
#include "Functionals/Muller.hpp"
#include "Functionals/Power.hpp"

//Dictionary of functionals 
map<string, Functional> Funcs = {{"E1",E1_func},{"Hartree",Hartree_func},{"HF",HF_func},{"Muller",Muller_func},{"Power",Power_func}};

// Wrapper to python code
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;
PYBIND11_MODULE(Compute_1RDM, m){
    m.def("Optimize_1RDM", &Optimize_1RDM, " Given occupations, natural orbital matrix, number of electrons, \
    nuclear energy, overlap matrix, tensors of integrals (one and two bodies) and a functional returns the \
    optimized 1RDM (ie corresponding occupation and natural orbital matrix) and prints its energy");
    m.def("E",&E, "Given occupations, natural orbital matrix, number of electrons, \
    nuclear energy, overlap matrix, tensors of integrals (one and two bodies) and a \
    functional returns the corresponding energy");
    m.def("test",&test);
    m.doc() = "C++ extension to compute 1RDMs from PySCF";
}

//Used to test the library from Python
void test(string func, VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc,
                                    MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int){
                                
    int l = overlap.rows(); int ll = pow(l,2); int l2 = l*(l+1)/2; double epsi = 1e-6;
    // (10)(32) permutation already presente, due to type conversion
    Tensor<double,4> T(l,l,l,l); T = TensorCast(elec2int,l,l,l,l);  Eigen::array<int,4> index({ 1,3,0,2 });
    Tensor<double,4> T2 = T.shuffle(index);
    MatrixXd elec2int_x(ll, ll); elec2int_x = MatrixCast(T2, ll, ll);
    RDM1 gamma = RDM1(occ, orbital_mat, ne, Enuc, overlap, elec1int, elec2int, elec2int_x);
    VectorXd x = VectorXd::Random(l2-l);
    gamma.set_no(gamma.no*exp_unit(&x));
    cout<<setprecision(6)<<endl;
    MatrixXd g (l,l); g=gamma.mat();
    
    auto functional = Funcs.find(func); 

    /*//code to test expression of the gradient and Hessian
    VectorXd grad_num = grad_func(&functional->second,&gamma);
    VectorXd grad_test= functional->second.grad_E(&gamma);

    MatrixXd hess_num = hess_func(&functional->second,&gamma);
    MatrixXd hess_test= functional->second.hess_E_exa(&gamma);

    cout<<"grad num ="<<grad_num.transpose()<<endl;
    cout<<"grad test="<<grad_test.transpose()<<endl;

    if(l<7){
        cout<<"occ block:"<<endl;
        cout<<hess_num.block(0,0,l,l)<<endl<<endl;
        cout<<hess_test.block(0,0,l,l)<<endl<<endl;
        cout<<"coupling block:"<<endl;
        cout<<hess_num.block(0,l,l,l2-l).transpose()<<endl<<endl;
        cout<<hess_test.block(0,l,l,l2-l).transpose()<<endl<<endl;
        cout<<"NO block:"<<endl;
        cout<<hess_num.block(l,l,l2-l,l2-l)<<endl<<endl;
        cout<<hess_test.block(l,l,l2-l,l2-l)<<endl<<endl;
    }
    */
    
    auto t0 = chrono::high_resolution_clock::now(); int nrec = 50;
    
    for (int i = 0;i<nrec;i++){
        functional->second.hess_E_exa(&gamma);
    }
    auto t1 = chrono::high_resolution_clock::now();
    cout<<"H exa computation=";print_t(t1,t0,nrec); cout<<endl;
    auto t2 = chrono::high_resolution_clock::now();
    for (int i = 0;i<nrec;i++){
        functional->second.hess_E(&gamma);
    }
    auto t3 = chrono::high_resolution_clock::now();
    cout<<"H cheap computation=";print_t(t3,t2,nrec); cout<<endl;
}

/* 
Optimise the 1RDM defined by results from Interface.py to obtain the ground state energy
\param arg    func: functional to use (a new instance of the class can be defined to obtain a new functional)
              disp: if <1 prints details of the computation 
              epsi: relative error required 
              Maxiter: maximum number of iteration for one optimisation of the occupations/NOs
              other arguments are provided by the Interface.py and used to build the 1RDM
              hess : the Hessian approximation to use: analitycal, BFGS or SR1.
              file : file in which the outputs per iteration will be saved
\param result the vector of the of the occupations (n) and matrix of the NOs (no) 
              (the 1RDM = no * Diagonal(n) no.T)
              also prints the corresponding ground state energy.
*/
tuple<VectorXd, MatrixXd> Optimize_1RDM(string func, VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc,
    MatrixXd overlap, MatrixXd elec1int, MatrixXd elec2int, int disp, double epsi, int Maxiter, string hess, string file) {
    int l = overlap.rows(); int ll = pow(l, 2);
    Tensor<double, 4> T(l, l, l, l); T = TensorCast(elec2int, l, l, l, l);  Eigen::array<int, 4> index({ 1,3,0,2 });
    MatrixXd elec2int_x(ll, ll); elec2int_x = MatrixCast(T.shuffle(index), ll, ll);
    RDM1 gamma = RDM1(occ, orbital_mat, ne, Enuc, overlap, elec1int, elec2int, elec2int_x); double E; 
    auto functional = Funcs.find(func); double epsi_nno = epsi; 
    if (functional == Funcs.end()) {
        gamma.opti(&HF_func, hess, file, disp, epsi, epsi_nno, Maxiter);
        E = HF_func.E(&gamma);  
    }
    else{
        //Remark: this part is aimed for seniority 0 based functionals and will not be called by functionals 
        //        present in this implementation
        if(functional->second.needs_subspace()){
            gamma.subspace();
            gamma.solve_mu();
        }
        gamma.opti(&functional->second, hess, file, disp, epsi, epsi_nno, Maxiter);
        E = functional->second.E(&gamma);  
    }
    cout<<"E="<<E<<endl;
    return make_tuple(gamma.n(), gamma.no); 
}

/*
\param arg    func: functional to use (a new instance of the class can be defined to obtain a new functional)
              other arguments are provided by the Python part of Interface and used to build the 1RDM
\param result the energy of the corresponding 1RDM for the funtionla func.
*/
double E (string func, VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc,
                                    MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int){
    int l = overlap.rows(); int ll = pow(l, 2);
    
    
    Tensor<double, 4> T(l, l, l, l); T = TensorCast(elec2int, l, l, l, l);  Eigen::array<int, 4> index({ 1,3,0,2 });
    MatrixXd elec2int_x(ll, ll); elec2int_x = MatrixCast(T.shuffle(index), ll, ll);
    RDM1 gamma = RDM1(occ, orbital_mat, ne, Enuc, overlap, elec1int, elec2int, elec2int_x); double E;
    auto functional = Funcs.find(func);
    if (functional == Funcs.end()) {
        E = HF_func.E(&gamma);  
    }
    else{
        if(functional->second.needs_subspace()){
            gamma.subspace();
            gamma.solve_mu();
        }
        E = functional->second.E(&gamma);  
    }
    return E; 
}
