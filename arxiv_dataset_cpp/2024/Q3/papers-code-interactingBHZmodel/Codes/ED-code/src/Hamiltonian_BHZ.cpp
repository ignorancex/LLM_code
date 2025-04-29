#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "Parameters_BHZ.hpp"
#include "Connection_BHZ.hpp"
#include "Hamiltonian_BHZ.hpp"

using namespace std;
using namespace Eigen;

void Hamiltonian_BHZ::Initialize(){
    size_=Parameters_BHZ_.Ham_Size;
    Evals_=VectorXd::Zero(size_);
    Evecs_=MatrixXcd::Zero(size_,size_);
}


void Hamiltonian_BHZ::Diagonalizer(){

    //Diagonalize the Hamiltonian:->
    SelfAdjointEigenSolver<MatrixXcd> solver;
    solver.compute(Connection_BHZ_.C_mat);

    if(solver.info()!=Success)
        abort();

    Evals_ = solver.eigenvalues();
    Evecs_ = solver.eigenvectors();

    string Evals_out="Eigenvalues.txt";
    ofstream Evals_file_out(Evals_out.c_str());

    for(int i=0;i<Evals_.size();i++){
        Evals_file_out<<i<<"    "<<Evals_(i)<<endl;
    }
        
}



double Hamiltonian_BHZ::ChemicalPotential(double particles_){
    double mu_temp, eps_, dmu_by_dN, N_temp, dmu_by_dN_min, Ne_;
    eps_=1e-2;
    mu_temp=Evals_(0);
    N_temp=100000;
    Ne_=1.0*particles_;
    
    int iters=0;

    dmu_by_dN = 0.01*( Evals_(size_-1) - Evals_(0) )*( 1.0/(1.0*size_) );
    dmu_by_dN_min = 0.0001*( Evals_(size_-1) - Evals_(0) )*( 1.0/(1.0*size_) );

    while( abs(N_temp - Ne_) > eps_){
        N_temp=0;
        for(int i=0;i<size_;i++){
            N_temp += FermiFunction(Evals_(i),mu_temp);
        }

        mu_temp = mu_temp + dmu_by_dN*(Ne_-N_temp);
        iters++;

        if(iters%1000==0){
            dmu_by_dN = (1000.0/(10.0*iters))*dmu_by_dN;
            dmu_by_dN = max(dmu_by_dN, dmu_by_dN_min);
        }
    }

    cout<<"Calculated mu = "<<mu_temp<<endl;
    cout<<"Calculated # of particles = "<<N_temp<<endl;

    return mu_temp;
}



double Hamiltonian_BHZ::FermiFunction(double en_, double mu_){
    double ffn, temp;
    temp=Parameters_BHZ_.Temp;

    ffn = ( (1.0)/(1.0 + exp( (en_ - mu_)/temp )) );

    return ffn;
}