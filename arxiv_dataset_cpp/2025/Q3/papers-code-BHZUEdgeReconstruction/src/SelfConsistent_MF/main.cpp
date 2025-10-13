#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <random>
//extern "C" {
//#include <complex.h>
//}
#include <complex>
#include <cmath>
#include <cassert>
#include <utility>
//#include <cblas.h>
//#include <lapacke.h>
#include "tensors.h"
#include "Parameters_BHZ.h"
#include "MFParameters_BHZ.h"
#include "Hamiltonian_BHZ.h"
#include "Observables_BHZ.h"

using namespace std;

int main(int argc, char *argv[]){

//    if (argc<2){ throw std::invalid_argument("USE:: executable inputfile"); }
    string inputfile = argv[1];

    //Calling out the classes and their constructors:
    Parameters_BHZ Parameters_BHZ_;
    Parameters_BHZ_.Initialize(inputfile);

    mt19937_64 Generator_(Parameters_BHZ_.Seed_);

    MFParam_BHZ MFParam_BHZ_(Parameters_BHZ_, Generator_);
    Hamiltonian_BHZ Hamiltonian_BHZ_(Parameters_BHZ_);
    Observables_BHZ Observables_BHZ_(Parameters_BHZ_, Hamiltonian_BHZ_);

    //----------------------------------------------//
    //Initializing the order parameters:----->
    Mat_1_Complex_doub OPs_old;
    OPs_old.resize(Hamiltonian_BHZ_.H_size);

    OPs_old = MFParam_BHZ_.initialOPs_;
    //-----------------------------------------------//
    Mat_1_doub ionic_old = MFParam_BHZ_.initial_ionic;
    
    ofstream file_out_progress("output_selfconsistency.txt");
    if(!file_out_progress){
        cerr << "Failed to open output file." << endl;
        return 1;
    }

    //Self-consistency loop:---->
    cout << "target error = " << Parameters_BHZ_.Conv_err << endl;
    cout << "Max iterations = " << Parameters_BHZ_.Max_iters << endl;

    int iter = 0;
    double OP_error;
    OP_error= 10.0;

    double mu_old, mu_guess;

    while ((OP_error > Parameters_BHZ_.Conv_err) && (iter <= Parameters_BHZ_.Max_iters)){      

        if(Parameters_BHZ_.canonical_){
            OP_error=0.0;

            Hamiltonian_BHZ_.OPs_ = OPs_old;
            
            //Building connection matrix
            Hamiltonian_BHZ_.connectionMatrix();

            //Diagonalizing the Hamiltonian with new OP's
            Hamiltonian_BHZ_.Diagonalizer();

            //Calculating the chemical potential
            mu_guess = 0.5*(Hamiltonian_BHZ_.evals_[Parameters_BHZ_.total_particles_-1] + Hamiltonian_BHZ_.evals_[Parameters_BHZ_.total_particles_]);
            //Parameters_BHZ_.mu_canonical = Observables_BHZ_.chemicalpotential(Parameters_BHZ_.total_particles_, mu_guess);
            Observables_BHZ_.mu = Observables_BHZ_.chemicalpotential(Parameters_BHZ_.total_particles_, mu_guess);
        //    cout<<"Guess_mu="<<mu_guess<<", mu_canonical="<<Parameters_BHZ_.mu_canonical<<endl;

        //    Parameters_BHZ_.mu_canonical = Observables_BHZ_.chemicalpotential2(Parameters_BHZ_.total_particles_);

            //Obtaining new order parameters:
            Observables_BHZ_.getNewOPs();
            auto energies = Observables_BHZ_.getEnergies();

            OP_error = Observables_BHZ_.getOPError();

            //file_out_progress << iter << "      " << OP_error <<"   "<< Parameters_BHZ_.mu_canonical << "  " << energies.first << "    " << energies.second<<endl;
            file_out_progress << iter << "      " << OP_error <<"   "<< Observables_BHZ_.mu << "  " << energies.first << "    " << energies.second<<endl;

            //Updating the order parameters:
            Observables_BHZ_.updateOrderParams(iter);

            OPs_old = Observables_BHZ_.new_OPs_;
        }
        else{
            OP_error=0.0;

            Hamiltonian_BHZ_.OPs_ = OPs_old;
            Hamiltonian_BHZ_.ionic = ionic_old;
            
            //Building connection matrix
            Hamiltonian_BHZ_.connectionMatrix();

            //Diagonalizing the Hamiltonian with new OP's
            Hamiltonian_BHZ_.Diagonalizer();

            //Fixing the chemical potential
            Observables_BHZ_.mu = Parameters_BHZ_.mu_fixed;

            //Obtaining new order parameters:
            Observables_BHZ_.getNewOPs();
            auto energies = Observables_BHZ_.getEnergies();

            OP_error = Observables_BHZ_.getOPError();

            file_out_progress << iter << "      " << OP_error << "  " << energies.first << "    " << energies.second<<endl;

            //Updating the order parameters:
            Observables_BHZ_.updateOrderParams(iter);

            OPs_old = Observables_BHZ_.new_OPs_;

            if(Parameters_BHZ_.sitewiseIonic==true){
                Observables_BHZ_.updateIonicPotential();
                double ionic_error = 1e-3;
                double ionic_relax = Observables_BHZ_.ionic_relaxation;
                
                for(int rx=0;rx<ionic_old.size();rx++){
                    if(abs(Observables_BHZ_.new_ionic[rx] - ionic_old[rx])>ionic_error){
                        ionic_old[rx] = (1.0 - ionic_relax)*Hamiltonian_BHZ_.ionic[rx] + ionic_relax*Observables_BHZ_.new_ionic[rx];
                    }
                }
            }

        }

        iter++;
    }


    string File_out_ionic="Ionic_potential_profile.txt";
    ofstream file_out_ion(File_out_ionic.c_str());
    for(int rx=0; rx<MFParam_BHZ_.ionic_size;rx++){
        file_out_ion<<rx<<"     "<<Hamiltonian_BHZ_.ionic[rx]<<endl;
    }


    int cell, r;
    string File_out_OPs="Final_OPs.txt";
    ofstream file_out_OPs(File_out_OPs.c_str());
    for(int spin=0;spin<Hamiltonian_BHZ_.N_spin_;spin++){
        for(int rx=0; rx<Hamiltonian_BHZ_.lx_;rx++){
            for(int ry=0; ry<Hamiltonian_BHZ_.ly_;ry++){
                cell = ry + Hamiltonian_BHZ_.ly_*rx;
                for(int orb=0; orb<Hamiltonian_BHZ_.N_orbs_; orb++){               

                    r = Hamiltonian_BHZ_.makeIndex(rx, ry, orb, spin);

                    //file_out_OPs<<cell<<"   "<<orb<<"   "<<spin<<"  "<<Observables_BHZ_.new_OPs_[r].real()<<endl;
                    file_out_OPs<< r <<"    "<<Observables_BHZ_.new_OPs_[r].real()<<"   "<<Observables_BHZ_.new_OPs_[r].imag()<<endl;
                }
            }
        }
    }

    if(Parameters_BHZ_.get_ldoe==true){
        cout<<"Now calculating LDOE"<<endl;
        Observables_BHZ_.calculateLDOE();
    }
    if(Parameters_BHZ_.get_dos==true){
        cout<<"Now calculating DOS"<<endl;
        Observables_BHZ_.calculateDOS();
    }
    if(Parameters_BHZ_.get_sc==true){
        cout<<"Now calculating spin currents"<<endl;
        Observables_BHZ_.calculateSpinCurrents();
    }
    if(Parameters_BHZ_.get_Akxw==true){
        Observables_BHZ_.calculateAkxw();
    }
    if(Parameters_BHZ_.get_Akyw==true){
        cout<<"Now calculating Akyw"<<endl;
        Observables_BHZ_.calculateAkyw();
    }
    if(Parameters_BHZ_.get_ldos==true){
        Observables_BHZ_.calculateLDOS();
    }
    if(Parameters_BHZ_.get_wave_fn==true){
        Observables_BHZ_.calculateWaveFunctions();
    }
    if(Parameters_BHZ_.get_SScorr==true){
    Observables_BHZ_.calculateSpinSpinCorr();
    }

    return 0;
}
