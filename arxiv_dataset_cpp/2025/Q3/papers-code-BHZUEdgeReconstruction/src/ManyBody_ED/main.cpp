#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <vector>
#include <cassert>
#include <algorithm>

#include <complex>
#include <cmath>
#include <cassert>
#include <utility>

#include "tensors.h"
#include "functions.h"
#include "Hamiltonian.h"


int main(){
    //Parameters:-----------------------------//
    int Ncells=3,  Norbs=2,   Nparticles=2;

    double A_val=0.3, B_val=0.5, M_val=1.0;
    double U_val=1.0, JHbyU_=0.25;
    double J_val=JHbyU_*U_val;
    double Up_val=U_val-2.0*J_val;
//    double Up_val =0.0;
    double Vo_val=0.0;
    //----------------------------------------//

    functions f;
    f.ncells_ = Ncells;
    f.norbs_ = Norbs;
    f.nparticles_ = Nparticles;

    f.A_=A_val;     f.B_=B_val;     f.M_=M_val;
    f.U_=U_val;     f.J_=J_val;     f.Up_=Up_val;
    f.Vo_=Vo_val;
    //----------------------------------------//

    Mat_2_int basis = f.createBasis();
    cout<<"Hilbert-space size = "<<basis.size()<<endl;
    f.printBasis();

    Hamiltonian Hamiltonian_(f);    

    Hamiltonian_.connectionMatrix();
/*    for(int i=0;i<Hamiltonian_.Hsize;i++){
        for(int j=0;j<Hamiltonian_.Hsize;j++){
            cout<<Hamiltonian_.C_mat[i][j]<<" ";
        }
        cout<<endl;
    }
*/
    Hamiltonian_.Diagonalizer();
    Hamiltonian_.createSpinMatrices();
    Hamiltonian_.calculateMagneticObservables();

    //Printing first 6 Eigenvectors:
    string head("Printed_Eigenvectors_");
    string tail(".txt");
 //   string file_out_evecs_="Eigenvectors.txt";
  
    for(int n=0;n<6;n++){
        string file_out_evecs_ = head + to_string(n) + tail;
        ofstream file_out_evecs(file_out_evecs_.c_str());

        double normed=0.0;
        for(int m=0;m<Hamiltonian_.Hsize;m++){
            normed += norm(Hamiltonian_.evecs_[n][m]);
        }

        for(int m=0;m<Hamiltonian_.Hsize;m++){
            if(norm(Hamiltonian_.evecs_[n][m]) > 1e-7){
                file_out_evecs<<m<<"    "<<Hamiltonian_.evecs_[n][m]/normed<<endl;
            }
        }
    }
    

    return 0;
}

