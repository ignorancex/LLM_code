#include "def.hpp"
#include "tools.hpp"  
#include "fourier.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <sstream>

using namespace std;




int main(int argc, char* argv[])  
{

    string datafile = "./data/Pb/mve/glauber_mve_2";


    AmplitudeLib dipole(datafile);
    dipole.SetFTMethod(ACC_SERIES);
    set_fpu_state();
    init_workspace_fourier(1000); // max number of besselJ zeros
    set_fourier_precision(1.0e-12,1.0e-12);
    
    double N = 0.694;   //parameter in fragmentation function
    double e = 0.101;   //parameter in fragmentation function
    double mc = 1.5;  //charm quark mass in GeV
    double kD = StrToReal(argv[1]);   //momentum Dmeson
    double calls = 1e5;
    double ss = 5360.0; //energy in GeV
    double y = 0.5;       // Rapidity
    int Z = 82;      // Atomic number
    double  A = 208;     // Mass number
    double S = std::pow(17.4, 2) ; //parameter in 0n emission probability in fm2
    //double S = std::pow(10.4,2);
    double M = 208* 0.931; //mass in u to GeV

    
    parameters par;
    par.datafile = datafile;  
    par.dipole = &dipole;
    par.N = N;    
    par.e = e;    
    par.mc = mc;  
    par.kD = kD;   
    par.calls = calls;
    par.ss = ss;       
    par.y = y;       
    par.Z = Z;      
    par.A = A;     
    par.S = S; 
    par.M = M; 

    ///////////////////////////////////////////////
   
    
    double result = integralZf(&par);

    cout << par.kD << " " << result << endl;

    return 0;
}
