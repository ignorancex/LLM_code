
#include "amplitudelib.hpp"
#include "def.hpp"
#include "fourier.h"
#include <string>      
#include <iostream>    
#include <iomanip>   
#include <fstream>
#include <cmath>
#include <gsl/gsl_monte_vegas.h>
#include <gsl/gsl_rng.h>    
#include <gsl/gsl_errno.h>
#include <sstream>


using namespace std;


//////Compute the cross section for c-quark production
////// Here mt is defined with KD and zf in the integral_D0.cpp file
 
double partonicCrossSection( void* p)
{
    parameters *par = ((parameters*)p); 
    
    double N = par->N; 
    double mt = par->mt;
    double m = par->mc;
    double kD = par-> kD;
    double kp = par-> kp;
    double y = par -> y;
    double ss = par -> ss;
    double A = par -> A;
    double xbj = par -> xbj;
    double calls = par -> calls;
    string datafile = par -> datafile;
    
    // Read data
    //AmplitudeLib inst(datafile);  
    par->dipole->SetOutOfRangeErrors(false);
   
    // Compute the J0 transfer
     gsl_set_error_handler_off();
    // init_workspace_fourier(1000);
     //set_fourier_precision(1.0e-12,1.0e-12);
   
    
    double R = 1.2*std::pow(A, 1./3.) - 0.86*std::pow(A, -1./3.) ;  //radius in fm

    // Setup random number generator
    const gsl_rng_type *T;  
    gsl_rng *r;    
    gsl_rng_env_setup ();  
    T = gsl_rng_default;  
    r = gsl_rng_alloc (T); 


    // Function to be integrated for GSL
    gsl_monte_function F;
    F.f = &integrand;
    F.dim = 4;
    F.params = par;

    double max = 400;
    double low[] = {0,0,kp,2*R};   
    double up[] = {2*M_PI,30,400,100};
    
    double res, err;

    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc (F.dim);

    // Quick warmup
    gsl_monte_vegas_integrate (&F, low, up, F.dim, calls/10, r, s, &res, &err);
   
   int iter=0;
   
    do
      {
        gsl_monte_vegas_integrate (&F, low, up, F.dim, calls, r, s, &res, &err);
        //cout << "# result = " << res << " relative uncertainty " << err/res << " chi^2/dof " <<  gsl_monte_vegas_chisq (s) << endl;
        iter=iter+1;

      }
    while ((abs (gsl_monte_vegas_chisq (s) - 1.0) > 0.5 or err/res > 0.01) and iter < 6); 
    
    
    
    //cout << param.k << " " << res << " " << err << "# re. uncert. " << err/res << " chi^2/dof " <<  gsl_monte_vegas_chisq (s)  << " E: " << ss << " y: " << y << endl;


    gsl_monte_vegas_free (s);
    gsl_rng_free(r); 

    return res; 


}
    
