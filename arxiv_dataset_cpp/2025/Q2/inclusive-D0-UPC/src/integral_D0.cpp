#include "def.hpp"
#include "tools.hpp"
#include <iostream>
#include <gsl/gsl_integration.h> 



using namespace std;

double Fun(double zf, void* p)
{
  parameters *par = ((parameters*)p);
  par->zf = zf;
  
   double N = par->N; 
   double e = par->e;
   double mc = par->mc;
   double kD = par-> kD;
   double ss = par->ss;
   double y = par->y;

   
   double mt = sqrt( SQR(mc) + SQR(kD/zf) ); 
   double xbj = (mt / ss) * exp(-(y));
   double kp = (mt / sqrt(2)) * exp(y);
   par->xbj = xbj;
   par -> mt =mt;
   par -> kp =kp;
   
   if(xbj>0.01) xbj=0.01;   //Maximim value for xbj!
   par -> xbj=xbj;

   par->dipole->InitializeInterpolation(xbj);

       
    // Compute FT of the dipole at fixed x
    par->dipole->InitializeFTInterpolator(xbj);
  
   //std::cout << "xbj: " << xbj << std::endl;
   
     
   double parton_level = partonicCrossSection(par); 

   double FragFunction = (1./(zf*zf)) * N* zf*SQR(1-zf)/SQR(SQR(1-zf)+e*zf);

   return parton_level * FragFunction;

}



double integralZf(parameters* par) {
    
     gsl_set_error_handler_off();

    gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(1000); // Allocate workspace

    gsl_function F; 
    F.function = &Fun; 
    F.params = par; 
    

    double result, error; // Variables to hold the result and error estimate

    double lower_limit = 0.05; // Lower limit of integration
    double upper_limit = 1.0; // Upper limit of integration

    
    gsl_integration_qag(&F, lower_limit, upper_limit, 0, 1e-3, 1000, GSL_INTEG_GAUSS61, workspace, &result, &error);

    if (error/std::abs(result)> 0.1){
    cerr <<  "Note: uncertainty is more than 10%, the relative uncertainty is" << error/std::abs(result) << endl;
    }
    //std::cout << "Result of integration: " << result << " ± " << error << std::endl;

    
    gsl_integration_workspace_free(workspace);

    return result;
}
