#include <Rcpp.h>
using namespace Rcpp;

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp 
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

// [[Rcpp::export]]
List SEIRDV8cpp( double S0, double E0, double I0, double II0, double RE0, double RI0, 
                 double SI0, double RR0, double D0, double V0, NumericVector alpha11v, 
                 double beta1, NumericVector gamma1v, double gamma2, double eta1, NumericVector rho1v,
                 NumericVector phi1v, double zeta1, double zeta2, double kappa1, int n1 ) 
{
  // Create containers for the output
  NumericVector S0out( n1 );
  NumericVector E0out( n1 );
  NumericVector I0out( n1 );
  NumericVector II0out( n1 );
  NumericVector RE0out( n1 );
  NumericVector RI0out( n1 );
  NumericVector SI0out( n1 );
  NumericVector RR0out( n1 );
  NumericVector D0out( n1 );
  NumericVector V0out( n1 );
  NumericVector R0out( n1 );
  
  double S0n = S0;
  double E0n = E0;
  double I0n = I0;
  double II0n = II0;
  double RE0n = RE0;
  double RI0n = RI0;
  double SI0n = SI0;
  double RR0n = RR0;
  double D0n  = D0;
  double V10n = V0;
  double Rep0 = 0.0;
  
  for( int i = 0; i < n1; i++ ){
    double S0n = S0;
    double E0n = E0;
    double I0n = I0;
    double II0n = II0;
    double RE0n = RE0;
    double RI0n = RI0;
    double SI0n = SI0;
    double RR0n = RR0;
    double D0n  = D0;
    double V0n = V0;
    
    //+ alpha11v[i]*RI0n*E0n + alpha11v[i]*V0n*E0n
    
    S0 = std::max(0.0, S0n - alpha11v[i]*S0n*E0n - rho1v[i]*S0n);
    
    E0 = std::max(0.0, E0n + alpha11v[i]*S0n*E0n - beta1*E0n - gamma1v[i]*E0n - rho1v[i]*E0n);
    
    I0 = std::max(0.0, I0n + beta1*E0n - gamma1v[i]*I0n - eta1*I0n);
    
    RE0 = std::max(0.0, RE0n + gamma1v[i]*E0n - rho1v[i]*RE0n);
    
    RI0 = std::max(0.0, RI0n + gamma1v[i]*I0n -  zeta1*RI0n);
    
    SI0 = std::max(0.0, SI0n + zeta1*RI0n - phi1v[i]*alpha11v[i]*SI0n*E0n + zeta2*(1-kappa1)*V0n );
    
    II0 = std::max(0.0, II0n + phi1v[i]*alpha11v[i]*SI0n*E0n - II0n);  
    
    RR0 = std::max(0.0, RR0n + II0n );
    
    V0 = std::max(0.0, V0n  + rho1v[i]*S0n + rho1v[i]*E0n + rho1v[i]*RE0n - zeta2*(1-kappa1)*V0n);
    
    D0 = std::max(0.0, D0n + eta1*I0n );
    
    //Rep0 = alpha11v[i]*S0n/(beta1 + gamma1v[i] + rho1v[i] );
    
    
    S0out[ i ] = S0;
    E0out[ i ] = E0;
    I0out[ i ] = I0;
    II0out[ i ] = II0;
    
    RE0out[ i ] = RE0;
    RI0out[ i ] = RI0;
    SI0out[ i ] = SI0;
    RR0out[ i ] = RR0;
    
    D0out[ i ] = D0;
    V0out[ i ] = V0;
    
    R0out[ i ]= Rep0;
    
  }
  List Out2 = List::create( Named("S") = S0out, 
                            _["SI"] = SI0out,
                            _["E"] = E0out, 
                            _["I"] = I0out, 
                            _["II"] = II0out,
                            
                            _["RE"] = RE0out, 
                            _["RI"] = RI0out,
                            _["RR"] = RR0out,
                            
                            _["D"] = D0out, 
                            _["V"] = V0out,
                            
                            _["R0"] = R0out);
  return Out2;
}



