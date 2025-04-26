#include "euler1d.hpp"

// Convert conservation variables to natural variables
void toNatural( const t_params &params,
                t_data &data ){
   t_vectors U = data.U;
   t_vectors u = data.u;
   t_vector  p = data.p;
   for( int i = NXFIRST; i < NXLAST; i++ ){
      u[0][i] = U[1][i] / U[0][i]; // u = mx/rho
      p[i] = (params.gamma-1.0)*( U[2][i] - 0.5*U[1][i]*U[1][i]/U[0][i] );
   }
}

// Convert natural variables to conservation variables
void toConservation( const t_params &params,
                     t_data &data ){
   t_vectors U = data.U;
   t_vectors u = data.u;
   t_vector  p = data.p;
   for( int i = NXFIRST; i < NXLAST; i++ ){
      U[1][i] = u[0][i] * U[0][i]; // mx = u*rho
      U[2][i] = p[i]/(params.gamma-1.0) + 0.5*U[0][i]*u[0][i]*u[0][i];
   }
}
