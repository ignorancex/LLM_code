#include "mhd2d.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void toNaturalPoint
   ( const t_params &params
   , double U[PRB_DIM]
   , double u[VEL_DIM]
   , double &p
){
   double mm, bb;

   u[0] = U[1] / U[0]; // u = mx/rho
   u[1] = U[2] / U[0]; // v = my/rho
   u[2] = U[3] / U[0]; // w = mz/rho
   mm = U[1]*U[1] + U[2]*U[2] + U[3]*U[3];
   bb = U[4]*U[4] + U[5]*U[5] + U[6]*U[6];
   p = (params.gamma-1.0)*( U[7] - 0.5*mm/U[0] - 0.5*bb );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void toConservationPoint
   ( const t_params &params
   , double U[PRB_DIM]
   , double u[VEL_DIM]
   , double &p
){
   double uu, bb;

   U[1] = u[0] * U[0]; // mx = u*rho
   U[2] = u[1] * U[0]; // my = v*rho
   U[3] = u[2] * U[0]; // mz = w*rho
   uu = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
   bb = U[4]*U[4] + U[5]*U[5] + U[6]*U[6];
   U[7] = p/(params.gamma-1.0) + 0.5*U[0]*uu + 0.5*bb;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void toNaturalData
   ( const t_params &params
   , t_data &data
){
   double mm, bb;
   t_matrices U = data.U;
   t_matrices u = data.u;
   t_matrix   p = data.p;
   for( int i = NXFIRST; i < NXLAST; i++ ){
      for( int j = NYFIRST; j < NYLAST; j++ ){
         u[0][i][j] = U[1][i][j] / U[0][i][j]; // u = mx/rho
         u[1][i][j] = U[2][i][j] / U[0][i][j]; // v = my/rho
         u[2][i][j] = U[3][i][j] / U[0][i][j]; // w = mz/rho
         mm = U[1][i][j]*U[1][i][j] + U[2][i][j]*U[2][i][j] + U[3][i][j]*U[3][i][j];
         bb = U[4][i][j]*U[4][i][j] + U[5][i][j]*U[5][i][j] + U[6][i][j]*U[6][i][j];
         p[i][j] = (params.gamma-1.0)*( U[7][i][j] - 0.5*mm/U[0][i][j] - 0.5*bb );
      }
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void toConservationData
   ( const t_params &params
   , t_data &data
){
   double uu, bb;
   t_matrices U = data.U;
   t_matrices u = data.u;
   t_matrix   p = data.p;
   for( int i = NXFIRST; i < NXLAST; i++ ){
      for( int j = NYFIRST; j < NYLAST; j++ ){
         U[1][i][j] = u[0][i][j] * U[0][i][j]; // mx = u*rho
         U[2][i][j] = u[1][i][j] * U[0][i][j]; // my = v*rho
         U[3][i][j] = u[2][i][j] * U[0][i][j]; // mz = w*rho
         uu = u[0][i][j]*u[0][i][j] + u[1][i][j]*u[1][i][j] + u[2][i][j]*u[2][i][j];
         bb = U[4][i][j]*U[4][i][j] + U[5][i][j]*U[5][i][j] + U[6][i][j]*U[6][i][j];
         U[7][i][j] = p[i][j]/(params.gamma-1.0) + 0.5*U[0][i][j]*uu + 0.5*bb;
      }
   }
}
