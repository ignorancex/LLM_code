#include "mhd2d.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status methodCentralFD
   ( t_matrices  /*U*/
   , t_matrices  /*UL*/
   , double     &/*dt_step*/
   , const t_params &/*params*/
){
   return { true, ReturnStatus::ErrorNotImplemented, "central finite difference is not implemented" };

//   static int initialized = 0;
//   static t_matrices F;
//   static t_matrices G;
//
//   if( !initialized ){
//      F  = createMatrices( PRB_DIM, NX, NY );
//      G  = createMatrices( PRB_DIM, NX, NY );
//
//      initialized = 1;
//   }
//
//   // boundary condition
//   switch( params.boundary ){
//      case BOUNDARY_OPEN:
//         for( int k = 0; k < PRB_DIM; k++ ){
//            for( int i = 0; i < NX; i++ ){
//               U[k][i][NYFIRST-2] = U[k][i][NYFIRST];
//               U[k][i][NYFIRST-1] = U[k][i][NYFIRST];
//               U[k][i][NYLAST]    = U[k][i][NYLAST-1];
//               U[k][i][NYLAST+1]  = U[k][i][NYLAST-1];
//            }
//            for( int j = 0; j < NY; j++ ){
//               U[k][NXFIRST-2][j] = U[k][NXFIRST][j];
//               U[k][NXFIRST-1][j] = U[k][NXFIRST][j];
//               U[k][NXLAST][j]    = U[k][NXLAST-1][j];
//               U[k][NXLAST+1][j]  = U[k][NXLAST-1][j];
//            }
//         }
//         break;
//      case BOUNDARY_PERIODIC:
//         for( int k = 0; k < PRB_DIM; k++ ){
//            for( int i = 0; i < NX; i++ ){
//               U[k][i][NYFIRST-2] = U[k][i][NYLAST-2];
//               U[k][i][NYFIRST-1] = U[k][i][NYLAST-1];
//               U[k][i][NYLAST]    = U[k][i][NYFIRST];
//               U[k][i][NYLAST+1]  = U[k][i][NYFIRST+1];
//            }
//            for( int j = 0; j < NY; j++ ){
//               U[k][NXFIRST-2][j] = U[k][NXLAST-2][j];
//               U[k][NXFIRST-1][j] = U[k][NXLAST-1][j];
//               U[k][NXLAST][j]    = U[k][NXFIRST][j];
//               U[k][NXLAST+1][j]  = U[k][NXFIRST+1][j];
//            }
//         }
//   }
//
//   // flux F(U), G(U)
//   for( int i = 0; i < NX; i++ ){
//      for( int j = 0; j < NY; j++ ){
//         double r  = U[0][i][j];
//         double mx = U[1][i][j]; double u = mx / r;
//         double my = U[2][i][j]; double v = my / r;
//         double mz = U[3][i][j]; double w = mz / r;
//         double bx = U[4][i][j];
//         double by = U[5][i][j];
//         double bz = U[6][i][j];
//         double e  = U[7][i][j];
//         double uu =  u*u   + v*v   + w*w;
//         double ub =  u*bx  + v*by  + w*bz;
//         double bb = bx*bx + by*by + bz*bz;
//         double p = (params.gamma-1.0)*(e-0.5*r*uu-0.5*bb);
//         double ptot = p + 0.5*bb;
//
//         /* rho */ F[0][i][j] = mx;
//         /* mx  */ F[1][i][j] = mx*u - bx*bx + ptot;
//         /* my  */ F[2][i][j] = my*u - bx*by;
//         /* mz  */ F[3][i][j] = mz*u - bx*bz;
//         /* bx  */ F[4][i][j] = 0;
//         /* by  */ F[5][i][j] = by*u - bx*v;
//         /* bz  */ F[6][i][j] = bz*u - bx*w;
//         /* e   */ F[7][i][j] = (e+ptot)*u - bx*ub;
//
//         /* rho */ G[0][i][j] = my;
//         /* mx  */ G[1][i][j] = mx*v - by*bx;
//         /* my  */ G[2][i][j] = my*v - by*by + ptot;
//         /* mz  */ G[3][i][j] = mz*v - by*bz;
//         /* bx  */ G[4][i][j] = bx*v - by*u;
//         /* by  */ G[5][i][j] = 0;
//         /* bz  */ G[6][i][j] = bz*v - by*w;
//         /* e   */ G[7][i][j] = (e+ptot)*v - by*ub;
//      }
//   }
//
//   for( int k = 0; k < PRB_DIM; k++ ){
//      for( int i = NXFIRST; i < NXLAST; i++ ){
//         for( int j = NYFIRST; j < NYLAST; j++ ){
//            UL[k][i][j] = -(1.0/(2.0*params.dx)) * ( F[k][i+1][j] - F[k][i-1][j] )
//                          -(1.0/(2.0*params.dy)) * ( G[k][i][j+1] - G[k][i][j-1] );
//         }
//      }
//   }
//
//   return RET_OK;
}
