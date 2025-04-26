#include "euler1d.hpp"


/* FIXME: dt_step calculation */
void central_fd( t_vectors U, t_vectors UL, double &dt_step,
                 const t_params &params ){
   static int initialized = 0;
   static t_vectors F;

   if( !initialized ){
      F  = create_vectors( PRB_DIM, NX );

      initialized = 1;
   }

   // boundary condition
   switch( params.boundary ){
      case BOUNDARY_OPEN:
         for( int k = 0; k < PRB_DIM; k++ ){
            U[k][NXFIRST-2] = U[k][NXFIRST];
            U[k][NXFIRST-1] = U[k][NXFIRST];
            U[k][NXLAST]    = U[k][NXLAST-1];
            U[k][NXLAST+1]  = U[k][NXLAST-1];
         }
         break;
      case BOUNDARY_PERIODIC:
         for( int k = 0; k < PRB_DIM; k++ ){
            U[k][NXFIRST-2] = U[k][NXLAST-2];
            U[k][NXFIRST-1] = U[k][NXLAST-1];
            U[k][NXLAST]    = U[k][NXFIRST];
            U[k][NXLAST+1]  = U[k][NXFIRST+1];
         }
   }

   // flux F(U)
   for( int i = 0; i < NX; i++ ){
      double r  = U[0][i];
      double mx = U[1][i]; double u = mx / r;
      double e  = U[2][i];
      double p = (params.gamma-1.0)*(e-0.5*r*u*u);

      /* rho */ F[0][i] = mx;
      /* mx  */ F[1][i] = mx*u + p;
      /* e   */ F[2][i] = (e+p)*u;
   }

   for( int k = 0; k < PRB_DIM; k++ ){
      for( int i = NXFIRST; i < NXLAST; i++ ){
         UL[k][i] = -(1.0/(2.0*params.dx)) * ( F[k][i+1] - F[k][i-1] );
      }
   }
}
