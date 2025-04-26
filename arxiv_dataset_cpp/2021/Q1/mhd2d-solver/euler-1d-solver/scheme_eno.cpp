#include "euler1d.hpp"

// Eigenvalues / eigenvectors between points 1 and 2 for flux F
void get_eigens_f( double U1[PRB_DIM], double U2[PRB_DIM],
                   double gamma,
                   double lambda[PRB_DIM], double lv[PRB_DIM][PRB_DIM], double rv[PRB_DIM][PRB_DIM] );

void get_eigens_f( double U1[PRB_DIM], double U2[PRB_DIM],
                   double gamma,
                   double lambda[PRB_DIM], double lv[PRB_DIM][PRB_DIM], double rv[PRB_DIM][PRB_DIM] ){
   // preparation for half-points
   double u1 = U1[1] / U1[0];
   double u2 = U2[1] / U2[0];
   double p1 = (gamma-1.0)*( U1[2] - 0.5*U1[1]*U1[1]/U1[0] );
   double p2 = (gamma-1.0)*( U2[2] - 0.5*U2[1]*U2[1]/U2[0] );
   double h1 = (p1+U1[2]) / U1[0];
   double h2 = (p2+U2[2]) / U2[0];
   double rroot1 = sqrt( U1[0] );
   double rroot2 = sqrt( U2[0] );

   // half-point values
   double u = ( rroot1*u1 + rroot2*u2 ) / ( rroot1 + rroot2 );
   double h = ( rroot1*h1 + rroot2*h2 ) / ( rroot1 + rroot2 );

   // sound speed
   double c2 = (gamma-1.0)*( h - 0.5*u*u );
   double c  = sqrt( c2 );

   // other
   double alpha = (gamma-1.0) / c2;
   double beta  = 0.5*alpha*u*u;

   // eigenvalues
   lambda[0] = u - c;
   lambda[1] = u;
   lambda[2] = u + c;

   // right eigenvectors
   /* a1 = u - c (sound)*/
   rv[0][0] = 1.0;
   rv[0][1] = u - c;
   rv[0][2] = h - u*c;

   /* a2 = u (entropy) */
   rv[1][0] = 1.0;
   rv[1][1] = u;
   rv[1][2] = 0.5*u*u;

   /* a3 = u + c (sound) */
   rv[2][0] = 1.0;
   rv[2][1] = u + c;
   rv[2][2] = h + u*c;

   // left eigenvectors
   /* a1 = u - c (sound)*/
   lv[0][0] =  0.5*( beta + u/c );
   lv[0][1] = -0.5*( alpha*u + 1.0/c );
   lv[0][2] =  0.5*alpha;

   /* a2 = u (entropy) */
   lv[1][0] =  1.0 - beta;
   lv[1][1] =  alpha*u;
   lv[1][2] = -alpha;

   /* a3 = u + c (sound) */
   lv[2][0] =  0.5*( beta - u/c );
   lv[2][1] = -0.5*( alpha*u - 1.0/c );
   lv[2][2] =  0.5*alpha;
}

// ENO-Roe for systems of conservation laws
// r, m, e    : input, conservation variables
// rL, mL, eL : output, approximation to du/dt
void eno_system_roe( t_vectors U, t_vectors UL, double &dt_step,
                     const t_params &params ){
   static char initialized = 0;
   static t_vectors F;
   static t_vectors VUF;
   static t_vectors F_;
   static t_vectors w, Vw;

   if( !initialized ){
      F   = create_vectors( PRB_DIM, NX );
      VUF = create_vectors( PRB_DIM, NX );
      F_  = create_vectors( PRB_DIM, NX );
      w   = create_vectors( PRB_DIM, NX );
      Vw  = create_vectors( PRB_DIM, NX );

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
         break;
   }

   // problem-specific condition
   int piston_index = params.param_int[0];
   switch( params.problem_type ){
      case PROBLEM_PISTON:
         // u
         U[1][piston_index] = 0;//params.param_dbl[1]*U[0][piston_index];
         break;
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

   // undivided differences V_UF, V_UG
   for( int k = 0; k < PRB_DIM; k++ ){
      for( int i = 0; i < NX-1; i++ ){
         VUF[k][i] = F[k][i+1] - F[k][i];
      }
   }

   // eigenvalues & eigenvectors @ u_(i+1/2)
   double a[PRB_DIM], lv[PRB_DIM][PRB_DIM], rv[PRB_DIM][PRB_DIM];
   double U1[PRB_DIM], U2[PRB_DIM];
   double wm[PRB_DIM], wp[PRB_DIM], fw[PRB_DIM];

   // track maximum wave speed (ie, eigenvalue)
   double a_max_f = 0.0;

   /******************************
    *           F flux           *
    ******************************/
   for( int i = 1; i < NX-2; i++ ){
      if( params.problem_type == PROBLEM_PISTON && i < piston_index ){
         for( int k = 0; k < PRB_DIM; k++ )
            F_[k][i] = 0.0;
         continue;
      }

      // left point
      for( int k = 0; k < PRB_DIM; k++ )
         U1[k] = U[k][i];

      // right point
      for( int k = 0; k < PRB_DIM; k++ )
         U2[k] = U[k][i+1];

      // eigensystem
      get_eigens_f( U1, U2, params.gamma, a, lv, rv );
#ifdef DEBUG
      OUT << "$$ ENO: F Eigens @ " << i << " OK!\n";
#endif

      // update maximum wave speed
      for( int k = 0; k < PRB_DIM; k++ )
         if( a_max_f < fabs(a[k]) )
            a_max_f = fabs(a[k]);

      // local characteristics and undivided differences
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int l = i-1; l <= i+2; l++ ){
            w[k][l] = 0.0;
            for( int m = 0; m < PRB_DIM; m++ )
               w[k][l] += lv[k][m]*F[m][l];
         }
      }
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int l = i-1; l <= i+2; l++ ){
            Vw[k][l] = 0.0;
            for( int m = 0; m < PRB_DIM; m++ )
               Vw[k][l] += lv[k][m]*VUF[m][l];
         }
      }

      // apply open boundary
      if( params.boundary == BOUNDARY_OPEN ){
         for( int k = 0; k < PRB_DIM; k++ ){
            Vw[k][NXFIRST-2] = 1.0e200;
            Vw[k][NXFIRST-1] = 1.0e100;
            Vw[k][NXLAST-1]  = 1.0e100;
            Vw[k][NXLAST]    = 1.0e200;
            if( i < NXFIRST )
               a[k] = -1.0;
            if( i > NXLAST-2 )
               a[k] = 1.0;
         }
      }

//      // apply piston boundary
//      if( params.problem_type == PROBLEM_PISTON ){
//         for( int k = 0; k < PRB_DIM; k++ ){
//            Vw[k][piston_index-2] = 1.0e200;
//            Vw[k][piston_index-1] = 1.0e100;
//            if( i <= piston_index )
//               a[k] = 1.0;
//         }
//      }

      // minus edge
      for( int k = 0; k < PRB_DIM; k++ ){
         if( ABS(Vw[k][i-1]) < ABS(Vw[k][i]) ) // left (stencil -3/2, -1/2, +1/2)
            wm[k] = -(1.0/2.0)*w[k][i-1] + (3.0/2.0)*w[k][i];
         else // right (stencil -1/2, +1/2, +3/2)
            wm[k] = (1.0/2.0)*w[k][i] + (1.0/2.0)*w[k][i+1];
      }

      // plus edge
      for( int k = 0; k < PRB_DIM; k++ ){
         if( ABS(Vw[k][i]) < ABS(Vw[k][i+1]) ) // left (stencil -3/2, -1/2, +1/2)
            wp[k] = (1.0/2.0)*w[k][i] + (1.0/2.0)*w[k][i+1];
         else // right (stencil -1/2, +1/2, +3/2)
            wp[k] = (3.0/2.0)*w[k][i+1] - (1.0/2.0)*w[k][i+2];
      }

      // fluxes
      for( int k = 0; k < PRB_DIM; k++ ){
         if( a[k] >= 0 )
            fw[k] = wm[k];
         else
            fw[k] = wp[k];
      }
#ifdef DEBUG
      OUT << "$$ ENO: Characteristic fluxes OK!\n";
#endif

      // return to physical space fluxes
      for( int k = 0; k < PRB_DIM; k++ ){
         F_[k][i] = 0.0;
         for( int l = 0; l < PRB_DIM; l++ )
            F_[k][i] += rv[l][k]*fw[l];
      }
   }

#ifdef DEBUG
   OUT << "scheme_eno: a_max_f = " << a_max_f << "\n";
#endif // DEBUG

   // update local dt from max wavespeeds
   dt_step = params.dx/a_max_f;

   // step
   for( int k = 0; k < PRB_DIM; k++ ){
      for( int i = NXFIRST; i < NXLAST; i++ ){
         UL[k][i] = -(1.0/params.dx)*( F_[k][i] - F_[k][i-1] );
      }
   }
//   if( params.problem_type == PROBLEM_PISTON ){
//      UL[1][piston_index] = 0;
//   }
}
