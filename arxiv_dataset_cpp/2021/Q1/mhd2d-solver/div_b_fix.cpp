#include "mhd2d.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void divBCalculation
   ( t_matrices U
   , const t_params &params
   , double &maxdivb
   , double &totdivb
){
   // indexes of x, y magnetic field elements of the state vector
   const int kbx = 4, kby = 5;
   int nxfirst = NXFIRST;
   int nxlast  = NXLAST;
   int nyfirst = NYFIRST;
   int nylast  = NYLAST;

   // apply boundary conditions

   // left boundary
   switch( params.boundary[params.b_left] ){
   case BoundaryCondition::Undefined:
      break;
   case BoundaryCondition::Periodic:
      for( int k = kbx; k <= kby; k++ ){
         for( int j = NYFIRST; j < NYLAST; j++ ){
            U[k][NXFIRST-2][j] = U[k][NXLAST-2][j];
            U[k][NXFIRST-1][j] = U[k][NXLAST-1][j];
         }
      }
      break;
   case BoundaryCondition::Dirichlet:
   case BoundaryCondition::Neumann:
      nxfirst = NXFIRST+1;
      break;
   case BoundaryCondition::Open:
      nxfirst = NXFIRST+1;
      break;
   }

   // right boundary
   switch( params.boundary[params.b_right] ){
   case BoundaryCondition::Undefined:
      break;
   case BoundaryCondition::Periodic:
      for( int k = kbx; k <= kby; k++ ){
         for( int j = NYFIRST; j < NYLAST; j++ ){
            U[k][NXLAST][j]    = U[k][NXFIRST][j];
            U[k][NXLAST+1][j]  = U[k][NXFIRST+1][j];
         }
      }
      break;
   case BoundaryCondition::Dirichlet:
   case BoundaryCondition::Neumann:
      nxlast  = NXLAST-1;
      break;
   case BoundaryCondition::Open:
      nxlast  = NXLAST-1;
      break;
   }

   // bottom boundary
   switch( params.boundary[params.b_bottom] ){
   case BoundaryCondition::Undefined:
      break;
   case BoundaryCondition::Periodic:
      for( int k = kbx; k <= kby; k++ ){
         for( int i = NXFIRST; i < NXLAST; i++ ){
            U[k][i][NYFIRST-2] = U[k][i][NYLAST-2];
            U[k][i][NYFIRST-1] = U[k][i][NYLAST-1];
         }
      }
      break;
   case BoundaryCondition::Dirichlet:
   case BoundaryCondition::Neumann:
      nyfirst = NYFIRST+1;
      break;
   case BoundaryCondition::Open:
      nyfirst = NYFIRST+1;
      break;
   }

   // top boundary
   switch( params.boundary[params.b_top] ){
   case BoundaryCondition::Undefined:
      break;
   case BoundaryCondition::Periodic:
      for( int k = kbx; k <= kby; k++ ){
         for( int i = NXFIRST; i < NXLAST; i++ ){
            U[k][i][NYLAST]    = U[k][i][NYFIRST];
            U[k][i][NYLAST+1]  = U[k][i][NYFIRST+1];
         }
      }
      break;
   case BoundaryCondition::Dirichlet:
   case BoundaryCondition::Neumann:
      nylast  = NYLAST-1;
      break;
   case BoundaryCondition::Open:
      nylast  = NYLAST-1;
      break;
   }

   // find max |divb|, total |divb|
   double divb;
   maxdivb = 0.0;
   totdivb = 0.0;
   /*#ifdef OPENMP
   // omp-id: div_b_fix:div_b:5
   # pragma omp parallel \
     default( shared ) \
     private ( divb )

   # pragma omp for reduction ( + : totdivb ), reduction ( max : maxdivb )
   #endif*/
   for( int i = nxfirst; i < nxlast; i++ ){
      for( int j = nyfirst; j < nylast; j++ ){
         divb = fabs( (U[kbx][i+1][j]-U[kbx][i-1][j])/(2.0*params.dx)
                    + (U[kby][i][j+1]-U[kby][i][j-1])/(2.0*params.dy) );
         if( divb > maxdivb ) maxdivb = divb;
         totdivb += divb;
      }
   }

   // convert sum into integral
   totdivb *= params.dx*params.dy;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status divBCorrectionSOR
   ( t_matrices U
   , const t_params &params
){
   static char initialized = 0;
   static t_matrix phi;

   if( !initialized ){
      phi = createMatrix( NX, NY );

      // initialize mag. potential
      for( int i = 0; i < NX; i++ )
         for( int j = 0; j < NY; j++ )
            phi[i][j] = 0.0;

      initialized = 1;
   }


   // precompute factors used in calculation
   double bxfac = 4.0/(2.0*params.dx);
   double byfac = 4.0/(2.0*params.dy);
   double pxfac = 1.0/(params.dx*params.dx);
   double pyfac = 1.0/(params.dy*params.dy);

   // indexes of x, y magnetic field elements of the state vector
   const int kbx = 4, kby = 5;
   int nxfirst = NXFIRST;
   int nxlast  = NXLAST;
   int nyfirst = NYFIRST;
   int nylast  = NYLAST;

   // apply boundary conditions

   // left boundary
   switch( params.boundary[params.b_left] ){
   case BoundaryCondition::Undefined:
      break;
   case BoundaryCondition::Periodic:
      for( int k = kbx; k <= kby; k++ ){
         for( int j = NYFIRST; j < NYLAST; j++ ){
            U[k][NXFIRST-2][j] = U[k][NXLAST-2][j];
            U[k][NXFIRST-1][j] = U[k][NXLAST-1][j];
         }
      }
      break;
   case BoundaryCondition::Dirichlet:
   case BoundaryCondition::Neumann:
      nxfirst = NXFIRST+1;
      break;
   case BoundaryCondition::Open:
      nxfirst = NXFIRST+1;
      break;
   }

   // right boundary
   switch( params.boundary[params.b_right] ){
   case BoundaryCondition::Undefined:
      break;
   case BoundaryCondition::Periodic:
      for( int k = kbx; k <= kby; k++ ){
         for( int j = NYFIRST; j < NYLAST; j++ ){
            U[k][NXLAST][j]    = U[k][NXFIRST][j];
            U[k][NXLAST+1][j]  = U[k][NXFIRST+1][j];
         }
      }
      break;
   case BoundaryCondition::Dirichlet:
   case BoundaryCondition::Neumann:
      nxlast  = NXLAST-1;
      break;
   case BoundaryCondition::Open:
      nxlast  = NXLAST-1;
      break;
   }

   // bottom boundary
   switch( params.boundary[params.b_bottom] ){
   case BoundaryCondition::Undefined:
      break;
   case BoundaryCondition::Periodic:
      for( int k = kbx; k <= kby; k++ ){
         for( int i = NXFIRST; i < NXLAST; i++ ){
            U[k][i][NYFIRST-2] = U[k][i][NYLAST-2];
            U[k][i][NYFIRST-1] = U[k][i][NYLAST-1];
         }
      }
      break;
   case BoundaryCondition::Dirichlet:
   case BoundaryCondition::Neumann:
      nyfirst = NYFIRST+1;
      break;
   case BoundaryCondition::Open:
      nyfirst = NYFIRST+1;
      break;
   }

   // top boundary
   switch( params.boundary[params.b_top] ){
   case BoundaryCondition::Undefined:
      break;
   case BoundaryCondition::Periodic:
      for( int k = kbx; k <= kby; k++ ){
         for( int i = NXFIRST; i < NXLAST; i++ ){
            U[k][i][NYLAST]    = U[k][i][NYFIRST];
            U[k][i][NYLAST+1]  = U[k][i][NYFIRST+1];
         }
      }
      break;
   case BoundaryCondition::Dirichlet:
   case BoundaryCondition::Neumann:
      nylast  = NYLAST-1;
      break;
   case BoundaryCondition::Open:
      nylast  = NYLAST-1;
      break;
   }

   double r, r_max = 0.0, phi_max = 0.0;
   double coef = 1.0/(2*(pxfac+pyfac));
   double omega;
   if( params.divb_sor_omega_use_static ){
      omega = params.divb_sor_omega;
   } else {
      double t = cos(params.dx)+cos(params.dy);// could be cos(M_PI/params.nx)+cos(M_PI/params.ny) ?
      omega = (8.0-sqrt(64.0-16.0*t*t))/(t*t);
   }
   int converge_step = 0;
   for( int step = 0; step < params.divb_sor_rsteps; step++ ){
      // update result
      r_max = 0.0;
      phi_max = 0.0;

      for( int i = nxfirst; i < nxlast; i++ ){
         for( int j = nyfirst; j < nylast; j++ ){
            r  = pxfac*( phi[i+2][j] /*- 2.0*phi[i][j]*/ + phi[i-2][j] );
            r += pyfac*( phi[i][j+2] /*- 2.0*phi[i][j]*/ + phi[i][j-2] );
            r += bxfac*( U[4][i+1][j] - U[4][i-1][j] );
            r += byfac*( U[5][i][j+1] - U[5][i][j-1] );
            r *= coef;

            r -= phi[i][j];
            phi[i][j] += omega*r;

            if( fabs(r) > r_max ){
               r_max = fabs(r);
            }
            if( fabs(phi[i][j]) > phi_max ){
               phi_max = fabs(phi[i][j]);
            }
         }
      }

      // left boundary
      switch( params.boundary[params.b_left] ){
      case BoundaryCondition::Undefined:
         break;
      case BoundaryCondition::Periodic:
         for( int j = NYFIRST; j < NYLAST; j++ ){
            phi[NXFIRST-2][j] = phi[NXLAST-2] [j];
            phi[NXFIRST-1][j] = phi[NXLAST-1] [j];
         }
         break;
      case BoundaryCondition::Dirichlet:
         break;
      case BoundaryCondition::Neumann:
         break;
      case BoundaryCondition::Open:
         for( int j = NYFIRST; j < NYLAST; j++ ){
            phi[NXFIRST-1][j] = phi[NXFIRST][j];
         }
         break;
      }

      // right boundary
      switch( params.boundary[params.b_right] ){
      case BoundaryCondition::Undefined:
         break;
      case BoundaryCondition::Periodic:
         for( int j = NYFIRST; j < NYLAST; j++ ){
            phi[NXLAST]   [j] = phi[NXFIRST]  [j];
            phi[NXLAST+1] [j] = phi[NXFIRST+1][j];
         }
         break;
      case BoundaryCondition::Dirichlet:
         break;
      case BoundaryCondition::Neumann:
         break;
      case BoundaryCondition::Open:
         for( int j = NYFIRST; j < NYLAST; j++ ){
            phi[NXLAST][j]    = phi[NXLAST-1][j];
         }
         break;
      }

      // bottom boundary
      switch( params.boundary[params.b_bottom] ){
      case BoundaryCondition::Undefined:
         break;
      case BoundaryCondition::Periodic:
         for( int i = NXFIRST; i < NXLAST; i++ ){
            phi[i][NYFIRST-2]  = phi[i][NYLAST-2];
            phi[i][NYFIRST-1]  = phi[i][NYLAST-1];
         }
         break;
      case BoundaryCondition::Dirichlet:
         break;
      case BoundaryCondition::Neumann:
         break;
      case BoundaryCondition::Open:
         for( int i = NXFIRST; i < NXLAST; i++ ){
            phi[i][NYFIRST-1] = phi[i][NYFIRST];
         }
         break;
      }

      // top boundary
      switch( params.boundary[params.b_top] ){
      case BoundaryCondition::Undefined:
         break;
      case BoundaryCondition::Periodic:
         for( int i = NXFIRST; i < NXLAST; i++ ){
            phi[i][NYLAST]     = phi[i][NYFIRST];
            phi[i][NYLAST+1]   = phi[i][NYFIRST+1];
         }
         break;
      case BoundaryCondition::Dirichlet:
         break;
      case BoundaryCondition::Neumann:
         break;
      case BoundaryCondition::Open:
         for( int i = NXFIRST; i < NXLAST; i++ ){
            phi[i][NYLAST]    = phi[i][NYLAST-1];
         }
         break;
      }

      if( params.log_params.r_step )
         OUT << "# PHI LOOP @ step #" << step << " rmax = " << r_max/phi_max << LF;

      // check correctness
      if( r_max/phi_max <= params.divb_sor_rmax ){
         converge_step = step;
         break;
      }
   }

   if( params.log_params.r_end ){
      if( r_max/phi_max > params.divb_sor_rmax ){
         OUT << "# RMAX CONVERGE FAILED"
             << "; rmax = " << r_max/phi_max << " / " << params.divb_sor_rmax << LF;
      } else {
         OUT << "# RMAX CONVERGED @ step " << converge_step
             << "; rmax = " << r_max/phi_max << " / " << params.divb_sor_rmax << LF;
      }
   }

   // if failed to converge, return error flag
   if( r_max/phi_max > params.divb_sor_rmax ){
      return { true, ReturnStatus::ErrorNotConverged, "SOR divergence cleaning failed to converge" };
   }

   // correct magnetic field
   bxfac = 1.0/(2.0*params.dx);
   byfac = 1.0/(2.0*params.dy);
   for( int i = nxfirst-1; i < nxlast+1; i++ ){
      for( int j = nyfirst-1; j < nylast+1; j++ ){
         U[4][i][j] += /*bxfac**/( phi[i+1][j] - phi[i-1][j] )*bxfac;
         U[5][i][j] += /*byfac**/( phi[i][j+1] - phi[i][j-1] )*byfac;
      }
   }

   // converged, return success flag
   return { false, ReturnStatus::OK, "" };
}
