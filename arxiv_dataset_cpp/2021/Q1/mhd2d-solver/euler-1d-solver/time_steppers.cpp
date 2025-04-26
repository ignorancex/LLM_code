#include "euler1d.hpp"

int update_dt( double &dt, double dt_step, const t_params &params ){
   double dt_old = dt;
   if( params.time_mode == TIME_MODE_VARIABLE ){
      // resize dt to fit between 0.5*cfl and cfl
      while( dt < 0.5*params.cfl_number*dt_step ){
         dt *= 2.0;
#ifdef DEBUG
         OUT << dt << LF;
#endif // DEBUG
      }
      while( dt > params.cfl_number*dt_step ){
         dt /= 2.0;
#ifdef DEBUG
         OUT << dt << LF;
#endif // DEBUG
      }
      if( dt > params.dt_max )
         dt = params.dt_max;
      if( dt < params.dt_min )
         return RET_ERR_TIME_UNDERFLOW;
   }

   if( EPS_EQUAL(dt_old,dt) ){
//      OUT << "dt stayed at " << dt << ".\n";
      return RET_NO_CHANGE;
   } else {
//      OUT << "dt updated from " << dt_old << " to " << dt << ".\n";
      return RET_UPDATED;
   }
}

// Third order optimal TVD Runge-Kutta time stepping method
int rk3tvd( t_vectors U, double &dt, const t_params &params ){
   static int initialized = 0;
   static t_vectors U1, U2, UL;

   if( !initialized ){
      U1 = create_vectors( PRB_DIM, NX );
      U2 = create_vectors( PRB_DIM, NX );
      UL = create_vectors( PRB_DIM, NX );

      initialized = 1;
   }

   // Initialize loop conditions to pass the first time
   bool dt_changed = true;
   bool dt_reduced = false;
   double dt_step, dt_old;
   int retval;
   while( dt_changed ){
      dt_changed = false;

      // RK3, first step - calculate finitite differences
      if( params.scheme == METHOD_CENTRAL_FD )
         central_fd( U, UL, dt_step, params );
      else if( params.scheme == METHOD_ENO )
         eno_system_roe( U, UL, dt_step, params );
      else {
         OUT << "! Error: Spatial integration method not available." << LF;
         exit(1);
      }
      // RK3, first step - update dt
      dt_old = dt;
      retval = update_dt( dt, dt_step, params );
      if( retval == RET_ERR_TIME_UNDERFLOW )
         return RET_ERR_TIME_UNDERFLOW;
      // Take care so dt doesn't rebound upwards (leads into infinite loop)
      if( dt_reduced && dt > dt_old )
         dt = dt_old;
      if( dt < dt_old )
         dt_reduced = true;
      // RK3, first step - update variables
      for( int k = 0; k < PRB_DIM; k++ )
         for( int i = NXFIRST; i < NXLAST; i++ )
            U1[k][i] = U[k][i] + dt*UL[k][i];
      // RK3, second step - calculate finitite differences
      if( params.scheme == METHOD_CENTRAL_FD )
         central_fd( U1, UL, dt_step, params );
      else if( params.scheme == METHOD_ENO )
         eno_system_roe( U1, UL, dt_step, params );
      else {
         OUT << "! Error: Spatial integration method not available." << LF;
         exit(1);
      }
      // RK3, second step - update dt
      dt_old = dt;
      retval = update_dt( dt, dt_step, params );
      if( retval == RET_ERR_TIME_UNDERFLOW )
         return RET_ERR_TIME_UNDERFLOW;
      if( dt_reduced && dt > dt_old )
         dt = dt_old;
      else
         dt_changed = retval == RET_UPDATED;
      if( dt < dt_old )
         dt_reduced = true;
      if( dt_changed ) continue;
      // RK3, second step - update variables
      for( int k = 0; k < PRB_DIM; k++ )
         for( int i = NXFIRST; i < NXLAST; i++ )
            U2[k][i] = (3.0/4.0)*U[k][i] + (1.0/4.0)*U1[k][i] + (1.0/4.0)*dt*UL[k][i];
      // RK3, final step - calculate finitite differences
      if( params.scheme == METHOD_CENTRAL_FD )
         central_fd( U2, UL, dt_step, params );
      else if( params.scheme == METHOD_ENO )
         eno_system_roe( U2, UL, dt_step, params );
      else {
         OUT << "! Error: Spatial integration method not available." << LF;
         exit(1);
      }
      // RK3, final step - update dt
      dt_old = dt;
      retval = update_dt( dt, dt_step, params );
      if( retval == RET_ERR_TIME_UNDERFLOW )
         return RET_ERR_TIME_UNDERFLOW;
      if( dt_reduced && dt > dt_old )
         dt = dt_old;
      else
         dt_changed = retval == RET_UPDATED;
      if( dt < dt_old )
         dt_reduced = true;
      if( dt_changed ) continue;
      // RK3, final step - update variables
      for( int k = 0; k < PRB_DIM; k++ )
         for( int i = NXFIRST; i < NXLAST; i++ )
            U[k][i] = (1.0/3.0)*U[k][i] + (2.0/3.0)*U2[k][i] + (2.0/3.0)*dt*UL[k][i];
   }
   // Everything OK
   return RET_OK;
}

/// Euler time stepping method
int euler_step( t_vectors U, double &dt, const t_params &params ){
   static int initialized = 0;
   static t_vectors UL;

   if( !initialized ){
      UL = create_vectors( PRB_DIM, NX );

      initialized = 1;
   }

   double dt_step;
   int retval;
   // Run the method
   if( params.scheme == METHOD_CENTRAL_FD )
      central_fd( U, UL, dt_step, params );
   else if( params.scheme == METHOD_ENO )
      eno_system_roe( U, UL, dt_step, params );
   else {
      OUT << "! Error: Spatial integration method not available." << LF;
      exit(1);
   }
   // Update dt
   retval = update_dt( dt, dt_step, params );
   if( retval == RET_ERR_TIME_UNDERFLOW )
      return RET_ERR_TIME_UNDERFLOW;
   // Update variables
   for( int k = 0; k < PRB_DIM; k++ )
      for( int i = NXFIRST; i < NXLAST; i++ )
            U[k][i] = U[k][i] + dt*UL[k][i];

   // Everything OK
   return RET_OK;
}
