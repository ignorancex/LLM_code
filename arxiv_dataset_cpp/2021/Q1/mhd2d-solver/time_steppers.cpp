#include "mhd2d.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status update_dt
   ( double &dt
   , double dt_step
   , const t_params &params
){
   double dt_old = dt;
   if( params.time_mode == TimeStepMode::Variable ){
      // resize dt to fit between 0.5*cfl and cfl
      while( dt < 0.5*params.cfl_number*dt_step && dt<=params.dt_max ){
         dt *= 2.0;
#ifdef DEBUG_DT_UPDATE
         OUT << "*** DEBUG: raised dt to " << dt << LF;
#endif // DEBUG_DT_UPDATE
      }
      while( dt > params.cfl_number*dt_step && dt >= params.dt_min ){
         dt /= 2.0;
#ifdef DEBUG_DT_UPDATE
         OUT << "*** DEBUG: lowered dt to " << dt << LF;
#endif // DEBUG_DT_UPDATE
      }
      if( dt > params.dt_max )
         dt = params.dt_max;
      if( dt < params.dt_min )
         return { true, ReturnStatus::ErrorTimeUnderflow, "smallest allowed dt does not satisfy CFL condition\n! update_dt" };
   }

   if( EPS_EQUAL(dt_old,dt) ){
#ifdef DEBUG_DT_UPDATE
         OUT << "*** DEBUG: dt unchanged, dt = " << dt << LF;
#endif // DEBUG_DT_UPDATE
//      OUT << "dt stayed at " << dt << ".\n";
      return { false, ReturnStatus::NoChange, "" };
   } else {
#ifdef DEBUG_DT_UPDATE
         OUT << "*** DEBUG: dt updated, dt = " << dt << LF;
#endif // DEBUG_DT_UPDATE
      return { false, ReturnStatus::Updated, "" };
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status stepRK3TVD
   ( t_matrices  U
   , double     &dt
   , const t_params &params
){
   static int initialized = 0;
   static t_matrices U1, U2, UL;

   if( !initialized ){
      U1 = createMatrices( PRB_DIM, NX, NY );
      U2 = createMatrices( PRB_DIM, NX, NY );
      UL = createMatrices( PRB_DIM, NX, NY );

      initialized = 1;
   }

   // Initialize loop conditions to pass the first time
   bool dt_changed = true;
   bool dt_reduced = false;
   double dt_step, dt_old;
   t_status retval;
   while( dt_changed ){
      dt_changed = false;

      // RK3, first step - calculate finite differences
      switch( params.scheme ){
      case IntegrationMethod::Undefined:
         ERROUT << "ERROR: stepRK3TVD: Spatial integration method unknown." << LF;
         return { true, ReturnStatus::ErrorWrongParameter, "spatial integration method unknown\n! stepRK3TVD: first step" };
         break;
      case IntegrationMethod::CentralFD:
         retval = methodCentralFD( U, UL, dt_step, params );
         break;
      case IntegrationMethod::ENO_Roe:
      case IntegrationMethod::ENO_LF:
         retval = methodENOSystem( U, UL, dt_step, params );
         break;
      }
      if( retval.status != ReturnStatus::OK ){
         ERROUT << "ERROR: stepRK3TVD: First step FD." << LF;
         retval.message += "\n! stepRK3TVD: first step";
         return retval;
      }

      // RK3, first step - update dt
      dt_old = dt;
      retval = update_dt( dt, dt_step, params );
      if( retval.status == ReturnStatus::ErrorTimeUnderflow ){
         retval.message += "\n! stepRK3TVD: first step";
         return retval;
      }
      // Take care so dt doesn't rebound upwards (leads into infinite loop)
      if( dt_reduced && dt > dt_old )
         dt = dt_old;
      if( dt < dt_old ){
         dt_reduced = true;
      }

      // RK3, first step - update variables
      for( int k = 0; k < PRB_DIM; k++ )
         for( int i = NXFIRST; i < NXLAST; i++ )
            for( int j = NYFIRST; j < NYLAST; j++ )
               U1[k][i][j] = U[k][i][j] + dt*UL[k][i][j];

      // RK3, second step - calculate finite differences
      switch( params.scheme ){
      case IntegrationMethod::Undefined:
         ERROUT << "ERROR: stepRK3TVD: Spatial integration method unknown." << LF;
         return { true, ReturnStatus::ErrorWrongParameter, "spatial integration method unknown\n! stepRK3TVD: second step" };
         break;
      case IntegrationMethod::CentralFD:
         retval = methodCentralFD( U1, UL, dt_step, params );
         break;
      case IntegrationMethod::ENO_Roe:
      case IntegrationMethod::ENO_LF:
         retval = methodENOSystem( U1, UL, dt_step, params );
         break;
      }
      if( retval.status != ReturnStatus::OK ){
         ERROUT << "ERROR: stepRK3TVD: Second step FD." << LF;
         retval.message += "\n! stepRK3TVD: second step";
         return retval;
      }

      // RK3, second step - update dt
      dt_old = dt;
      retval = update_dt( dt, dt_step, params );
      if( retval.status == ReturnStatus::ErrorTimeUnderflow ){
         retval.message += "\n! stepRK3TVD: second step";
         return retval;
      }
      if( dt_reduced && dt > dt_old )
         dt = dt_old;
      else
         dt_changed = ( retval.status == ReturnStatus::Updated );
      if( dt < dt_old ){
         dt_reduced = true;
      }
      if( dt_changed ) continue;

      // RK3, second step - update variables
      for( int k = 0; k < PRB_DIM; k++ )
         for( int i = NXFIRST; i < NXLAST; i++ )
            for( int j = NYFIRST; j < NYLAST; j++ )
               U2[k][i][j] = (3.0/4.0)*U[k][i][j] + (1.0/4.0)*U1[k][i][j] + (1.0/4.0)*dt*UL[k][i][j];

      // RK3, final step - calculate finite differences
      switch( params.scheme ){
      case IntegrationMethod::Undefined:
         ERROUT << "ERROR: stepRK3TVD: Spatial integration method unknown." << LF;
         return { true, ReturnStatus::ErrorWrongParameter, "spatial integration method unknown\n! stepRK3TVD: last step" };
         break;
      case IntegrationMethod::CentralFD:
         retval = methodCentralFD( U2, UL, dt_step, params );
         break;
      case IntegrationMethod::ENO_Roe:
      case IntegrationMethod::ENO_LF:
         retval = methodENOSystem( U2, UL, dt_step, params );
         break;
      }
      if( retval.status != ReturnStatus::OK ){
         ERROUT << "ERROR: stepRK3TVD: Last step FD." << LF;
         retval.message += "\n! stepRK3TVD: last step";
         return retval;
      }

      // RK3, final step - update dt
      dt_old = dt;
      retval = update_dt( dt, dt_step, params );
      if( retval.status == ReturnStatus::ErrorTimeUnderflow ){
         retval.message += "\n! stepRK3TVD: last step";
         return retval;
      }
      if( dt_reduced && dt > dt_old )
         dt = dt_old;
      else
         dt_changed = ( retval.status == ReturnStatus::Updated );
      if( dt < dt_old ){
         dt_reduced = true;
      }
      if( dt_changed ) continue;

      // RK3, final step - update variables
      for( int k = 0; k < PRB_DIM; k++ )
         for( int i = NXFIRST; i < NXLAST; i++ )
            for( int j = NYFIRST; j < NYLAST; j++ )
               U[k][i][j] = (1.0/3.0)*U[k][i][j] + (2.0/3.0)*U2[k][i][j] + (2.0/3.0)*dt*UL[k][i][j];
   }

   // Everything OK
   return { false, ReturnStatus::OK, "" };
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status stepEuler
   ( t_matrices  U
   , double     &dt
   , const t_params &params
){
   static int initialized = 0;
   static t_matrices UL;

   if( !initialized ){
      UL = createMatrices( PRB_DIM, NX, NY );

      initialized = 1;
   }

   double dt_step;
   t_status retval;
   // Run the method
   switch( params.scheme ){
   case IntegrationMethod::Undefined:
      ERROUT << "ERROR: stepEuler: Spatial integration method unknown." << LF;
         return { true, ReturnStatus::ErrorWrongParameter, "spatial integration method unknown\n! stepEuler" };
         break;
   case IntegrationMethod::CentralFD:
      retval = methodCentralFD( U, UL, dt_step, params );
      break;
   case IntegrationMethod::ENO_Roe:
   case IntegrationMethod::ENO_LF:
      retval = methodENOSystem( U, UL, dt_step, params );
      break;
   }
   if( retval.status != ReturnStatus::OK ){
      ERROUT << "ERROR: stepEuler: FD." << LF;
      retval.message += "\n! stepEuler";
      return retval;
   }

   // Update dt
   retval = update_dt( dt, dt_step, params );
   if( retval.status == ReturnStatus::ErrorTimeUnderflow ){
      retval.message += "\n! stepEuler";
      return retval;
   }
   // Update variables
   for( int k = 0; k < PRB_DIM; k++ )
      for( int i = NXFIRST; i < NXLAST; i++ )
         for( int j = NYFIRST; j < NYLAST; j++ )
            U[k][i][j] = U[k][i][j] + dt*UL[k][i][j];

   // Everything OK
   return { false, ReturnStatus::OK, "" };
}
