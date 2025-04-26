/******************************************************************************
 *                                                                            *
 *               Numerical simulation of ideal Euler equations                *
 *                             in one dimension                               *
 *                                                                            *
 ******************************************************************************
 * Author: Tretler Rudolf                                                     *
 ******************************************************************************
 * Tatsuno Lab                                                                *
 * University of Electro-Communications, Tokyo                                *
 ******************************************************************************
 * Source code for Masters' thesis                                            *
 *                                                                            *
 *                                                                 2014-12-18 *
 ******************************************************************************/

#include "euler1d.hpp"

void apply_piston( t_data &data, t_params &params ){
   int index_old = params.param_int[0];
   double x_new = -params.start_x+params.param_dbl[0] + params.param_dbl[1]*data.t_current;
   int index_new = x_new/params.dx + NXFIRST;
   if( index_new < NXFIRST ) index_new = NXFIRST-1;

   // if piston moved to the right, push the plasma
   if( index_new > index_old ){
      // TODO
   }

   // if piston moved to the left, leave the original state behind
   if( index_new < index_old ){
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int i = index_new; i < index_old; i++ ){
            data.U[k][i] = params.param_dbl[k+2];
         }
      }
//      for( int i = index_new; i < index_old; i++ ){
//         data.U[1][i] = params.param_dbl[1]*data.U[1][i];
//      }
   }

   // fill in the rest of left side
   for( int k = 0; k < PRB_DIM; k++ ){
      for( int i = NXFIRST; i < index_new; i++ ){
         data.U[k][i] = params.param_dbl[k+2];
      }
   }

   params.param_int[0] = index_new;
}

int main( int argc, char ** argv ){
   namespace nchrono = std::chrono;
   /* variable declarations */
   // output file information
   t_output output_grid;
   t_output output_non_grid;
   // parameters
   t_params params;
   // variables
   t_data data;

   /* clocks */
   nchrono::steady_clock::time_point clstart, clmain;
   nchrono::duration<double> duration;
   clstart = nchrono::steady_clock::now();

   /* initialize */
   inputData( argv[1], output_grid, output_non_grid, params, data );
   openFile( output_grid );
   openFile( output_non_grid );
   outputGridData( output_grid, params, data, 0, 0 );
   outputNonGridData( output_non_grid, params, data, 0, 0 );

   int step_progress = 0;
   if( params.time_mode == TIME_MODE_CONSTANT )
      step_progress = (params.steps>100) ? params.steps/100 : 1;

   // main loop variables
   bool output_to_screen, output_to_file, output_to_non_grid;
   bool done = false;
   int step = 1;
   int record_index = 1;
   int retval;
   double output_time_file   = output_grid.skip_t;
   double output_time_screen = params.t_max/100.0;
   double output_time_non_grid = output_non_grid.skip_t;

   /* main loop clock */
   clmain = nchrono::steady_clock::now();
   duration = clmain - clstart;
   OUT << "@ t = " << duration.count() << " sec : Init finished.\n";

   /* main loop */
   while( !done ){
      // Problem-specific preparation
      if( params.problem_type == PROBLEM_PISTON ){
         apply_piston( data, params );
      }

      // Step
      if( params.time_stepping == STEP_EULER )
         retval = euler_step( data.U, data.dt, params );
      else if( params.time_stepping == STEP_RK3TVD )
         retval = rk3tvd( data.U, data.dt, params );
      else if( params.problem_type == PROBLEM_RIEMANN ){
         riemann_solver( data, params );
      } else {
         ERROUT << "! Error: main: Unknown time stepping mode." << LF;
         exit(1);
      }
      // Process the return value
      if( retval == RET_ERR_TIME_UNDERFLOW ){
         ERROUT << "ERROR: Time step size smaller than the smallest allowed value.\n"
                << "       Reached at step #" << step << ", at t = " << data.t_current << ".\n"
                << "       The simulation will now terminate." << LF;
         break;
      } else if( retval == RET_RIEMANN_FAILED_TO_CONVERGE ){
         ERROUT << "ERROR: Riemann solver failed to converge.\n"
                << "       Failed at step #" << step << ", at t = " << data.t_current << ".\n"
                << "       The simulation will now terminate." << LF;
         break;
      }

      // Problem-specific cleanup
      if( params.problem_type == PROBLEM_PISTON ){
         apply_piston( data, params );
      }

      // Update time
      data.t_current += data.dt;
#ifdef DEBUG
      OUT << "t = " << data.t_current << "\n";
#endif // DEBUG

      // Time mode-specific behavior
      if( params.time_mode == TIME_MODE_CONSTANT ){
         done = step == params.steps;
         output_to_screen = (step%step_progress) == 0;
      } else if( params.time_mode == TIME_MODE_VARIABLE ){
         done = data.t_current >= params.t_max;
         output_to_screen = data.t_current >= output_time_screen;
         while( output_to_screen && ( output_time_screen <= data.t_current ) )
            output_time_screen += params.t_max/100.0;
      } else {
         OUT << "! Error: main: Unknown time mode." << LF;
         exit(1);
      }

      // Output mode-specific behavior
      if( output_grid.skip_mode == OUT_MODE_STEP ){
         output_to_file = step%output_grid.skip_steps == 0;
      } else if( output_grid.skip_mode == OUT_MODE_TIME ){
         output_to_file = data.t_current >= output_time_file;
         while( output_to_file && ( output_time_file <= data.t_current ) )
            output_time_file += output_grid.skip_t;
      } else {
         ERROUT << "! Error: main: Unknown output mode for grid output." << LF;
         exit(1);
      }
      if( output_non_grid.skip_mode == OUT_MODE_STEP ){
         output_to_non_grid = step%output_non_grid.skip_steps == 0;
      } else if( output_non_grid.skip_mode == OUT_MODE_TIME ){
         output_to_non_grid = data.t_current >= output_time_non_grid;
         while( output_to_non_grid && ( output_time_non_grid <= data.t_current ) )
            output_time_non_grid += output_non_grid.skip_t;
      } else {
         ERROUT << "! Error: main: Unknown output mode for non-grid output." << LF;
         exit(1);
      }

      // Output data to file
      if( output_to_file || output_to_non_grid )
         toNatural( params, data );
      if( output_to_non_grid )
         outputNonGridData( output_non_grid, params, data, step, record_index );
      if( output_to_file )
         outputGridData( output_grid, params, data, step, record_index++ );

      // Output progress report to screen
      if( output_to_screen ){
         duration = nchrono::steady_clock::now() - clstart;
         if( params.time_mode == TIME_MODE_CONSTANT )
            OUT << "@ t = " << duration.count() << " sec : "
                << step << " / " << params.steps << " steps ("
                << (100.0*step/params.steps) << "%) done." << LF;
         else if( params.time_mode == TIME_MODE_VARIABLE )
            OUT << "@ t = " << duration.count() << " sec : "
                << step << " steps done, "
                << data.t_current << " / " << params.t_max << " simulated time ("
                << (100.0*data.t_current/params.t_max) << "%)." << LF;
      }

      step++;
   }

   /* cleanup */
   closeFile( output_grid );
   closeFile( output_non_grid );

   duration = nchrono::steady_clock::now() - clstart;
   OUT << "@ t = " << duration.count() << " sec : Simulation finished.\n";
   duration = nchrono::steady_clock::now() - clmain;
   OUT << " - Main loop :         " << duration.count() << " seconds,\n"
       << " - Time step average : " << duration.count()/(step-1) << " seconds.\n";

   return 0;
}
