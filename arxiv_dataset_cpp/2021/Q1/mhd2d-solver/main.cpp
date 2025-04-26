/******************************************************************************
 *                                                                            *
 *                Numerical simulation of ideal MHD equations                 *
 *                             in two dimensions                              *
 *                                                                            *
 ******************************************************************************
 * Author: Tretler Rudolf                                                     *
 ******************************************************************************
 * Tatsuno Lab                                                                *
 * University of Electro-Communications, Tokyo                                *
 ******************************************************************************
 * Source code for research.                                                  *
 *                                                                            *
 *                                                                 2014-12-18 *
 ******************************************************************************/

#include "mhd2d.hpp"

#include <exception>
#ifdef __linux__
#include <execinfo.h>
#endif // __linux__

#include "spatialmethodcentralfd2.hpp"
#include "spatialmethodenoroe.hpp"
#include "spatialmethodenolf.hpp"

#include "timeintegrationeuler.hpp"
#include "timeintegrationrk3.hpp"

#ifdef USE_THREAD_EXCEPTIONS
/// http://stackoverflow.com/questions/11828539/elegant-exceptionhandling-in-openmp
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ThreadException::ThreadException
   (
): Ptr(nullptr) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ThreadException::~ThreadException
   (
){
   this->Rethrow();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ThreadException::Rethrow
   (
){
   if(this->Ptr) std::rethrow_exception(this->Ptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ThreadException::CaptureException
   (
){
   std::unique_lock<std::mutex> guard(this->Lock);
   this->Ptr = std::current_exception();
}
#endif // USE_THREAD_EXCEPTIONS

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void defaultExceptionHandler();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main
   ( int    argc
   , char **argv
){
   std::set_terminate( defaultExceptionHandler );

   namespace nchrono = ::std::chrono;

   #ifdef DEBUG
      OUT << "Program arguments, argc = " << argc << LF;
      for( int i = 0; i < argc; i++ ){
         OUT << " - argv[" << i << "] = " << argv[i] << LF;
      }
   #endif // DEBUG

   /* Variable declarations */
   // output file information
   t_output output_grid;
   t_output output_non_grid;
   t_output output_characteristics;
   // parameters
   t_params params;
   // variables
   t_data data;

   /* Clocks */
   nchrono::steady_clock::time_point clstart, clmain;
   nchrono::duration<double> duration;
   clstart = nchrono::steady_clock::now();

   /* Check command line arguments */
   if( argc < 2 ){
      OUT << "Usage: " << argv[0] << " config_file\n";
      exit(0);
   }

   /* Initialize */
   // Output files
   inputData( argv[1], output_grid, output_non_grid, params, data );
   output_characteristics = output_grid;
   output_characteristics.single_file = false;
   output_characteristics.binary = true;
   output_characteristics.filename = std::string{"char-"} + output_grid.filename;
   if( output_grid.single_file ){
      openFile( output_grid );
   }
   openFile( output_non_grid );
   outputGridData( output_grid, params, data, 0, 0 );
   outputNonGridData( output_non_grid, params, data, 0, 0 );

   // Spatial integrator
   auto bufferWidth = size_t{NXFIRST};
   auto boundary = t_boundary{ params.boundary[params.b_right]
                             , params.boundary[params.b_top]
                             , params.boundary[params.b_left]
                             , params.boundary[params.b_bottom] };
   auto method_ptr = std::unique_ptr<SpatialIntegrationMethod>{};
   auto stepper = std::unique_ptr<TimeIntegrationMethod>{};
   switch( params.scheme ){
   case IntegrationMethod::Undefined:
      criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                   + "main: Unknown space integration method." );
   case IntegrationMethod::CentralFD:
      method_ptr = std::unique_ptr<SpatialIntegrationMethod>{ new SpatialMethodCentralFD2( params.nx, params.ny, bufferWidth, params.dx, params.dy, boundary, params.gamma ) };
      break;
   case IntegrationMethod::ENO_Roe:
      method_ptr = std::unique_ptr<SpatialIntegrationMethod>{ new SpatialMethodEnoRoe( params.nx, params.ny, bufferWidth, params.dx, params.dy, boundary, params.gamma ) };
      break;
   case IntegrationMethod::ENO_LF:
      method_ptr = std::unique_ptr<SpatialIntegrationMethod>{ new SpatialMethodEnoLF( params.nx, params.ny, bufferWidth, params.dx, params.dy, boundary, params.gamma ) };
      break;
   }
   method_ptr->initializeDirichletBoundaries( data.U );

   // Time integrator
   switch( params.time_stepping ){
   case TimeStepMethod::Undefined:
      criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                   + "main: Unknown time stepping method." );
   case TimeStepMethod::Euler:
      stepper = std::unique_ptr<TimeIntegrationMethod>{ new TimeIntegrationEuler( params.nx, params.ny, bufferWidth, params.dt_min, params.dt_max, params.cfl_number, std::move(method_ptr) ) };
      break;
   case TimeStepMethod::RungeKutta3_TVD:
      stepper = std::unique_ptr<TimeIntegrationMethod>{ new TimeIntegrationRK3( params.nx, params.ny, bufferWidth, params.dt_min, params.dt_max, params.cfl_number, std::move(method_ptr) ) };
      break;
   }

   // Frequency of reporting progress to stdout
   int step_progress = 0;
   if( params.time_mode == TimeStepMode::Constant )
      step_progress = (params.steps>100) ? params.steps/100 : 1;

   // main loop variables
   bool output_to_screen, output_to_file, output_to_non_grid;
   bool done = false;
   int step = 1;
   int record_index = 1;
   t_status retval;
   double output_time_file   = output_grid.skip_t;
   double output_time_screen = params.t_max/100.0;
   double output_time_non_grid = output_non_grid.skip_t;

   /* main loop clock */
   clmain = nchrono::steady_clock::now();
   duration = clmain - clstart;
   OUT << "@ t = " << duration.count() << " sec : Init finished.\n";

   /* main loop */
   while( !done ){
      // step
      #ifdef OLD_STYLE
         switch( params.time_stepping ){
         case TimeStepMethod::Undefined:
            criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                         + "main: Unknown time stepping method." );
            break;
         case TimeStepMethod::Euler:
            retval = stepEuler( data.U, data.dt, params );
            break;
         case TimeStepMethod::RungeKutta3_TVD:
            retval = stepRK3TVD( data.U, data.dt, params );
            break;
         }
      #else
         retval = stepper->step( data.U, data.cx, data.cy, data.LUx, data.LUy, data.borderFlux, data.dt );
      #endif // OLD_STYLE

      // Process the return value
      if( retval.status == ReturnStatus::ErrorTimeUnderflow ){
         ERROUT << "ERROR: main: Time step size smaller than the smallest allowed value.\n"
                << "       Reached at step #" << step << ", at t = " << data.t_current << ".\n"
                << "       The simulation will now terminate." << LF;
         break;
      }
      if( retval.isError ){
         // other errors
         ERROUT << "ERROR: main: Terminating simulation." << LF;
         ERROUT << retval.message;
         break;
      }

      // correct div B
      if( step%params.divb_skip_steps == 0 ){
         switch( params.divb_method ){
         case DivBCorrectionMethod::Undefined:
            break;
         case DivBCorrectionMethod::SOR:
            if( params.divb_sor_steps <= 0 ){
               break;
            }
            double divbsum, divbmax;
            for( int i = 0; i < params.divb_sor_steps; i++ ){
               divBCalculation( data.U, params, divbmax, divbsum );
               if( params.log_params.divb_step ){
                  OUT << "divb corrector, step #" << step << ", loop #" << i << ": divbmax = " << divbmax << "\n";
               }
               if( divbmax < params.divb_sor_max ){
                  if( params.log_params.divb_step ){
                     OUT << "divb corrector, step #" << step << ": divb converged in " << i << " steps.\n";
                  }
                  break;
               }
               retval = divBCorrectionSOR( data.U, params );
               if( retval.status == ReturnStatus::ErrorNotConverged ){
                  ERROUT << "WARNING: Div B corrector, r failed to converge.\n";
                  break;
               }
            }
            divBCalculation( data.U, params, divbmax, divbsum );
            if( divbmax > params.divb_sor_max ){
               ERROUT << "WARNING: Div B corrector failed to converge: divbmax = " << divbmax << "\n"
                      << "         Simulation resumed, but B may be erroneous." << LF;
               //break;
            }
            break;
         }
      }

      // update time
      data.t_current += data.dt;
#ifdef DEBUG_SHOW_SIMULATED_TIME
      OUT << "*** DEBUG: step = " << step << " / " << params.steps
          << ", t = " << data.t_current << LF;
#endif // DEBUG_SHOW_SIMULATED_TIME

      // time mode-specific behavior
      switch( params.time_mode ){
      case TimeStepMode::Undefined:
         criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                      + "main: Unknown time stepping mode." );
         break;
      case TimeStepMode::Constant:
         done = step == params.steps;
         output_to_screen = step%step_progress == 0;
         break;
      case TimeStepMode::Variable:
         done = data.t_current >= params.t_max;
         output_to_screen = data.t_current >= output_time_screen;
         while( output_to_screen && ( output_time_screen <= data.t_current ) )
            output_time_screen += params.t_max/100.0;
         break;
      }

      // output mode-specific behavior
      switch( output_grid.skip_mode ){
      case WriteSkipMode::Undefined:
         criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                      + "main: Unknown grid output mode." );
         break;
      case WriteSkipMode::Step:
         output_to_file = step%output_grid.skip_steps == 0;
         break;
      case WriteSkipMode::Time:
         output_to_file = data.t_current >= output_time_file;
         while( output_to_file && ( output_time_file <= data.t_current ) )
            output_time_file += output_grid.skip_t;
         break;
      }
      switch( output_non_grid.skip_mode ){
      case WriteSkipMode::Undefined:
         criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                      + "main: Unknown non-grid output mode." );
         break;
      case WriteSkipMode::Step:
         output_to_non_grid = step%output_non_grid.skip_steps == 0;
         break;
      case WriteSkipMode::Time:
         output_to_non_grid = data.t_current >= output_time_non_grid;
         while( output_to_non_grid && ( output_time_non_grid <= data.t_current ) )
            output_time_non_grid += output_non_grid.skip_t;
         break;
      }

      // output data to file
      if( output_to_file || output_to_non_grid )
         toNaturalData( params, data );
      if( output_to_file ){
         outputGridData( output_grid, params, data, step, record_index++ );
         if( params.log_params.characteristics && step > 0
             && ( params.scheme == IntegrationMethod::ENO_Roe || params.scheme == IntegrationMethod::ENO_LF ) ){
            outputCharacteristicsBinary( output_characteristics, params, data.cx, data.cy, data.LUx, data.LUy, data.t_current, step, record_index );
         }
      }
      if( output_to_non_grid )
         outputNonGridData( output_non_grid, params, data, step, record_index );


      // output progress report to screen
      if( output_to_screen ){
         duration = nchrono::steady_clock::now() - clstart;
         switch( params.time_mode ){
         case TimeStepMode::Undefined:
            criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                         + "main: Unknown time stepping mode." );
            break;
         case TimeStepMode::Constant:
            OUT << "@ t = " << duration.count() << " sec : "
                << step << " / " << params.steps << " steps ("
                << (100.0*step/params.steps) << "%) done." << LF;
            break;
         case TimeStepMode::Variable:
            OUT << "@ t = " << duration.count() << " sec : "
                << step << " steps done, "
                << data.t_current << " / " << params.t_max << " simulated time ("
                << (100.0*data.t_current/params.t_max) << "%)." << LF;
            break;
         }
      }

      // stop if NaN encountered
      if( std::isnan( data.U[0][NXFIRST][NYFIRST] ) ){
         ERROUT << "WARNING: NaN detected. Stopping simulation at step " << step
                << ", t = " << data.t_current << ", with "
                << record_index-1 << " grids recorded." << LF;
         done = true;
      }

      step++;
   }

   /* Cleanup */
   if( output_grid.single_file ){
      closeFile( output_grid );
   }
   closeFile( output_non_grid );

   // output program stats
   duration = nchrono::steady_clock::now() - clstart;
   OUT << "@ t = " << duration.count() << " sec : Simulation finished.\n"
       << " - Grids recorded     : " << record_index-1 << "\n"
       << " - Successful steps   : " << step-1 << "\n"
       << " - Simulated time     : " << data.t_current << "\n";
   duration = nchrono::steady_clock::now() - clmain;
   OUT << " - Main loop duration : " << duration.count() << " seconds\n"
       << " - Time step average  : " << duration.count()/(step-1) << " seconds.\n";

   return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void criticalError
   ( ReturnStatus       error
   , const std::string  message
){
   ERROUT << "Critical error!\n"
          << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
          << message << "\n"
          << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
          << "Terminating execution.\n";

   auto errcode = int{0};

   switch( error ){
   case ReturnStatus::OK                     : errcode = 0; break;
   case ReturnStatus::NoChange               : errcode = 0; break;
   case ReturnStatus::Updated                : errcode = 0; break;

   case ReturnStatus::ErrorNotImplemented    : errcode = 100; break;
   case ReturnStatus::ErrorTimeUnderflow     : errcode = 101; break;
   case ReturnStatus::ErrorNotConverged      : errcode = 102; break;
   case ReturnStatus::ErrorNegativeDensity   : errcode = 103; break;
   case ReturnStatus::ErrorNegativePressure  : errcode = 104; break;

   case ReturnStatus::ErrorFileNotFound      : errcode = 200; break;
   case ReturnStatus::ErrorFailedToParse     : errcode = 201; break;
   case ReturnStatus::ErrorWrongParameter    : errcode = 202; break;
   }

   exit( errcode );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void defaultExceptionHandler
   (
){
   ERROUT << "ERROR: Unhandled exception caught. Aborting.";

   #ifdef __linux__
      // Print backtrace (Linux only).
      // Source: https://stackoverflow.com/questions/3355683/c-stack-trace-from-unhandled-exception/3356421#3356421
      void *trace_elems[20];
      int trace_elem_count( backtrace( trace_elems, 20 ) );
      char **stack_syms( backtrace_symbols( trace_elems, trace_elem_count ) );

      ERROUT << "Backtrace:\n";
      for( int i = 0 ; i < trace_elem_count ; ++i ){
         ERROUT << i << ": " << stack_syms[i] << "\n";
      }
      free( stack_syms );
   #endif // __linux__

   std::abort();
}
