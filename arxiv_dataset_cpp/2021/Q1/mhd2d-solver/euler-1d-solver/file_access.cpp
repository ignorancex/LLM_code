#include "euler1d.hpp"

void openFile( t_output &output ){
   if( output.filename == "stdout" )
      output.file = stdout;
   else {
      output.file = fopen( output.filename.c_str(), "w" );
      if( !output.file ){
         OUT << "! Error: Unable to open file " << output.filename << " for writing." << LF;
         exit(1);
      }
   }
}

void closeFile( t_output output ){
   if( output.file != stdout )
      fclose( output.file );
}

// Data input doesn't clash with gnuplot,
// redirection can be used for data file as-is
void inputData( const std::string filename,
                t_output &output_grid, t_output &output_non_grid, t_params &params, t_data &data ){
   using boost::property_tree::ptree;
   ptree pt;
   read_ini( filename, pt );

   std::string tempstr;

   params.gamma = pt.get<double>( "problem.gamma" );

   params.nx = pt.get<int>( "problem.Nx" );

   tempstr = pt.get<std::string>( "time.mode" );
   if( tempstr == "constant" ){
      params.time_mode = TIME_MODE_CONSTANT;
      params.steps  = pt.get<int>   ( "time.steps" );
      params.dt_max = pt.get<double>( "time.dt" );
   } else if( tempstr == "variable" ){
      params.time_mode = TIME_MODE_VARIABLE;
      params.cfl_number = pt.get<double>( "time.cfl_number" );
      params.dt_max     = pt.get<double>( "time.dt_max" );
      params.dt_min     = pt.get<double>( "time.dt_min" );
      params.t_max      = pt.get<double>( "time.t_max" );
   }
   data.dt        = params.dt_max;
   data.t_current = 0.0;

   data.U = create_vectors( PRB_DIM, NX );
   data.u = create_vectors( VEL_DIM, NX );
   data.p = create_vector( NX );

   tempstr = pt.get<std::string>( "problem.type" );
   if( tempstr == "shock tube" ){
      double Lx = pt.get<double>( "problem.Lx" ); params.dx = Lx/params.nx;
      params.start_x = pt.get<double>( "problem.start_x" );

      params.problem_type = PROBLEM_NORMAL;

      double left[PRB_DIM];
      double right[PRB_DIM];
      left[0]      = pt.get<double>( "problem.rho_l" );
      left[1]      = pt.get<double>( "problem.u_l" );
      left[2]      = pt.get<double>( "problem.p_l" );
      right[0]     = pt.get<double>( "problem.rho_r" );
      right[1]     = pt.get<double>( "problem.u_r" );
      right[2]     = pt.get<double>( "problem.p_r" );

      for( int i = NXFIRST; i < NXLAST; i++ ){
         if( (i-NX/2) < 0 ){
            data.U[0][i] = left[0];
            data.u[0][i] = left[1];
            data.p[i]    = left[2];
         } else {
            data.U[0][i] = right[0];
            data.u[0][i] = right[1];
            data.p[i]    = right[2];
         }
      }
      toConservation( params, data );

      tempstr = pt.get<std::string>( "problem.boundary" );
      if( tempstr == "open" )
         params.boundary = BOUNDARY_OPEN;
      else if( tempstr == "periodic" )
         params.boundary = BOUNDARY_PERIODIC;
      else {
         OUT << "! Error: Unknown boundary condition." << LF;
         exit(1);
      }
   } else if( tempstr == "piston" ){
      double Lx = pt.get<double>( "problem.Lx" ); params.dx = Lx/params.nx;
      params.start_x = pt.get<double>( "problem.start_x" );

      params.problem_type = PROBLEM_PISTON;
      params.param_dbl_n  = 5;
      params.param_int_n  = 1;
      params.param_dbl    = create_vector( params.param_dbl_n );
      params.param_int    = (int *)malloc( params.param_int_n*sizeof( *params.param_int ) );
      params.param_dbl[0] = pt.get<double>( "problem.piston_x" );
      params.param_dbl[1] = pt.get<double>( "problem.piston_u" );
      params.param_int[0] = (params.param_dbl[0]-params.start_x)/params.dx + NXFIRST;


      double left[PRB_DIM];
      double right[PRB_DIM];
      left[0]      = pt.get<double>( "problem.rho_l" );
      left[1]      = pt.get<double>( "problem.u_l" );
      left[2]      = pt.get<double>( "problem.p_l" );
      right[0]     = pt.get<double>( "problem.rho_r" );
      right[1]     = pt.get<double>( "problem.u_r" );
      right[2]     = pt.get<double>( "problem.p_r" );

      int piston_index = params.param_int[0];
      for( int i = NXFIRST; i < NXLAST; i++ ){
         if( i < piston_index ){
            data.U[0][i] = left[0];
            data.u[0][i] = left[1];
            data.p[i]    = left[2];
         } else {
            data.U[0][i] = right[0];
            data.u[0][i] = right[1];
            data.p[i]    = right[2];
         }
      }
      toConservation( params, data );

      params.param_dbl[2] = data.U[0][NXFIRST];
      params.param_dbl[3] = data.U[1][NXFIRST];
      params.param_dbl[4] = data.U[2][NXFIRST];

      tempstr = pt.get<std::string>( "problem.boundary" );
      if( tempstr == "open" )
         params.boundary = BOUNDARY_OPEN;
      else if( tempstr == "periodic" )
         params.boundary = BOUNDARY_PERIODIC;
      else {
         OUT << "! Error: Unknown boundary condition." << LF;
         exit(1);
      }
   } else if( tempstr == "riemann solver" ){
      double Lx = pt.get<double>( "problem.Lx" ); params.dx = Lx/params.nx;
      params.start_x = pt.get<double>( "problem.start_x" );

      params.problem_type = PROBLEM_RIEMANN;
      params.time_stepping = STEP_RIEMANN;

      params.param_dbl_n  = 6;
      params.param_int_n  = 1;
      params.param_dbl    = create_vector( params.param_dbl_n );

      params.param_int[0] = pt.get<int>( "problem.riemann_iterations" );

      params.param_dbl[0] = pt.get<double>( "problem.rho_l" );
      params.param_dbl[1] = pt.get<double>( "problem.u_l" );
      params.param_dbl[2] = pt.get<double>( "problem.p_l" );
      params.param_dbl[3] = pt.get<double>( "problem.rho_r" );
      params.param_dbl[4] = pt.get<double>( "problem.u_r" );
      params.param_dbl[5] = pt.get<double>( "problem.p_r" );

      for( int i = NXFIRST; i < NXLAST; i++ ){
         if( (i-NX/2) < 0 ){
            data.U[0][i] = params.param_dbl[0];
            data.u[0][i] = params.param_dbl[1];
            data.p[i]    = params.param_dbl[2];
         } else {
            data.U[0][i] = params.param_dbl[3];
            data.u[0][i] = params.param_dbl[4];
            data.p[i]    = params.param_dbl[5];
         }
      }
      toConservation( params, data );

      params.boundary = BOUNDARY_OPEN;
   }

   if( params.problem_type != PROBLEM_RIEMANN ){
      tempstr = pt.get<std::string>( "problem.time_method" );
      if( tempstr == "euler" )
         params.time_stepping = STEP_EULER;
      else if( tempstr == "rk3" )
         params.time_stepping = STEP_RK3TVD;
      else {
         OUT << "! Error: Time stepping method not available." << LF;
         exit(1);
      }

      tempstr = pt.get<std::string>( "problem.space_method" );
      if( tempstr == "central_fd" )
         params.scheme = METHOD_CENTRAL_FD;
      else if( tempstr == "eno" )
         params.scheme = METHOD_ENO;
      else {
         OUT << "! Error: Space integration method not available." << LF;
         exit(1);
      }
   }

   output_grid.filename = pt.get<std::string>( "output_grid.datafile" );
   output_grid.natural      = pt.get<bool>( "output_grid.natural" );
   output_grid.conservation = pt.get<bool>( "output_grid.conservation" );

   tempstr = pt.get<std::string>( "output_grid.mode" );
   if( tempstr == "step" ){
      output_grid.skip_mode  = OUT_MODE_STEP;
      output_grid.skip_steps = pt.get<int>( "output_grid.skip_steps" );
   } else if( tempstr == "time" ){
      output_grid.skip_mode = OUT_MODE_TIME;
      output_grid.skip_t    = pt.get<double>( "output_grid.skip_t" );
   } else {
      OUT << "! Error: Output mode not available." << LF;
      exit(1);
   }

   output_non_grid.filename = pt.get<std::string>( "output_non_grid.datafile" );

   tempstr = pt.get<std::string>( "output_non_grid.mode" );
   if( tempstr == "step" ){
      output_non_grid.skip_mode  = OUT_MODE_STEP;
      output_non_grid.skip_steps = pt.get<int>( "output_non_grid.skip_steps" );
   } else if( tempstr == "time" ){
      output_non_grid.skip_mode = OUT_MODE_TIME;
      output_non_grid.skip_t    = pt.get<double>( "output_non_grid.skip_t" );
   } else {
      OUT << "! Error: Output mode not available." << LF;
      exit(1);
   }
}

// gnuplot-ready format
void outputGridData( const t_output &output, const t_params &params, const t_data &data, int step, int index ){
   t_vectors U = data.U;
   t_vectors u = data.u;
   t_vector  p = data.p;

   // find maximum velocity
   double uu, uumax = 0.0;
   for( int i = NXFIRST; i < NXLAST; i++ ){
      uu = u[0][i]*u[0][i];
      if( uu > uumax ) uumax = uu;
   }

   fprintf( output.file, "# record #%d, step #%d, t = %.5f, u_max = %.2f\n", index, step, data.t_current, sqrt(uumax) );
   fprintf( output.file, "# 1)i  2)x  3)density  4)velocity  5)pressure  6)moment  7)energy\n" );

   //double start_x;
   //start_x = -(NXLAST-NXFIRST+4)/2.0*params.dx;

   for( int i = NXFIRST; i < NXLAST; i++ ){
      if( params.problem_type == PROBLEM_PISTON && i < params.param_int[0] )
         continue;

      fprintf( output.file, "%d\t%+.5e\t%+.5e",
               i-NXFIRST,
               params.start_x + (i-NXFIRST)*params.dx, // x
               U[0][i]      // rho
      );
      if( output.natural )
         fprintf( output.file, "\t%+.5e\t%+.5e",
                  u[0][i],  // u
                  p[i]      // p
         );
      if( output.conservation )
         fprintf( output.file, "\t%+.5e\t%+.5e",
                  U[1][i],  // mx
                  U[2][i]   // e
         );
      fprintf( output.file, "\n" );
   }
   fprintf( output.file, "\n\n" );
   fflush( output.file );
}

void outputNonGridData( const t_output &output, const t_params &params, const t_data &data, int step, int index ){
   static bool initialized = false;
   t_vectors U = data.U;
   t_vectors u = data.u;
   t_vector  p = data.p;

   // find maximum velocity
   double totenergy   = 0.0;
   double totentropy  = 0.0;
   double totmass     = 0.0;
   double totmomentum = 0.0;
   double uu, uumax   = 0.0;
   for( int i = NXFIRST; i < NXLAST; i++ ){
      uu = u[0][i]*u[0][i];
      if( uu > uumax ) uumax = uu;

      totenergy   += U[2][i];
      totentropy  +=  log( p[i]*pow( U[0][i], -params.gamma ) );
      totmass     += U[0][i];
      totmomentum += U[0][i]*u[0][i];
   }
   totenergy   *= params.dx;
   totentropy  *= params.dx;
   totmass     *= params.dx;
   totmomentum *= params.dx;

   if( !initialized ){
      fprintf( output.file, "# 1)record_index 2)step 3)t 4)dt 5)u_max 6)total_mass 7)total_momentum 8)total_energy 9)total_entropy\n" );
      initialized = true;
   }

   fprintf( output.file, "%d\t%d\t%.5e\t%.5e\t%.2e\t%.8e\t%.8e\t%.8e\t%.8e\n", index, step, data.t_current, data.dt, sqrt(uumax), totmass, totmomentum, totenergy, totentropy );

   fflush( output.file );
}
