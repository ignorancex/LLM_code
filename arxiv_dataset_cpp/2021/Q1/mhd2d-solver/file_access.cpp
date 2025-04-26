#include "mhd2d.hpp"
#include "file_access.hpp"

// needed for eigensystem calculation in readProblemWaveTest
#include "scheme_eno.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void openFile
   ( t_output &output
){
   if( output.filename == "stdout" ){
      output.file = stdout;
   } else {
      if( output.binary ){
         output.file = fopen( output.filename.c_str(), "wb" );
      } else {
         output.file = fopen( output.filename.c_str(), "w" );
      }
      if( !output.file ){
         criticalError( ReturnStatus::ErrorFileNotFound, std::string{}
                      + "openFile: Unable to open file '"
                      + output.filename + "' for writing." );
      }
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void closeFile
   ( t_output &output
){
   if( output.file != stdout ){
      fclose( output.file );
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Read value of a key, show warning and default value if the key is missing */
template <class T>
T readEntry
   ( boost::property_tree::ptree pt
   , std::string section
   , std::string name
   , T           defaultValue
){
   T value;

   try {
      // get value
      value = pt.get<T>( section+"."+name );
   } catch( boost::property_tree::ptree_error &err ) {
      // show warning if key is missing
      ERROUT << "WARNING: readEntry: Key \"" << name << "\" in section [" << section << "] not found.\n"
             << "         Assuming " << name << " = " << defaultValue << "\n";
      value = defaultValue;
   }

   return value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void inputData
   ( const std::string filename
   , t_output &output_grid
   , t_output &output_non_grid
   , t_params &params
   , t_data   &data
){
   namespace nptree = ::boost::property_tree;
   nptree::ptree pt;
   std::string tempstr;

   // Parse configuration file
   try {
      nptree::read_ini( filename, pt );
   } catch( std::exception &e ) {
      criticalError( ReturnStatus::ErrorFailedToParse, std::string{}
                   + "inputData: " + e.what() );
   }

   // Global problem parameters
   params.gamma = readEntry<double>( pt, "problem", "gamma", 1.4 );

   // Grid size
   params.nx = readEntry<int>( pt, "problem", "Nx", 16 );
   params.ny = readEntry<int>( pt, "problem", "Ny", 16 );

   // Prepare data storage
   data.U = createMatrices( PRB_DIM, NX, NY );
   data.u = createMatrices( VEL_DIM, NX, NY );
   data.p = createMatrix( NX, NY );
   data.cx  = createMatrices( PRB_DIM, NX, NY );
   data.cy  = createMatrices( PRB_DIM, NX, NY );
   data.LUx = createMatrices( PRB_DIM, NX, NY );
   data.LUy = createMatrices( PRB_DIM, NX, NY );
   data.borderFlux.left  = createVectors( PRB_DIM, NY );
   data.borderFlux.right = createVectors( PRB_DIM, NY );
   data.borderFlux.up    = createVectors( PRB_DIM, NX );
   data.borderFlux.down  = createVectors( PRB_DIM, NX );

   // Time stepping mode
   // - keep a constant dt, or
   // - automatically change dt to satisfy CFL condition
   tempstr = readEntry<std::string>( pt, "time", "mode", "constant" );
   params.time_mode = fromString<TimeStepMode>( tempstr );
   switch( params.time_mode ){
   case TimeStepMode::Undefined:
      criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                   + "inputData: in section [time], key \"mode\":\n"
                   + "Unknown time stepping mode: " + tempstr );
      break;
   case TimeStepMode::Constant:
      params.steps  = readEntry<int>   ( pt, "time", "steps", 8     );
      params.dt_max = readEntry<double>( pt, "time", "dt",    0.125 );
      params.cfl_number = double{ 0.0 };
      break;
   case TimeStepMode::Variable:
      params.cfl_number = readEntry<double>( pt, "time", "cfl number", 0.5 );
      params.dt_max     = readEntry<double>( pt, "time", "dt_max",     1.0 );
      params.dt_min     = readEntry<double>( pt, "time", "dt_min",     params.dt_max/16.0 );
      params.t_max      = readEntry<double>( pt, "time", "t_max",      1.0 );
      break;
   }

   // Initialize time
   data.dt        = params.dt_max;
   data.t_current = 0.0;

   // DivB correction
   readParamsDivBCorrector( pt, params );

   // Set up the problem
   tempstr = readEntry<std::string>( pt, "problem", "type", "shock tube" );
   if( tempstr == "shock tube" ){
      readProblemShockTube( pt, params, data );
   } else if( tempstr == "explosion" ){
      readProblemExplosion( pt, params, data );
   } else if( tempstr == "orszag-tang vortex" ){
      readProblemOTVortex( pt, params, data );
   } else if( tempstr == "plasma sheet" ){
      readProblemPlasmaSheet( pt, params, data );
   } else if( tempstr == "wave test" ){
      readProblemWaveTest( pt, params, data );
   } else {
      criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                   + "inputData: in section [problem], key \"type\":\n"
                   + "Unknown problem type: " + tempstr );
   }

   // Negative pressure handling
   params.break_on_neg_pressure = readEntry<bool>( pt, "problem", "halt on negative pressure", false );

   // Time integration method
   tempstr = readEntry<std::string>( pt, "problem", "time method", "rk3" );
   params.time_stepping = fromString<TimeStepMethod>( tempstr );
   if( params.time_stepping == TimeStepMethod::Undefined ){
      criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                   + "inputData: in section [problem], key \"time method\":\n"
                   + "Unknown time stepping method: " + tempstr );
   }

   // Space integration method
   tempstr = readEntry<std::string>( pt, "problem", "space method", "eno-roe" );
   params.scheme = fromString<IntegrationMethod>( tempstr );
   if( params.scheme == IntegrationMethod::Undefined ){
      criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                   + "inputData: in section [problem], key \"space method\":\n"
                   + "Unknown space integration method: " + tempstr );
   }

   // Data output file
   output_grid.filename = readEntry<std::string>( pt, "output grid", "datafile", "%f.g.dat" );
   output_grid.natural      = readEntry<bool>( pt, "output grid", "natural",      true );
   output_grid.conservation = readEntry<bool>( pt, "output grid", "conservation", true );

   // Binary or text file
   output_grid.binary = readEntry<bool>( pt, "output grid", "binary", false );
   std::string binary_hint_file;
   if( output_grid.binary ){
      output_grid.binary_double = readEntry<bool>( pt, "output grid", "binary use double", false );
      binary_hint_file = readEntry<std::string>( pt, "output grid", "binary hint", "%f.hint" );
   } else {
      binary_hint_file = "dummy";
   }

   // Single file or separate files
   output_grid.single_file = readEntry<bool>( pt, "output grid", "single file", true );

   // How often to output data, can be given as time or as steps
   tempstr = readEntry<std::string>( pt, "output grid", "mode", "step" );
   output_grid.skip_mode = fromString<WriteSkipMode>( tempstr );
   switch( output_grid.skip_mode ){
   case WriteSkipMode::Undefined:
      criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                   + "inputData: in section [output grid], key \"mode\":\n"
                   + "Unknown output mode: " + tempstr );
      break;
   case WriteSkipMode::Step:
      output_grid.skip_steps = readEntry<int>( pt, "output grid", "skip steps", 1 );
      break;
   case WriteSkipMode::Time:
      output_grid.skip_t = readEntry<double>( pt, "output grid", "skip t", params.dt_max );
      break;
   }

   // Overview data output
   output_non_grid.filename = readEntry<std::string>( pt, "output non grid", "datafile", "%f.ng.dat" );

   tempstr = readEntry<std::string>( pt, "output non grid", "mode", "step" );
   output_non_grid.skip_mode = fromString<WriteSkipMode>( tempstr );
   switch( output_non_grid.skip_mode ){
   case WriteSkipMode::Undefined:
      criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                   + "inputData: in section [output non grid], key \"mode\":\n"
                   + "Unknown output mode: " + tempstr );
      break;
   case WriteSkipMode::Step:
      output_non_grid.skip_steps = readEntry<int>( pt, "output non grid", "skip steps", 1 );
      break;
   case WriteSkipMode::Time:
      output_non_grid.skip_t = readEntry<double>( pt, "output non grid", "skip t", params.dt_max );
      break;
   }

   // Construct time string
   std::time_t tt = std::chrono::system_clock::to_time_t( std::chrono::system_clock::now() );
   struct std::tm *ptm = std::localtime(&tt);
   char buffer[80];
   strftime( buffer, 80, "%Y-%m-%d-%H-%M-%S", ptm );
   std::string timestr( buffer );
   // Get filename without the extension
   std::string mainname = filename.substr( 0, filename.find_last_of( "." ) );
   // Replace masks for output filenames
   boost::replace_all( output_grid.filename, "%f", filename );
   boost::replace_all( output_grid.filename, "%m", mainname );
   boost::replace_all( output_grid.filename, "%t", timestr  );
   boost::replace_all( output_non_grid.filename, "%f", filename );
   boost::replace_all( output_non_grid.filename, "%m", mainname );
   boost::replace_all( output_non_grid.filename, "%t", timestr  );
   boost::replace_all( binary_hint_file, "%f", filename );
   boost::replace_all( binary_hint_file, "%m", mainname );
   boost::replace_all( binary_hint_file, "%t", timestr  );
   if( output_grid.single_file ){
      // remove fields that are only used for separate output
      boost::replace_all( output_grid.filename, "%rs", "" );
      boost::replace_all( output_grid.filename, "%ri", "" );
      boost::replace_all( output_grid.filename, "%rt", "" );
   }

   // Read formatting strings
   if( output_grid.filename.find("%rs") != std::string::npos ){
      output_grid.format_step = readEntry<std::string>( pt, "output grid", "rs format", "%08d" );
   }
   if( output_grid.filename.find("%ri") != std::string::npos ){
      output_grid.format_index = readEntry<std::string>( pt, "output grid", "ri format", "%08d" );
   }
   if( output_grid.filename.find("%rt") != std::string::npos ){
      output_grid.format_simtime = readEntry<std::string>( pt, "output grid", "rt format", "%015.8f" );
   }

   // Logging parameters
   params.log_params.r_step            = readEntry<bool>( pt, "logging", "phi correction",         false );
   params.log_params.r_end             = readEntry<bool>( pt, "logging", "phi correction final",   false );
   params.log_params.divb_step         = readEntry<bool>( pt, "logging", "divb correction",        false );
   params.log_params.characteristics   = readEntry<bool>( pt, "logging", "characteristics",        false );

   // Create the hint file that records the binary file structure
   if( output_grid.binary ){
      outputBinaryHintFile( binary_hint_file, params, output_grid );
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void readParamsDivBCorrector
   ( boost::property_tree::ptree &pt
   , t_params &params
){
   std::string tempstr;

   tempstr = readEntry<std::string>( pt, "divb corrector", "method", "sor" );
   params.divb_method = fromString<DivBCorrectionMethod>( tempstr );
   switch( params.divb_method ){
   case DivBCorrectionMethod::Undefined:
      criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                   + "readParamsDivBCorrector: in section [divb corrector], key \"method\":\n"
                   + "Unknown correction method:" + tempstr );
      break;
   case DivBCorrectionMethod::SOR:
      params.divb_sor_rsteps = readEntry<double>( pt, "divb corrector", "sor rsteps",  0       );
      params.divb_sor_rmax   = readEntry<double>( pt, "divb corrector", "sor rmax",    1.0e-5  );
      params.divb_sor_steps  = readEntry<int>(    pt, "divb corrector", "sor steps",   0       );
      params.divb_sor_max    = readEntry<double>( pt, "divb corrector", "sor divbmax", 1.0e-10 );
      params.divb_sor_omega_use_static = readEntry<bool>  ( pt, "divb corrector", "sor overrelaxation param use static", false );
      params.divb_sor_omega            = readEntry<double>( pt, "divb corrector", "sor overrelaxation param",            0.0   );
      break;
   }

   // How often to calculate the correction
   params.divb_skip_steps = readEntry<int>( pt, "divb corrector", "stepping rate", 1 );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void readProblemShockTube
   ( boost::property_tree::ptree &pt
   , t_params &params
   , t_data   &data
){
   std::string tempstr;

   // Domain size and position
   double Lx = readEntry<double>( pt, "problem", "Lx", 2.0 ); params.dx = Lx/params.nx;
   double Ly = readEntry<double>( pt, "problem", "Ly", 2.0 ); params.dy = Ly/params.ny;
   params.start_x = readEntry<double>( pt, "problem", "start_x", -1.0 );
   params.start_y = readEntry<double>( pt, "problem", "start_y", -1.0 );

   // Shock tube angle to the x-axis
   double angle = readEntry<double>( pt, "problem", "interface angle", 0.0 );

   double x0 = 0.5*Lx-0.5*params.dx;
   double y0 = 0.5*Ly-0.5*params.dy;
   double slope = tan( (90.0-angle) * M_PI/180.0 );

   double left[PRB_DIM];
   double right[PRB_DIM];

   left[0]  = readEntry<double>( pt, "problem", "rho_l", 1.0   );
   left[1]  = readEntry<double>( pt, "problem", "u_l",   0.0   );
   left[2]  = readEntry<double>( pt, "problem", "v_l",   0.0   );
   left[3]  = readEntry<double>( pt, "problem", "w_l",   0.0   );
   left[4]  = readEntry<double>( pt, "problem", "Bx_l",  0.75  );
   left[5]  = readEntry<double>( pt, "problem", "By_l",  1.0   );
   left[6]  = readEntry<double>( pt, "problem", "Bz_l",  0.0   );
   left[7]  = readEntry<double>( pt, "problem", "p_l",   1.0   );

   right[0] = readEntry<double>( pt, "problem", "rho_r", 0.125 );
   right[1] = readEntry<double>( pt, "problem", "u_r",   0.0   );
   right[2] = readEntry<double>( pt, "problem", "v_r",   0.0   );
   right[3] = readEntry<double>( pt, "problem", "w_r",   0.0   );
   right[4] = readEntry<double>( pt, "problem", "Bx_r",  0.75  );
   right[5] = readEntry<double>( pt, "problem", "By_r", -1.0   );
   right[6] = readEntry<double>( pt, "problem", "Bz_r",  0.0   );
   right[7] = readEntry<double>( pt, "problem", "p_r",   0.1   );

   for( int i = NXFIRST; i < NXLAST; i++ ){
      double line_y = y0 - slope*( (i-NXFIRST)*params.dx - x0 );
      double line_j = NYFIRST + (line_y/params.dy+0.5);
      int jump = line_j < NYFIRST ? NYFIRST :
                 line_j > NYLAST  ? NYLAST  :
                                    line_j;
      for( int j = NYFIRST; j < jump; j++ ){
         data.U[0][i][j] = left[0];
         data.u[0][i][j] = left[1];
         data.u[1][i][j] = left[2];
         data.u[2][i][j] = left[3];
         data.U[4][i][j] = left[4];
         data.U[5][i][j] = left[5];
         data.U[6][i][j] = left[6];
         data.p[i][j]    = left[7];
      }
      for( int j = jump; j < NYLAST; j++ ){
         data.U[0][i][j] = right[0];
         data.u[0][i][j] = right[1];
         data.u[1][i][j] = right[2];
         data.u[2][i][j] = right[3];
         data.U[4][i][j] = right[4];
         data.U[5][i][j] = right[5];
         data.U[6][i][j] = right[6];
         data.p[i][j]    = right[7];
      }
   }
   toConservationData( params, data );

   std::vector<std::string> boundary_name(params.b_count);
   boundary_name[params.b_left  ] = "left";
   boundary_name[params.b_top   ] = "top";
   boundary_name[params.b_right ] = "right";
   boundary_name[params.b_bottom] = "bottom";
   for( int b = 0; b < params.b_count; b++ ){
      std::string boundary_key = "boundary "+boundary_name[b];
      tempstr = readEntry<std::string>( pt, "problem", boundary_key, "open" );

      params.boundary[b] = fromString<BoundaryCondition>( tempstr );
      switch( params.boundary[b] ){
      case BoundaryCondition::Undefined:
         criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                      + "readProblemShockTube: in section [problem], key \"" + boundary_key + "\":\n"
                      + "Unknown boundary condition: " + tempstr );
         break;
      case BoundaryCondition::Dirichlet:
      case BoundaryCondition::Periodic:
      case BoundaryCondition::Neumann:
      case BoundaryCondition::Open:
         break;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void readProblemExplosion
   ( boost::property_tree::ptree &pt
   , t_params &params
   , t_data   &data
){
   std::string tempstr;

   // Domain size and position
   double Lx = readEntry<double>( pt, "problem", "Lx", 2.0 ); params.dx = Lx/params.nx;
   double Ly = readEntry<double>( pt, "problem", "Ly", 2.0 ); params.dy = Ly/params.ny;
   params.start_x = readEntry<double>( pt, "problem", "start_x", -1.0 );
   params.start_y = readEntry<double>( pt, "problem", "start_y", -1.0 );

   // give meaningful names
   t_matrix U_rho = data.U[0];
   t_matrix U_u   = data.u[0];
   t_matrix U_v   = data.u[1];
   t_matrix U_w   = data.u[2];
   t_matrix U_bx  = data.U[4];
   t_matrix U_by  = data.U[5];
   t_matrix U_bz  = data.U[6];
   t_matrix U_p   = data.p;
   std::vector<t_matrix> U_natural = { U_rho, U_u, U_v, U_w, U_bx, U_by, U_bz, U_p };

   double in[PRB_DIM];
   double out[PRB_DIM];

   double r = readEntry<double>( pt, "problem", "r",       0.4   );
   auto is_inside = [&params, &r]( size_t i, size_t j ){
      return (i-NX/2)*(i-NX/2)*params.dx*params.dx + (j-NY/2)*(j-NY/2)*params.dy*params.dy < r*r;
   };

   in[0]    = readEntry<double>( pt, "problem", "rho_in",  1.0   );
   in[1]    = readEntry<double>( pt, "problem", "u_in",    0.0   );
   in[2]    = readEntry<double>( pt, "problem", "v_in",    0.0   );
   in[3]    = readEntry<double>( pt, "problem", "w_in",    0.0   );
   in[4]    = readEntry<double>( pt, "problem", "Bx_in",   0.0   );
   in[5]    = readEntry<double>( pt, "problem", "By_in",   0.0   );
   in[6]    = readEntry<double>( pt, "problem", "Bz_in",   0.0   );
   in[7]    = readEntry<double>( pt, "problem", "p_in",    1.0   );

   out[0]   = readEntry<double>( pt, "problem", "rho_out", 0.125 );
   out[1]   = readEntry<double>( pt, "problem", "u_out",   0.0   );
   out[2]   = readEntry<double>( pt, "problem", "v_out",   0.0   );
   out[3]   = readEntry<double>( pt, "problem", "w_out",   0.0   );
   out[4]   = readEntry<double>( pt, "problem", "Bx_out",  0.0   );
   out[5]   = readEntry<double>( pt, "problem", "By_out",  0.0   );
   out[6]   = readEntry<double>( pt, "problem", "Bz_out",  0.0   );
   out[7]   = readEntry<double>( pt, "problem", "p_out",   0.1   );

   for( int i = NXFIRST; i < NXLAST; i++ ){
      for( int j = NYFIRST; j < NYLAST; j++ ){
         if( is_inside(i,j) ){
            U_rho[i][j] = in[0];
            U_u  [i][j] = in[1];
            U_v  [i][j] = in[2];
            U_w  [i][j] = in[3];
            U_bx [i][j] = in[4];
            U_by [i][j] = in[5];
            U_bz [i][j] = in[6];
            U_p  [i][j] = in[7];
         } else {
            U_rho[i][j] = out[0];
            U_u  [i][j] = out[1];
            U_v  [i][j] = out[2];
            U_w  [i][j] = out[3];
            U_bx [i][j] = out[4];
            U_by [i][j] = out[5];
            U_bz [i][j] = out[6];
            U_p  [i][j] = out[7];
         }
      }
   }

   // create transition layer
   std::string transition_type = readEntry<std::string>( pt, "problem", "transition type", "jump" );
   if( transition_type == "jump" ){
      // nothing to do
   } else {
      auto transition_points = 2*readEntry<size_t>( pt, "problem", "transition points", 1 );

      // prepare transition coefficients
      std::vector<double> transition_coef( transition_points );
      if( transition_type == "linear" ){
         double diff = 1.0/transition_points;
         for( auto i = size_t{0}; i < transition_points; i++ ){
            transition_coef[i] = (transition_points-i - 0.5)*diff;
         }
      } else {
         criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                      + "readProblemPlasmaSheet: in section [problem], key \"transition type\":\n"
                      + "Unknown transition type: " + tempstr );
      }

      auto mag_pressure = []( double bx, double by, double bz ){
         return 0.5*(bx*bx+by*by+bz*bz);
      };
      auto combine = []( double coef, double first, double second ){
         return coef*first + (1.0-coef)*second;
      };

      // calculate transition values
      std::vector< std::vector<double> > transition_U( transition_points );
      for( size_t p = 0; p < transition_points; p++ ){
         transition_U[p].resize( PRB_DIM );
         for( size_t k = 0; k < PRB_DIM; k++ ){
            transition_U[p][k] = combine( transition_coef[p], in[k], out[k] );
         }
         // fix pressure
         double ptot_in  =  in[7] + mag_pressure(  in[4],  in[5],  in[6] );
         double ptot_out = out[7] + mag_pressure( out[4], out[5], out[6] );
         double ptot = combine( transition_coef[p], ptot_in, ptot_out );
         transition_U[p][7] = ptot - mag_pressure( transition_U[p][4], transition_U[p][5], transition_U[p][6] );
      }

      // find border points
      std::vector< std::pair< size_t, size_t > > border_up;
      std::vector< std::pair< size_t, size_t > > border_dn;
      std::vector< std::pair< size_t, size_t > > border_lt;
      std::vector< std::pair< size_t, size_t > > border_rt;
      for( auto i = NXFIRST; i < NXLAST; i++ ){
         for( auto j = NYFIRST; j < NYLAST; j++ ){
            if( !is_inside(i,j) ) continue;
            if( !is_inside(i-1,j) ) border_lt.push_back( { i , j } );
            if( !is_inside(i+1,j) ) border_rt.push_back( { i , j } );
            if( !is_inside(i,j-1) ) border_dn.push_back( { i , j } );
            if( !is_inside(i,j+1) ) border_up.push_back( { i , j } );
         }
      }

      // apply transition, from inside out
      for( size_t p = 0; p < transition_points; p++ ){
         auto p_dif = p - transition_points/2 + 1;
         for( size_t k = 0; k < PRB_DIM; k++ ){
            for( auto &point: border_up ){
               U_natural[k][point.first][point.second+p_dif] = transition_U[p][k];
            }
            for( auto &point: border_dn ){
               U_natural[k][point.first][point.second-p_dif] = transition_U[p][k];
            }
            for( auto &point: border_lt ){
               U_natural[k][point.first-p_dif][point.second] = transition_U[p][k];
            }
            for( auto &point: border_rt ){
               U_natural[k][point.first+p_dif][point.second] = transition_U[p][k];
            }
         }
      }
   }



   toConservationData( params, data );

   std::vector<std::string> boundary_name(params.b_count);
   boundary_name[params.b_left  ] = "left";
   boundary_name[params.b_top   ] = "top";
   boundary_name[params.b_right ] = "right";
   boundary_name[params.b_bottom] = "bottom";
   for( int b = 0; b < params.b_count; b++ ){
      std::string boundary_key = "boundary "+boundary_name[b];
      tempstr = readEntry<std::string>( pt, "problem", boundary_key, "open" );

      params.boundary[b] = fromString<BoundaryCondition>( tempstr );
      switch( params.boundary[b] ){
      case BoundaryCondition::Undefined:
         criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                      + "readProblemExplosion: in section [problem], key \"" + boundary_key + "\":\n"
                      + "Unknown boundary condition: " + tempstr );
         break;
      case BoundaryCondition::Dirichlet:
      case BoundaryCondition::Periodic:
      case BoundaryCondition::Neumann:
      case BoundaryCondition::Open:
         break;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void readProblemOTVortex
   ( boost::property_tree::ptree &pt
   , t_params &params
   , t_data   &data
){
   double x, y;

   // Domain size and position
   double Lx = 2.0*M_PI; params.dx = Lx/params.nx;
   double Ly = 2.0*M_PI; params.dy = Ly/params.ny;
   double x0 = readEntry<double>( pt, "problem", "x0 percent", 0.0 );
   double y0 = readEntry<double>( pt, "problem", "y0 percent", 0.0 );
   params.start_x = x0*2.0*M_PI;
   params.start_y = y0*2.0*M_PI;

   for( int i = NXFIRST; i < NXLAST; i++ ){
      for( int j = NYFIRST; j < NYLAST; j++ ){
         x       = params.start_x + (i-NXFIRST)*params.dx;
         y       = params.start_y + (j-NYFIRST)*params.dy;

         /* rho */ data.U[0][i][j] = params.gamma*params.gamma;
         /*  u  */ data.u[0][i][j] = -sin(y);
         /*  v  */ data.u[1][i][j] =  sin(x);
         /*  w  */ data.u[2][i][j] = 0;
         /*  Bx */ data.U[4][i][j] = -sin(y);
         /*  By */ data.U[5][i][j] =  sin(2*x);
         /*  Bz */ data.U[6][i][j] = 0;
         /*  p  */ data.p   [i][j] = params.gamma;
      }
   }
   toConservationData( params, data );

   for( int b = 0; b < params.b_count; b++ ){
      params.boundary[b] = BoundaryCondition::Periodic;
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void readProblemPlasmaSheet
   ( boost::property_tree::ptree &pt
   , t_params &params
   , t_data   &data
){
   std::string tempstr;

   // Domain size and position
   double Lx = readEntry<double>( pt, "problem", "Lx", 2.0 ); params.dx = Lx/params.nx;
   double Ly = readEntry<double>( pt, "problem", "Ly", 2.0 ); params.dy = Ly/params.ny;
   params.start_x = readEntry<double>( pt, "problem", "start_x", 0.0 );
   params.start_y = readEntry<double>( pt, "problem", "start_y", 0.0 );

   std::vector<t_matrix> U_natural = { data.U[0], data.u[0], data.u[1], data.u[2], data.U[4], data.U[5], data.U[6], data.p };
   t_matrix U_rho = data.U[0];
   t_matrix U_u   = data.u[0];
   t_matrix U_v   = data.u[1];
   t_matrix U_w   = data.u[2];
   t_matrix U_bx  = data.U[4];
   t_matrix U_by  = data.U[5];
   t_matrix U_bz  = data.U[6];
   t_matrix U_p   = data.p;

   double left[PRB_DIM];
   double right[PRB_DIM];
   double up[PRB_DIM];
   double down[PRB_DIM];

   left[0]  = readEntry<double>( pt, "problem", "rho_l", 0.1 );
   left[1]  = readEntry<double>( pt, "problem", "u_l",   0.0 );
   left[2]  = readEntry<double>( pt, "problem", "v_l",   0.0 );
   left[3]  = readEntry<double>( pt, "problem", "w_l",   0.0 );
   left[4]  = readEntry<double>( pt, "problem", "Bx_l",  0.0 );
   left[5]  = readEntry<double>( pt, "problem", "By_l",  0.0 );
   left[6]  = readEntry<double>( pt, "problem", "Bz_l",  0.0 );
   left[7]  = readEntry<double>( pt, "problem", "p_l",   0.1 );

   right[0] = readEntry<double>( pt, "problem", "rho_r", 1.0 );
   right[1] = readEntry<double>( pt, "problem", "u_r",   0.0 );
   right[2] = readEntry<double>( pt, "problem", "v_r",   0.0 );
   right[3] = readEntry<double>( pt, "problem", "w_r",   0.0 );
   right[4] = readEntry<double>( pt, "problem", "Bx_r",  0.0 );
   right[5] = readEntry<double>( pt, "problem", "By_r",  0.0 );
   right[6] = readEntry<double>( pt, "problem", "Bz_r",  0.0 );
   right[7] = readEntry<double>( pt, "problem", "p_r",   1.0 );

   up[0]    = readEntry<double>( pt, "problem", "rho_u", 0.1   );
   up[1]    = readEntry<double>( pt, "problem", "u_u",   0.0   );
   up[2]    = readEntry<double>( pt, "problem", "v_u",   0.0   );
   up[3]    = readEntry<double>( pt, "problem", "w_u",   0.0   );
   up[4]    = readEntry<double>( pt, "problem", "Bx_u", -0.5   );
   up[5]    = readEntry<double>( pt, "problem", "By_u",  0.0   );
   up[6]    = readEntry<double>( pt, "problem", "Bz_u",  0.0   );
   up[7]    = readEntry<double>( pt, "problem", "p_u",   0.875 );

   down[0]  = readEntry<double>( pt, "problem", "rho_d", 0.1   );
   down[1]  = readEntry<double>( pt, "problem", "u_d",   0.0   );
   down[2]  = readEntry<double>( pt, "problem", "v_d",   0.0   );
   down[3]  = readEntry<double>( pt, "problem", "w_d",   0.0   );
   down[4]  = readEntry<double>( pt, "problem", "Bx_d",  0.5   );
   down[5]  = readEntry<double>( pt, "problem", "By_d",  0.0   );
   down[6]  = readEntry<double>( pt, "problem", "Bz_d",  0.0   );
   down[7]  = readEntry<double>( pt, "problem", "p_d",   0.875 );

   double sheet_start_head = readEntry<double>( pt, "problem", "sheet start", params.start_x+Lx/2 );
   double sheet_start_foot = readEntry<double>( pt, "problem", "sheet start foot", sheet_start_head );
   double sheet_thickness  = readEntry<double>( pt, "problem", "sheet thickness", Ly/2.0 );

   int lobe_width = (Ly-sheet_thickness)/2.0/params.dy;
   int uplimit = NYLAST  - lobe_width;
   int dnlimit = NYFIRST + lobe_width;

   std::vector<int> sheet_start_index(NY);

   double ellip_a  = sheet_start_head - sheet_start_foot;
   double ellip_b  = 0.5*(sheet_thickness-params.dy);
   double ellip_b2 = ellip_b*ellip_b;
   double ellip_x, ellip_y;
   for( int j = dnlimit; j < uplimit; j++ ){
      ellip_y = (j-1.5) * params.dy - Ly/2;
      ellip_x = sheet_start_foot + ellip_a*sqrt( 1.0 - (ellip_y*ellip_y/ellip_b2) );
      sheet_start_index[j] = (-params.start_x+ellip_x)/params.dx + NXFIRST;
   }

   std::string sheet_profile = readEntry<std::string>( pt, "problem", "sheet profile", "constant" );

   if( sheet_profile == "constant" ){
      for( int i = NXFIRST; i < NXLAST; i++ ){
         for( int j = NYFIRST; j < dnlimit; j++ ){
            U_rho[i][j] = down[0];
            U_u  [i][j] = down[1];
            U_v  [i][j] = down[2];
            U_w  [i][j] = down[3];
            U_bx [i][j] = down[4];
            U_by [i][j] = down[5];
            U_bz [i][j] = down[6];
            U_p  [i][j] = down[7];
         }
         for( int j = dnlimit; j < uplimit; j++ ){
            if( i < sheet_start_index[j] ){
               U_rho[i][j] = left[0];
               U_u  [i][j] = left[1];
               U_v  [i][j] = left[2];
               U_w  [i][j] = left[3];
               U_bx [i][j] = left[4];
               U_by [i][j] = left[5];
               U_bz [i][j] = left[6];
               U_p  [i][j] = left[7];
            } else {
               U_rho[i][j] = right[0];
               U_u  [i][j] = right[1];
               U_v  [i][j] = right[2];
               U_w  [i][j] = right[3];
               U_bx [i][j] = right[4];
               U_by [i][j] = right[5];
               U_bz [i][j] = right[6];
               U_p  [i][j] = right[7];
            }
         }
         for( int j = uplimit; j < NYLAST; j++ ){
            U_rho[i][j] = up[0];
            U_u  [i][j] = up[1];
            U_v  [i][j] = up[2];
            U_w  [i][j] = up[3];
            U_bx [i][j] = up[4];
            U_by [i][j] = up[5];
            U_bz [i][j] = up[6];
            U_p  [i][j] = up[7];
         }
      }
   } else if( sheet_profile == "tanh" ){
      double left_bb   = left[4]*left[4] + left[5]*left[5] + left[6]*left[6];
      double right_bb   = right[4]*right[4] + right[5]*right[5] + right[6]*right[6];

      for( int i = NXFIRST; i < NXLAST; i++ ){
         for( int j = NYFIRST; j < dnlimit; j++ ){
            U_rho[i][j] = down[0];
            U_u  [i][j] = down[1];
            U_v  [i][j] = down[2];
            U_w  [i][j] = down[3];
            U_bx [i][j] = down[4];
            U_by [i][j] = down[5];
            U_bz [i][j] = down[6];
            U_p  [i][j] = down[7];
         }
         for( int j = dnlimit; j < uplimit; j++ ){
            // tanh(y) profile for B, with -1 <= y <= 1
            double coef_b = tanh( 2.0*(j-dnlimit)/((uplimit-1)-dnlimit) - 1.0 );

            if( i < sheet_start_index[j] ){
               U_rho[i][j] = left[0];
               U_u  [i][j] = left[1];
               U_v  [i][j] = left[2];
               U_w  [i][j] = left[3];
               U_bx [i][j] = left[4]*coef_b;
               U_by [i][j] = left[5]*coef_b;
               U_bz [i][j] = left[6]*coef_b;
               // calculate pressure so that total pressure is kept constant
               U_p[i][j] = left[7] + 0.5*left_bb*(1.0-coef_b*coef_b);
            } else {
               U_rho[i][j] = right[0];
               U_u  [i][j] = right[1];
               U_v  [i][j] = right[2];
               U_w  [i][j] = right[3];
               U_bx [i][j] = right[4]*coef_b;
               U_by [i][j] = right[5]*coef_b;
               U_bz [i][j] = right[6]*coef_b;
               // calculate pressure so that total pressure is kept constant
               U_p[i][j] = right[7] + 0.5*right_bb*(1.0-coef_b*coef_b);
            }
         }
         for( int j = uplimit; j < NYLAST; j++ ){
            U_rho[i][j] = up[0];
            U_u  [i][j] = up[1];
            U_v  [i][j] = up[2];
            U_w  [i][j] = up[3];
            U_bx [i][j] = up[4];
            U_by [i][j] = up[5];
            U_bz [i][j] = up[6];
            U_p  [i][j] = up[7];
         }
      }
   }

   std::string transition_type = readEntry<std::string>( pt, "problem", "transition type", "jump" );
   if( transition_type == "jump" ){
      // nothing to do
   } else {
      int transition_points = 2*readEntry<int>( pt, "problem", "transition points", 1 );

      // prepare transition coefficients
      std::vector<double> transition_coef( transition_points );
      if( transition_type == "linear" ){
         double diff = 1.0/transition_points;
         for( int i = 0; i < transition_points; i++ ){
            transition_coef[i] = (transition_points-i - 0.5)*diff;
         }
      } else {
         criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                      + "readProblemPlasmaSheet: in section [problem], key \"transition type\":\n"
                      + "Unknown transition type: " + tempstr );
      }

      // apply transitions to the sheet-lobe boundary
      int dn_btm = dnlimit - transition_points - 1; // start of transition around dnlimit
      int dn_top = dnlimit + transition_points;     // end of transition around dnlimit
      int up_btm = uplimit - transition_points - 1; // start of transition around uplimit
      int up_top = uplimit + transition_points;     // end of transition around uplimit
      auto mag_pressure = []( double bx, double by, double bz ){
         return 0.5*(bx*bx+by*by+bz*bz);
      };
      auto combine = []( double coef, double first, double second ){
         return coef*first + (1.0-coef)*second;
      };

      for( int i = NXFIRST; i < NXLAST; i++ ){
         double ptot_dn_btm = U_p[i][dn_btm] + mag_pressure( U_bx[i][dn_btm], U_by[i][dn_btm], U_bz[i][dn_btm] );
         double ptot_dn_top = U_p[i][dn_top] + mag_pressure( U_bx[i][dn_top], U_by[i][dn_top], U_bz[i][dn_top] );
         double ptot_up_btm = U_p[i][up_btm] + mag_pressure( U_bx[i][up_btm], U_by[i][up_btm], U_bz[i][up_btm] );
         double ptot_up_top = U_p[i][up_top] + mag_pressure( U_bx[i][up_top], U_by[i][up_top], U_bz[i][up_top] );

         // transition around dnlimit
         for( int p = 0; p < transition_points; p++ ){
            int j = dnlimit - (transition_points/2) + p;
            for( auto &v: U_natural ){
               v[i][j] = combine( transition_coef[p], v[i][dn_btm], v[i][dn_top] );
            }
            // fix pressure
            double ptot = combine( transition_coef[p], ptot_dn_btm, ptot_dn_top );
            U_p[i][j] = ptot - mag_pressure( U_bx[i][j], U_by[i][j], U_bz[i][j] );
         }

         // transition around uplimit
         for( int p = 0; p < transition_points; p++ ){
            int j = uplimit - (transition_points/2) + p;
            for( auto &v: U_natural ){
               v[i][j] = combine( transition_coef[p], v[i][up_btm], v[i][up_top] );
            }
            // fix pressure
            double ptot = combine( transition_coef[p], ptot_up_btm, ptot_up_top );
            U_p[i][j] = ptot - mag_pressure( U_bx[i][j], U_by[i][j], U_bz[i][j] );
         }
      }

      // apply transitions to the left sheet-right sheet boundary
      for( int p = 0; p < transition_points; p++ ){
         for( int j = dnlimit; j < uplimit; j++ ){
            int i = sheet_start_index[j] - (transition_points/2) + p;
            int i_left = sheet_start_index[j] - (transition_points/2) - 1;
            int i_rght = sheet_start_index[j] + (transition_points/2);

            for( auto &v: U_natural ){
               v[i][j] = combine( transition_coef[p], v[i_left][j], v[i_rght][j] );
            }
            // fix pressure
            double ptot_left = U_p[i_left][j] + mag_pressure( U_bx[i_left][j], U_by[i_left][j], U_bz[i_left][j] );
            double ptot_rght = U_p[i_rght][j] + mag_pressure( U_bx[i_rght][j], U_by[i_rght][j], U_bz[i_rght][j] );
            double ptot = combine( transition_coef[p], ptot_left, ptot_rght );
            U_p[i][j] = ptot - mag_pressure( U_bx[i][j], U_by[i][j], U_bz[i][j] );
         }
      }
   }

   toConservationData( params, data );

   std::vector<std::string> boundary_name(params.b_count);
   boundary_name[params.b_left  ] = "left";
   boundary_name[params.b_top   ] = "top";
   boundary_name[params.b_right ] = "right";
   boundary_name[params.b_bottom] = "bottom";
   for( int b = 0; b < params.b_count; b++ ){
      std::string boundary_key = "boundary "+boundary_name[b];
      tempstr = readEntry<std::string>( pt, "problem", boundary_key, "open" );

      params.boundary[b] = fromString<BoundaryCondition>( tempstr );
      switch( params.boundary[b] ){
      case BoundaryCondition::Undefined:
         criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                      + "readProblemPlasmaSheet: in section [problem], key \"" + boundary_key + "\":\n"
                      + "Unknown boundary condition: " + tempstr );
         break;
      case BoundaryCondition::Dirichlet:
         params.boundary_dirichlet_U[params.b_left  ] = createVectors( PRB_DIM, NY );
         params.boundary_dirichlet_U[params.b_top   ] = createVectors( PRB_DIM, NX );
         params.boundary_dirichlet_U[params.b_right ] = createVectors( PRB_DIM, NY );
         params.boundary_dirichlet_U[params.b_bottom] = createVectors( PRB_DIM, NX );
         for( int k = 0; k < PRB_DIM; k++ ){
            for( int j = NYFIRST; j < NYLAST; j++ ){
               params.boundary_dirichlet_U[params.b_left ][k][j] = data.U[k][NXFIRST ][j];
               params.boundary_dirichlet_U[params.b_right][k][j] = data.U[k][NXLAST-1][j];
            }
            for( int i = NXFIRST; i < NXLAST; i++ ){
               params.boundary_dirichlet_U[params.b_bottom][k][i] = data.U[k][i][NYFIRST ];
               params.boundary_dirichlet_U[params.b_top   ][k][i] = data.U[k][i][NYLAST-1];
            }
         }
         break;
      case BoundaryCondition::Periodic:
      case BoundaryCondition::Neumann:
      case BoundaryCondition::Open:
         break;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void readProblemWaveTest
   ( boost::property_tree::ptree &pt
   , t_params &params
   , t_data   &data
){
   // Domain size and position
   auto Lx = readEntry<double>( pt, "problem", "Lx", 1.0 ); params.dx = Lx/params.nx;
   auto Ly = readEntry<double>( pt, "problem", "Ly", 1.0 ); params.dy = Ly/params.ny;
   params.start_x = readEntry<double>( pt, "problem", "start_x", -0.5 );
   params.start_y = readEntry<double>( pt, "problem", "start_y", -0.5 );

   // give meaningful names
   t_matrix U_rho = data.U[0];
   t_matrix U_u   = data.u[0];
   t_matrix U_v   = data.u[1];
   t_matrix U_w   = data.u[2];
   t_matrix U_bx  = data.U[4];
   t_matrix U_by  = data.U[5];
   t_matrix U_bz  = data.U[6];
   t_matrix U_p   = data.p;
   auto U_natural = std::vector<t_matrix>{ U_rho, U_u, U_v, U_w, U_bx, U_by, U_bz, U_p };

   // read average values
   auto U_natural_avg = std::vector<double>{};
   U_natural_avg.resize( PRB_DIM );
   U_natural_avg[0] = readEntry<double>( pt, "problem", "rho",  1.0   );
   U_natural_avg[1] = readEntry<double>( pt, "problem", "u",    0.0   );
   U_natural_avg[2] = readEntry<double>( pt, "problem", "v",    0.0   );
   U_natural_avg[3] = readEntry<double>( pt, "problem", "w",    0.0   );
   U_natural_avg[4] = readEntry<double>( pt, "problem", "Bx",   0.0   );
   U_natural_avg[5] = readEntry<double>( pt, "problem", "By",   0.0   );
   U_natural_avg[6] = readEntry<double>( pt, "problem", "Bz",   0.0   );
   U_natural_avg[7] = readEntry<double>( pt, "problem", "p",    1.0   );

   // read problem options
   auto wavenumber_x = readEntry<int>( pt, "problem", "wavenumber x", 1 );
   auto wavenumber_y = readEntry<int>( pt, "problem", "wavenumber y", 0 );
   if( wavenumber_x < 0 || wavenumber_y < 0 ){
      criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                   + "readProblemWaveTest: wavenumber must be nonnegative.\n" );
   }
   auto epsilon = readEntry<double>( pt, "problem", "epsilon", 1.0e-8 );
   auto wave_type = readEntry<std::string>( pt, "problem", "wave", "entropy" );

   // get eigenvector for wave being tested
   double U_con_avg[PRB_DIM];
   for( size_t k = 0; k < PRB_DIM; k++ ){
      U_con_avg[k] = U_natural_avg[k];
   }
   double u[PRB_DIM];
   u[0] = U_natural_avg[1];
   u[1] = U_natural_avg[2];
   u[2] = U_natural_avg[3];
   auto p = U_natural_avg[7];
   toConservationPoint( params, U_con_avg, u, p );
   double eigenvalues[PRB_DIM];
   double eigenvectorsleft[PRB_DIM][PRB_DIM];
   double eigenvectorsright[PRB_DIM][PRB_DIM];
   auto retval = getEigens_F( U_con_avg, U_con_avg, params.gamma, eigenvalues, eigenvectorsleft, eigenvectorsright, true  );
   if( retval.isError ){
      criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                   + "readProblemWaveTest: failed to calculate eigensystem from average values.\n"
                   + retval.message );
   }
   double *eigenvector;
   if( wave_type == "negative fast magnetosonic" ){
      eigenvector = eigenvectorsright[0];
   } else if( wave_type == "negative alfven" ){
      eigenvector = eigenvectorsright[1];
   } else if( wave_type == "negative slow magnetosonic" ){
      eigenvector = eigenvectorsright[2];
   } else if( wave_type == "entropy" ){
      eigenvector = eigenvectorsright[3];
   } else if( wave_type == "positive slow magnetosonic" ){
      eigenvector = eigenvectorsright[4];
   } else if( wave_type == "positive alfven" ){
      eigenvector = eigenvectorsright[5];
   } else if( wave_type == "positive fast magnetosonic" ){
      eigenvector = eigenvectorsright[6];
   } else {
      criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                   + "readProblemWaveTest: in section [problem], key \"wave\":\n"
                   + "Unknown wave type: " + wave_type );
   }
   OUT << "chosen eigenvector: ( "
       << eigenvector[0] << ", "
       << eigenvector[1] << ", "
       << eigenvector[2] << ", "
       << eigenvector[3] << ", "
       << eigenvector[4] << ", "
       << eigenvector[5] << ", "
       << eigenvector[6] << ", "
       << eigenvector[7] << ")\n";

   auto x = double{};
   auto y = double{};
   auto scale_x = 2*M_PI / Lx;
   auto scale_y = 2*M_PI / Ly;
   for( auto k = size_t{0}; k < PRB_DIM; k++ ){
      for( auto i = NXFIRST; i < NXLAST; i++ ){
         for( auto j = NYFIRST; j < NYLAST; j++ ){
            x = params.start_x + (i-NXFIRST)*params.dx;
            y = params.start_y + (j-NYFIRST)*params.dy;
            data.U[k][i][j] = U_con_avg[k]
                            + epsilon*eigenvector[k]*std::cos( scale_x*wavenumber_x*x + scale_y*wavenumber_y*y );
         }
      }
   }
   toNaturalData( params, data );

   auto rot_angle = std::atan2( wavenumber_y, wavenumber_x );
   auto ct = std::cos( rot_angle );
   auto st = std::sin( rot_angle );
   for( auto i = NXFIRST; i < NXLAST; i++ ){
      for( auto j = NYFIRST; j < NYLAST; j++ ){
         auto rot_u  = ct*U_u[i][j]  - st*U_v[i][j];
         auto rot_v  = st*U_u[i][j]  + ct*U_v[i][j];
         auto rot_bx = ct*U_bx[i][j] - st*U_by[i][j];
         auto rot_by = st*U_bx[i][j] + ct*U_by[i][j];
         U_u[i][j]  = rot_u;
         U_v[i][j]  = rot_v;
         U_bx[i][j] = rot_bx;
         U_by[i][j] = rot_by;
      }
   }
   toConservationData( params, data );

   auto boundary_name = std::vector<std::string>( params.b_count );
   boundary_name[params.b_left  ] = "left";
   boundary_name[params.b_top   ] = "top";
   boundary_name[params.b_right ] = "right";
   boundary_name[params.b_bottom] = "bottom";
   for( int b = 0; b < params.b_count; b++ ){
      auto boundary_key = std::string{"boundary "} + boundary_name[b];
      auto boundary_type = readEntry<std::string>( pt, "problem", boundary_key, "open" );

      params.boundary[b] = fromString<BoundaryCondition>( boundary_type );
      switch( params.boundary[b] ){
      case BoundaryCondition::Undefined:
         criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                      + "readProblemWaveTest: in section [problem], key \"" + boundary_key + "\":\n"
                      + "Unknown boundary condition: " + boundary_type );
         break;
      case BoundaryCondition::Dirichlet:
         params.boundary_dirichlet_U[params.b_left  ] = createVectors( PRB_DIM, NY );
         params.boundary_dirichlet_U[params.b_top   ] = createVectors( PRB_DIM, NX );
         params.boundary_dirichlet_U[params.b_right ] = createVectors( PRB_DIM, NY );
         params.boundary_dirichlet_U[params.b_bottom] = createVectors( PRB_DIM, NX );
         for( int k = 0; k < PRB_DIM; k++ ){
            for( int j = NYFIRST; j < NYLAST; j++ ){
               params.boundary_dirichlet_U[params.b_left ][k][j] = data.U[k][NXFIRST ][j];
               params.boundary_dirichlet_U[params.b_right][k][j] = data.U[k][NXLAST-1][j];
            }
            for( int i = NXFIRST; i < NXLAST; i++ ){
               params.boundary_dirichlet_U[params.b_bottom][k][i] = data.U[k][i][NYFIRST ];
               params.boundary_dirichlet_U[params.b_top   ][k][i] = data.U[k][i][NYLAST-1];
            }
         }
         break;
      case BoundaryCondition::Periodic:
      case BoundaryCondition::Neumann:
      case BoundaryCondition::Open:
         break;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void outputBinaryHintFile
   ( const std::string filename
   , const t_params &params
   , const t_output &output_grid
){
   t_output bin_hint;
   bin_hint.binary   = false;
   bin_hint.filename = filename;
   openFile( bin_hint );

   // doubles: x, y, 5 common (rho,Bx,By,Bz,divb), 4 natural (u,v,w,p), 4 conservation (mx,my,mz,e)
   int         field_count      = 2 + 5 + (output_grid.natural?4:0) + (output_grid.conservation?4:0);
   int         field_size       = output_grid.binary_double ? sizeof(double) : sizeof(float);
   const char *binary_data_type = output_grid.binary_double ? "double" : "float";
   int i = 0;
   fprintf( bin_hint.file, "data fields: %d)x  %d)y  %d)rho", i+1, i+2, i+3 ); i+=3;
   if( output_grid.natural ){
      fprintf( bin_hint.file, "  %d)u  %d)v  %d)w  %d)p", i+1, i+2, i+3, i+4 ); i+=4;
   }
   fprintf( bin_hint.file, "  %d)Bx  %d)By  %d)Bz %d)divB", i+1, i+2, i+3, i+4 ); i+=4;
   if( output_grid.conservation ){
      fprintf( bin_hint.file, "  %d)mx  %d)my  %d)mz  %d)e", i+1, i+2, i+3, i+4 ); i+=4;
   }
   fprintf( bin_hint.file, "\n" );

   // characteristics: x, y, u, v, csx, csy, cax, cay, cfx, cfy,
   //                  LUx[u-cfx], LUy[v-cfy], LUx[u-cax], LUy[v-cay], LUx[u-csx], LUy[v-csy],
   //                  LUx[u], LUy[v],
   //                  LUx[u+csx], LUy[v+csy], LUx[u+cax], LUy[v+cay], LUx[u+cfx], LUy[v+cfy]
   //                  dummy, dummy
   int char_field_count = 2 + 2*(1+3) + 2*(3+1+3+1);
   fprintf( bin_hint.file, "characteristics data fields:"
      "\n"
      "  1)x"
      "  2)y"
      "\n"
      "  3)u"
      "  4)v"
      "  5)csx"
      "  6)csy"
      "  7)cax"
      "  8)cay"
      "  9)cfx"
      "  10)cfy"
      "\n"
      "  11)LUx[u-cfx]"
      "  12)LUy[v-cfy]"
      "  13)LUx[u-cax]"
      "  14)LUy[v-cay]"
      "  15)LUx[u-csx]"
      "  16)LUy[v-csy]"
      "\n"
      "  17)LUx[u]"
      "  18)LUy[v]"
      "\n"
      "  19)LUx[u+csx]"
      "  20)LUy[v+csy]"
      "  21)LUx[u+cax]"
      "  22)LUy[v+cay]"
      "  23)LUx[u+cfx]"
      "  24)LUy[v+cfy]"
      "\n"
      "  25)dummy"
      "  26)dummy"
   );
   fprintf( bin_hint.file, "\n\n" );

   fprintf( bin_hint.file, "nx = %d\nny = %d\n", params.nx, params.ny );
   fprintf( bin_hint.file, "xrange = [%.2f:%.2f]\nyrange = [%.2f:%.2f]\n",
                           params.start_x, params.start_x + params.nx*params.dx,
                           params.start_y, params.start_y + params.ny*params.dy );
   fprintf( bin_hint.file, "block_size = %d*%d\n", field_count, field_size );
   fprintf( bin_hint.file, "record = (nx,ny)\n" );
   fprintf( bin_hint.file, "WARNING: for versions of gnuplot < 5.0.5, record = (ny,nx)" );
   fprintf( bin_hint.file, "skip   = index*nx*ny*block_size\n" );
   fprintf( bin_hint.file, "format = \"%%%d%s\"\n", field_count, binary_data_type );

   fprintf( bin_hint.file, "\nexample gnuplot command:\n" );
   fprintf( bin_hint.file, "datafile = \"%s\"\n", output_grid.filename.c_str() );
   fprintf( bin_hint.file, "index = 0\n" );
   fprintf( bin_hint.file, "nx = %d\nny = %d\n", params.nx, params.ny );
   fprintf( bin_hint.file, "set xrange [%.2f:%.2f]\nset yrange [%.2f:%.2f]\n",
                           params.start_x, params.start_x + params.nx*params.dx,
                           params.start_y, params.start_y + params.ny*params.dy );
   fprintf( bin_hint.file, "block_size = %d*%d\n", field_count, field_size );
   fprintf( bin_hint.file, "plot datafile binary record=(nx,ny) skip=index*nx*ny*block_size"
                           " format=\"%%%d%s\" u 1:2:3 w image\n", field_count, binary_data_type );

   fprintf( bin_hint.file, "\nexample gnuplot command for characteristics:\n" );
   fprintf( bin_hint.file, "datafile = \"char-%s\"\n", output_grid.filename.c_str() );
   fprintf( bin_hint.file, "index = 0\n" );
   fprintf( bin_hint.file, "nx = %d\nny = %d\n", params.nx, params.ny );
   fprintf( bin_hint.file, "set xrange [%.2f:%.2f]\nset yrange [%.2f:%.2f]\n",
                           params.start_x, params.start_x + params.nx*params.dx,
                           params.start_y, params.start_y + params.ny*params.dy );
   fprintf( bin_hint.file, "block_size = %d*%d\n", char_field_count, field_size );
   fprintf( bin_hint.file, "plot datafile binary record=(nx,ny) skip=index*nx*ny*block_size"
                           " format=\"%%%d%s\" u 1:2:3 w image\n", char_field_count, binary_data_type );

   closeFile( bin_hint );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void outputGridDataBinary
   ( const t_output &output
   , const t_params &params
   , const t_data   &data
   , int    step
   , int    index
   , int    divb_nxfirst
   , int    divb_nxlast
   , int    divb_nyfirst
   , int    divb_nylast
   , double /*uumax*/ // unused
){
   // local aliases
   t_matrices U = data.U;
   t_matrices u = data.u;
   t_matrix   p = data.p;

   // fields: x, y, 5 common (rho,Bx,By,Bz,divb), 4 natural (u,v,w,p), 4 conservation (mx,my,mz,e)
   int field_count = 2 + 5 + (output.natural?4:0) + (output.conservation?4:0);

   // data records
   int      ri;
   size_t   record_count   = field_count*params.nx*params.ny;
   t_vector records_double = createVector( record_count );
   float   *records_float  = (float *)malloc( record_count*sizeof(*records_float) );

   // used in divb calculation
   const int kbx  = 4;
   const int kby  = 5;
   double    divb;

   t_output local_output = output;
   if( !local_output.single_file ){
      char buf[80];
      std::snprintf( buf, 80, local_output.format_step.c_str(), step );
      std::string str_step( buf );
      std::snprintf( buf, 80, local_output.format_index.c_str(), index );
      std::string str_index( buf );
      std::snprintf( buf, 80, local_output.format_simtime.c_str(), data.t_current );
      std::string str_time(buf);
      // Replace masks for output filenames
      boost::replace_all( local_output.filename, "%rs", str_step  );
      boost::replace_all( local_output.filename, "%ri", str_index );
      boost::replace_all( local_output.filename, "%rt", str_time  );
      openFile( local_output );
   }

   if( local_output.binary_double ){
      ri = 0;
      for( int i = NXFIRST; i < NXLAST; i++ ){
         for( int j = NYFIRST; j < NYLAST; j++ ){
            // calculate divb
            if( ( i >= divb_nxfirst && i < divb_nxlast ) &&
                ( j >= divb_nyfirst && j < divb_nylast ) ){
               divb = fabs( (U[kbx][i+1][j]-U[kbx][i-1][j])/(2.0*params.dx)
                          + (U[kby][i][j+1]-U[kby][i][j-1])/(2.0*params.dy) );
            } else {
               divb = 0.0;
            }

            // fill record
            records_double[ri++] = params.start_x + (i-NXFIRST)*params.dx;
            records_double[ri++] = params.start_y + (j-NYFIRST)*params.dy;
            records_double[ri++] = U[0][i][j];     // rho
            if( local_output.natural ){
               records_double[ri++] = u[0][i][j];  // u
               records_double[ri++] = u[1][i][j];  // v
               records_double[ri++] = u[2][i][j];  // w
               records_double[ri++] = p[i][j];     // p
            }
            records_double[ri++] = U[4][i][j];     // Bx
            records_double[ri++] = U[5][i][j];     // By
            records_double[ri++] = U[6][i][j];     // Bz
            records_double[ri++] = divb;           // div B
            if( local_output.conservation ){
               records_double[ri++] = U[1][i][j];  // mx
               records_double[ri++] = U[2][i][j];  // my
               records_double[ri++] = U[3][i][j];  // mz
               records_double[ri++] = U[7][i][j];  // e
            }
         }
      }

      // write all of the records to file at once
      fwrite( records_double, sizeof(*records_double), record_count, local_output.file );
   } else {
      ri = 0;
      for( int i = NXFIRST; i < NXLAST; i++ ){
         for( int j = NYFIRST; j < NYLAST; j++ ){
            // calculate divb
            if( ( i >= divb_nxfirst && i < divb_nxlast ) &&
                ( j >= divb_nyfirst && j < divb_nylast ) ){
               divb = fabs( (U[kbx][i+1][j]-U[kbx][i-1][j])/(2.0*params.dx)
                          + (U[kby][i][j+1]-U[kby][i][j-1])/(2.0*params.dy) );
            } else {
               divb = 0.0;
            }

            // fill record
            records_float[ri++] = params.start_x + (i-NXFIRST)*params.dx;
            records_float[ri++] = params.start_y + (j-NYFIRST)*params.dy;
            records_float[ri++] = U[0][i][j];     // rho
            if( local_output.natural ){
               records_float[ri++] = u[0][i][j];  // u
               records_float[ri++] = u[1][i][j];  // v
               records_float[ri++] = u[2][i][j];  // w
               records_float[ri++] = p[i][j];     // p
            }
            records_float[ri++] = U[4][i][j];     // Bx
            records_float[ri++] = U[5][i][j];     // By
            records_float[ri++] = U[6][i][j];     // Bz
            records_float[ri++] = divb;           // div B
            if( local_output.conservation ){
               records_float[ri++] = U[1][i][j];  // mx
               records_float[ri++] = U[2][i][j];  // my
               records_float[ri++] = U[3][i][j];  // mz
               records_float[ri++] = U[7][i][j];  // e
            }
         }
      }

      // write all of the records to file at once
      fwrite( records_float, sizeof(*records_float), record_count, local_output.file );
   }

   if( !local_output.single_file ){
      closeFile( local_output );
   }

   free( records_float );
   freeVector( records_double );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void outputCharacteristicsBinary
   ( const t_output    &output
   , const t_params    &params
   , const t_matrices  &cx
   , const t_matrices  &cy
   , const t_matrices  &LUx
   , const t_matrices  &LUy
   , double time
   , int    step
   , int    index
){
   // characteristic velocities (cx, cy): u-cfx, v-cfy, u-cax, v-cay, u-csx, v-csy, u, v, u+csx, v+csy, u+cax, v+cay, u+cfx, v+cfy, dummy, dummy
   // output fields: x, y, u, v, csx, csy, cax, cay, cfx, cfy,
   //                LUx[u-cfx], LUy[v-cfy], LUx[u-cax], LUy[v-cay], LUx[u-csx], LUy[v-csy],
   //                LUx[u], LUy[v],
   //                LUx[u+csx], LUy[v+csy], LUx[u+cax], LUy[v+cay], LUx[u+cfx], LUy[v+cfy]
   int field_count = 2 + 2*(1+3) + 2*(3+1+3+1);

   // data records
   int      ri;
   size_t   record_count   = field_count*params.nx*params.ny;
   t_vector records_double = createVector( record_count );
   float   *records_float  = (float *)malloc( record_count*sizeof(*records_float) );

   t_output local_output = output;
   char buf[80];
   std::snprintf( buf, 80, local_output.format_step.c_str(), step );
   std::string str_step( buf );
   std::snprintf( buf, 80, local_output.format_index.c_str(), index );
   std::string str_index( buf );
   std::snprintf( buf, 80, local_output.format_simtime.c_str(), time );
   std::string str_time(buf);
   // Replace masks for output filenames
   boost::replace_all( local_output.filename, "%rs", str_step  );
   boost::replace_all( local_output.filename, "%ri", str_index );
   boost::replace_all( local_output.filename, "%rt", str_time  );
   openFile( local_output );

   if( local_output.binary_double ){
      ri = 0;
      for( int i = NXFIRST; i < NXLAST; i++ ){
         for( int j = NYFIRST; j < NYLAST; j++ ){
            // fill record
            records_double[ri++] = params.start_x + (i-NXFIRST)*params.dx;
            records_double[ri++] = params.start_y + (j-NYFIRST)*params.dy;

            records_double[ri++] = cx[3][i][j];               // u
            records_double[ri++] = cy[3][i][j];               // v
            records_double[ri++] = cx[4][i][j] - cx[3][i][j]; // csx
            records_double[ri++] = cy[4][i][j] - cy[3][i][j]; // csy
            records_double[ri++] = cx[5][i][j] - cx[3][i][j]; // cax
            records_double[ri++] = cy[5][i][j] - cy[3][i][j]; // cay
            records_double[ri++] = cx[6][i][j] - cx[3][i][j]; // cfx
            records_double[ri++] = cy[6][i][j] - cy[3][i][j]; // cfy

            records_double[ri++] = LUx[0][i][j]; // LUx[u-cfx]
            records_double[ri++] = LUy[0][i][j]; // LUy[v-cfy]
            records_double[ri++] = LUx[1][i][j]; // LUx[u-cax]
            records_double[ri++] = LUy[1][i][j]; // LUy[v-cay]
            records_double[ri++] = LUx[2][i][j]; // LUx[u-csx]
            records_double[ri++] = LUy[2][i][j]; // LUy[v-csy]
            records_double[ri++] = LUx[3][i][j]; // LUx[u]
            records_double[ri++] = LUy[3][i][j]; // LUy[v]
            records_double[ri++] = LUx[4][i][j]; // LUx[u+csx]
            records_double[ri++] = LUy[4][i][j]; // LUy[v+csy]
            records_double[ri++] = LUx[5][i][j]; // LUx[u+cax]
            records_double[ri++] = LUy[5][i][j]; // LUy[v+cay]
            records_double[ri++] = LUx[6][i][j]; // LUx[u+cfx]
            records_double[ri++] = LUy[6][i][j]; // LUy[v+cfy]
            records_double[ri++] = LUx[7][i][j]; // ???
            records_double[ri++] = LUy[7][i][j]; // ???
         }
      }

      // write all of the records to file at once
      fwrite( records_double, sizeof(*records_double), record_count, local_output.file );
   } else {
      ri = 0;
      for( int i = NXFIRST; i < NXLAST; i++ ){
         for( int j = NYFIRST; j < NYLAST; j++ ){
            // fill record
            records_float[ri++] = params.start_x + (i-NXFIRST)*params.dx;
            records_float[ri++] = params.start_y + (j-NYFIRST)*params.dy;

            records_float[ri++] = cx[3][i][j];               // u
            records_float[ri++] = cy[3][i][j];               // v
            records_float[ri++] = cx[4][i][j] - cx[3][i][j]; // csx
            records_float[ri++] = cy[4][i][j] - cy[3][i][j]; // csy
            records_float[ri++] = cx[5][i][j] - cx[3][i][j]; // cax
            records_float[ri++] = cy[5][i][j] - cy[3][i][j]; // cay
            records_float[ri++] = cx[6][i][j] - cx[3][i][j]; // cfx
            records_float[ri++] = cy[6][i][j] - cy[3][i][j]; // cfy

            records_float[ri++] = LUx[0][i][j]; // LUx[u-cfx]
            records_float[ri++] = LUy[0][i][j]; // LUy[v-cfy]
            records_float[ri++] = LUx[1][i][j]; // LUx[u-cax]
            records_float[ri++] = LUy[1][i][j]; // LUy[v-cay]
            records_float[ri++] = LUx[2][i][j]; // LUx[u-csx]
            records_float[ri++] = LUy[2][i][j]; // LUy[v-csy]
            records_float[ri++] = LUx[3][i][j]; // LUx[u]
            records_float[ri++] = LUy[3][i][j]; // LUy[v]
            records_float[ri++] = LUx[4][i][j]; // LUx[u+csx]
            records_float[ri++] = LUy[4][i][j]; // LUy[v+csy]
            records_float[ri++] = LUx[5][i][j]; // LUx[u+cax]
            records_float[ri++] = LUy[5][i][j]; // LUy[v+cay]
            records_float[ri++] = LUx[6][i][j]; // LUx[u+cfx]
            records_float[ri++] = LUy[6][i][j]; // LUy[v+cfy]
            records_float[ri++] = LUx[7][i][j]; // ???
            records_float[ri++] = LUy[7][i][j]; // ???
         }
      }

      // write all of the records to file at once
      fwrite( records_float, sizeof(*records_float), record_count, local_output.file );
   }

   closeFile( local_output );

   free( records_float );
   freeVector( records_double );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void outputGridDataText
   ( const t_output &output
   , const t_params &params
   , const t_data   &data
   , int    step
   , int    index
   , int    divb_nxfirst
   , int    divb_nxlast
   , int    divb_nyfirst
   , int    divb_nylast
   , double uumax
){
   // local aliases
   t_matrices U = data.U;
   t_matrices u = data.u;
   t_matrix   p = data.p;

   // header
   fprintf( output.file, "# record #%d, step #%d, t = %.5f, u_max = %.2f\n", index, step, data.t_current, sqrt(uumax) );
   int i = 0;
   fprintf( output.file, "# %d)i  %d)j  %d)x  %d)y  %d)rho", i+1, i+2, i+3, i+4, i+5 ); i+=5;
   if( output.natural ){
      fprintf( output.file, "  %d)u  %d)v  %d)w  %d)p", i+1, i+2, i+3, i+4 ); i+=4;
   }
   fprintf( output.file, "  %d)Bx  %d)By  %d)Bz %d)divB", i+1, i+2, i+3, i+4 ); i+=4;
   if( output.conservation ){
      fprintf( output.file, "  %d)mx  %d)my  %d)mz  %d)e", i+1, i+2, i+3, i+4 ); i+=4;
   }
   fprintf( output.file, "\n" );

   // used in divb calculation
   const int kbx = 4, kby = 5;
   double divb;

   // data
   for( int i = NXFIRST; i < NXLAST; i++ ){
      for( int j = NYFIRST; j < NYLAST; j++ ){
         if( ( i >= divb_nxfirst && i < divb_nxlast ) &&
             ( j >= divb_nyfirst && j < divb_nylast ) ){
            divb = fabs( (U[kbx][i+1][j]-U[kbx][i-1][j])/(2.0*params.dx)
                       + (U[kby][i][j+1]-U[kby][i][j-1])/(2.0*params.dy) );
         } else {
            divb = 0.0;
         }
         fprintf( output.file, "%d\t%d\t%+.5e\t%+.5e\t%+.5e",
                  i-NXFIRST, j-NYFIRST,
                  params.start_x + (i-NXFIRST)*params.dx, // x
                  params.start_y + (j-NYFIRST)*params.dy, // y
                  U[0][i][j]      // rho
         );
         if( output.natural )
            fprintf( output.file, "\t%+.5e\t%+.5e\t%+.5e\t%+.5e",
                     u[0][i][j],  // u
                     u[1][i][j],  // v
                     u[2][i][j],  // w
                     p[i][j]      // p
            );

         fprintf( output.file, "\t%+.5e\t%+.5e\t%+.5e\t%+.5e",
                  U[4][i][j],  // Bx
                  U[5][i][j],  // By
                  U[6][i][j],  // Bz
                  divb         // div B
         );
         if( output.conservation )
            fprintf( output.file, "\t%+.5e\t%+.5e\t%+.5e\t%+.5e",
                     U[1][i][j],  // mx
                     U[2][i][j],  // my
                     U[3][i][j],  // mz
                     U[7][i][j]   // e
            );
         fprintf( output.file, "\n" );
      }
      fprintf( output.file, "\n" );
   }
   fprintf( output.file, "\n\n" );
   fflush( output.file );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void outputGridData
   ( const t_output &output
   , const t_params &params
   , const t_data   &data
   , int step
   , int index
){
   // local aliases
   t_matrices U = data.U;
   t_matrices u = data.u;

   // apply boundary conditions, required for div B calculation
   const int kbx = 4, kby = 5;
   int divb_nxfirst = NXFIRST;
   int divb_nxlast  = NXLAST;
   int divb_nyfirst = NYFIRST;
   int divb_nylast  = NYLAST;

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
      break;
   case BoundaryCondition::Neumann:
      divb_nxfirst = NXFIRST+1;
      break;
   case BoundaryCondition::Open:
      for( int k = kbx; k <= kby; k++ ){
         for( int j = NYFIRST; j < NYLAST; j++ ){
            U[k][NXFIRST-2][j] = U[k][NXFIRST][j];
            U[k][NXFIRST-1][j] = U[k][NXFIRST][j];
         }
      }
      divb_nxfirst = NXFIRST+1;
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
      break;
   case BoundaryCondition::Neumann:
      divb_nxlast  = NXLAST-1;
      break;
   case BoundaryCondition::Open:
      for( int k = kbx; k <= kby; k++ ){
         for( int j = NYFIRST; j < NYLAST; j++ ){
            U[k][NXLAST][j]    = U[k][NXLAST-1][j];
            U[k][NXLAST+1][j]  = U[k][NXLAST-1][j];
         }
      }
      divb_nxlast  = NXLAST-1;
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
      break;
   case BoundaryCondition::Neumann:
      divb_nyfirst = NYFIRST+1;
      break;
   case BoundaryCondition::Open:
      for( int k = kbx; k <= kby; k++ ){
         for( int i = NXFIRST; i < NXLAST; i++ ){
            U[k][i][NYFIRST-2] = U[k][i][NYFIRST];
            U[k][i][NYFIRST-1] = U[k][i][NYFIRST];
         }
      }
      divb_nyfirst = NYFIRST+1;
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
      break;
   case BoundaryCondition::Neumann:
      divb_nylast  = NYLAST-1;
      break;
   case BoundaryCondition::Open:
      for( int k = kbx; k <= kby; k++ ){
         for( int i = NXFIRST; i < NXLAST; i++ ){
            U[k][i][NYLAST]    = U[k][i][NYLAST-1];
            U[k][i][NYLAST+1]  = U[k][i][NYLAST-1];
         }
      }
      divb_nylast  = NYLAST-1;
      break;
   }

   // find maximum velocity
   double uu, uumax = 0.0;
   for( int i = NXFIRST; i < NXLAST; i++ ){
      for( int j = NYFIRST; j < NYLAST; j++ ){
         uu = u[0][i][j]*u[0][i][j]
            + u[1][i][j]*u[1][i][j]
            + u[2][i][j]*u[2][i][j];
         if( uu > uumax ) uumax = uu;
      }
   }

   // binary or text output
   if( output.binary ){
      outputGridDataBinary( output, params, data, step, index, divb_nxfirst, divb_nxlast, divb_nyfirst, divb_nylast, uumax );
   } else {
      outputGridDataText( output, params, data, step, index, divb_nxfirst, divb_nxlast, divb_nyfirst, divb_nylast, uumax );
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void outputNonGridData
   ( const t_output &output
   , const t_params &params
   , const t_data   &data
   , int step
   , int index
){
   static bool initialized = false;
   t_matrices U = data.U;
   t_matrices u = data.u;
   t_matrix   p = data.p;

   // find maximum velocity and totals for energy, entropy, mass and momentum
   double totenergy = 0.0;
   double entropy, totentropy = 0.0;
   double totmass = 0.0;
   double totmomentum = 0.0;
   double uu, uumax = 0.0;
   for( int i = NXFIRST; i < NXLAST; i++ ){
      for( int j = NYFIRST; j < NYLAST; j++ ){
         uu = u[0][i][j]*u[0][i][j]
            + u[1][i][j]*u[1][i][j]
            + u[2][i][j]*u[2][i][j];
         if( uu > uumax ) uumax = uu;

         entropy = log( p[i][j]*pow( U[0][i][j], -params.gamma ) );
         totentropy  += entropy;
         totenergy   += U[7][i][j];
         totmass     += U[0][i][j];
         totmomentum += U[0][i][j]*sqrt(uu);
      }
   }
   // convert sums into integrals
   totentropy  *= params.dx*params.dy;
   totenergy   *= params.dx*params.dy;
   totmass     *= params.dx*params.dy;
   totmomentum *= params.dx*params.dy;

   // find max |divb|, integral |divb|
   double maxdivb = 0.0, totdivb = 0.0;
   divBCalculation( U, params, maxdivb, totdivb );

   // output header only once
   if( !initialized ){
      fprintf( output.file, "# 1)record_index 2)step 3)t 4)dt 5)u_max 6)|divB|_max 7)sum|divB| 8)total_mass 9)total_momentum 10)total_energy 11)total_entropy\n" );
      initialized = true;
   }

   // data
   fprintf( output.file, "%d\t%d\t%.5e\t%.5e\t%.2e\t%.2e\t%.2e\t%.8e\t%.8e\t%.8e\t%.8e\n",
            index, step, data.t_current, data.dt,
            sqrt(uumax), maxdivb, totdivb,
            totmass, totmomentum, totenergy, totentropy );

   // force write
   fflush( output.file );
}
