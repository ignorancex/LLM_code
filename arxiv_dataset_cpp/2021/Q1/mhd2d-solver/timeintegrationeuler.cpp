#include "timeintegrationeuler.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TimeIntegrationEuler::TimeIntegrationEuler
   ( size_t nx
   , size_t ny
   , size_t bufferWidth
   , double dtMin
   , double dtMax
   , double cflNumber
   , std::unique_ptr<SpatialIntegrationMethod> method
)
   : TimeIntegrationMethod{ nx, ny, bufferWidth, dtMin, dtMax, cflNumber, std::move( method ) }
{
   UL = createMatrices( PRB_DIM, nxTotal, nyTotal );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TimeIntegrationEuler::~TimeIntegrationEuler
   (
){
   freeMatrices( UL );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status TimeIntegrationEuler::step
   ( t_matrices      U
   , t_matrices      cx
   , t_matrices      cy
   , t_matrices      LUx
   , t_matrices      LUy
   , borderVectors   borderFlux
   , double         &dtCurrent
){
   auto dt        = dtCurrent;
   auto dtIdeal   = double{0.0};
   auto retval    = t_status{};

   // Spatial integration, check for errors
   retval = method->integrate( U, UL, borderFlux, dtIdeal );
   if( retval.isError ){
      retval.message += "\n! TimeIntegrationEuler::step: spatial integration";
      return retval;
   }

   // Update time step, check for errors
   retval = updateDt( dt, dtIdeal );
   if( retval.isError ){
      retval.message += "\n! TimeIntegrationEuler::step: update time step";
      return retval;
   }

   // Update return variables
   for( auto k = size_t{0}; k < PRB_DIM; k++ )
      for( auto i = nxFirst; i < nxLast; i++ )
         for( auto j = nyFirst; j < nyLast; j++ )
            U[k][i][j] = U[k][i][j] + dt*UL[k][i][j];
   dtCurrent = dt;
   method->getCharacteristicsX( cx, LUx );
   method->getCharacteristicsY( cy, LUy );

   // Everything OK
   return { false, ReturnStatus::OK, "" };
}
