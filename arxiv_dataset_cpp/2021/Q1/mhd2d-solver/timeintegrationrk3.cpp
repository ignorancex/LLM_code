#include "timeintegrationrk3.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TimeIntegrationRK3::TimeIntegrationRK3
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
   U1 = createMatrices( PRB_DIM, nxTotal, nyTotal );
   U2 = createMatrices( PRB_DIM, nxTotal, nyTotal );
   UL = createMatrices( PRB_DIM, nxTotal, nyTotal );

   borderFlux1.left  = createVectors( PRB_DIM, nyTotal );
   borderFlux1.right = createVectors( PRB_DIM, nyTotal );
   borderFlux1.up    = createVectors( PRB_DIM, nxTotal );
   borderFlux1.down  = createVectors( PRB_DIM, nxTotal );
   borderFlux2.left  = createVectors( PRB_DIM, nyTotal );
   borderFlux2.right = createVectors( PRB_DIM, nyTotal );
   borderFlux2.up    = createVectors( PRB_DIM, nxTotal );
   borderFlux2.down  = createVectors( PRB_DIM, nxTotal );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TimeIntegrationRK3::~TimeIntegrationRK3
   (
){
   freeMatrices( U1 );
   freeMatrices( U2 );
   freeMatrices( UL );

   freeVectors( borderFlux1.left  );
   freeVectors( borderFlux1.right );
   freeVectors( borderFlux1.up    );
   freeVectors( borderFlux1.down  );
   freeVectors( borderFlux2.left  );
   freeVectors( borderFlux2.right );
   freeVectors( borderFlux2.up    );
   freeVectors( borderFlux2.down  );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status TimeIntegrationRK3::step
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

   auto dtFirstStep  = dtCurrent;
   auto dtSecondStep = dtMax;
   auto dtFinalStep  = dtMax;

   auto done = bool{false};

   // First step: spatial integration, check for errors
   retval = method->integrate( U, UL, borderFlux, dtIdeal );
   if( retval.isError ){
      retval.message += "\n! TimeIntegrationRK3::step: (first) spatial integration";
      return retval;
   }

   // First step: update time step, check for errors
   retval = updateDt( dtFirstStep, dtIdeal );
   if( retval.isError ){
      retval.message += "\n! TimeIntegrationRK3::step: (first) update time step";
      return retval;
   }

   while( !done ){
      dt = std::min({ dtFirstStep, dtSecondStep, dtFinalStep });

      // First step: update variables
      for( auto k = size_t{0}; k < PRB_DIM; k++ )
         for( auto i = nxFirst; i < nxLast; i++ )
            for( auto j = nyFirst; j < nyLast; j++ )
               U1[k][i][j] = U[k][i][j] + dt*UL[k][i][j];

      // Second step: spatial integration, check for errors
      retval = method->integrate( U1, UL, borderFlux1, dtIdeal );
      if( retval.isError ){
         retval.message += "\n! TimeIntegrationRK3::step: (second) spatial integration";
         return retval;
      }

      // Second step: update time step, check for errors
      retval = updateDt( dtSecondStep, dtIdeal );
      if( retval.isError ){
         retval.message += "\n! TimeIntegrationRK3::step: (second) update time step";
         return retval;
      }
      if( dtSecondStep < dt ){
         continue;
      }

      // Second step: update variables
      for( auto k = size_t{0}; k < PRB_DIM; k++ )
         for( auto i = nxFirst; i < nxLast; i++ )
            for( auto j = nyFirst; j < nyLast; j++ )
               U2[k][i][j] = (3.0/4.0)*U[k][i][j] + (1.0/4.0)*U1[k][i][j] + (1.0/4.0)*dt*UL[k][i][j];

      // Final step: spatial integration, check for errors
      retval = method->integrate( U2, UL, borderFlux2, dtIdeal );
      if( retval.isError ){
         retval.message += "\n! TimeIntegrationRK3::step: (last) spatial integration";
         return retval;
      }

      // Final step: update time step, check for errors
      retval = updateDt( dtFinalStep, dtIdeal );
      if( retval.isError ){
         retval.message += "\n! TimeIntegrationRK3::step: (last) update time step";
         return retval;
      }
      if( dtFinalStep < dt ){
         continue;
      }

      // Final step: update variables
      for( auto k = size_t{0}; k < PRB_DIM; k++ )
         for( auto i = nxFirst; i < nxLast; i++ )
            for( auto j = nyFirst; j < nyLast; j++ )
               U[k][i][j] = (1.0/3.0)*U[k][i][j] + (2.0/3.0)*U2[k][i][j] + (2.0/3.0)*dt*UL[k][i][j];

      for( auto k = size_t{0}; k < PRB_DIM; k++ ){
         for( auto i = nxFirst; i < nxLast; i++ ){
            borderFlux.up[k][i]   = (1.0/6.0)*borderFlux.up[k][i]   + (1.0/6.0)*borderFlux1.up[k][i]   + (2.0/3.0)*borderFlux2.up[k][i];
            borderFlux.down[k][i] = (1.0/6.0)*borderFlux.down[k][i] + (1.0/6.0)*borderFlux1.down[k][i] + (2.0/3.0)*borderFlux2.down[k][i];
         }
         for( auto j = nyFirst; j < nyLast; j++ ){
            borderFlux.left[k][j]  = (1.0/6.0)*borderFlux.left[k][j]  + (1.0/6.0)*borderFlux1.left[k][j]  + (2.0/3.0)*borderFlux2.left[k][j];
            borderFlux.right[k][j] = (1.0/6.0)*borderFlux.right[k][j] + (1.0/6.0)*borderFlux1.right[k][j] + (2.0/3.0)*borderFlux2.right[k][j];
         }
      }

      done = true;
   }

   dtCurrent = dt;
   method->getCharacteristicsX( cx, LUx );
   method->getCharacteristicsY( cy, LUy );

   // Everything OK
   return { false, ReturnStatus::OK, "" };
}
