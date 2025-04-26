#include "spatialmethodcentralfd2.hpp"

#include <stdexcept>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
SpatialMethodCentralFD2::SpatialMethodCentralFD2
   ( size_t       nx
   , size_t       ny
   , size_t       bufferWidth
   , double       dx
   , double       dy
   , t_boundary   boundary
   , double       gamma
)
   : SpatialIntegrationMethod{ nx, ny, bufferWidth, dx, dy, boundary, gamma }
{
   if( bufferWidth < minimumBufferWidth ){
      throw std::domain_error( "Buffer is too small for second order"
                               "central finite difference; need at least 1 extra grid" );
   }

   F = createMatrices( PRB_DIM, nxTotal, nyTotal );
   G = createMatrices( PRB_DIM, nxTotal, nyTotal );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
SpatialMethodCentralFD2::~SpatialMethodCentralFD2
   (
){
   freeMatrices( F );
   freeMatrices( G );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
size_t SpatialMethodCentralFD2::requiredBufferWidth
   (
){
   return minimumBufferWidth;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status SpatialMethodCentralFD2::integrate
   ( t_matrices      U
   , t_matrices      UL
   , borderVectors   /*borderFlux*/
   , double         &dtIdeal
){
   auto pressureOK = bool{true};
   auto pressureI  = size_t{0};
   auto pressureJ  = size_t{0};

   double cxmax = 0.0;
   double cymax = 0.0; // maximum wave velocities

   applyBoundaryConditions( U );

   // flux F(U), G(U)
   for( auto i = nxFirst-1; i < nxLast+1; i++ ){
      for( auto j = nyFirst-1; j < nyLast+1; j++ ){
         // skip corners
         if( i < nxFirst && j < nxFirst ) continue;
         if( i < nxFirst && j >= nxLast ) continue;
         if( i >= nxLast && j < nxFirst ) continue;
         if( i >= nxLast && j >= nxLast ) continue;

         double r  = U[0][i][j];
         double mx = U[1][i][j]; double u = mx / r;
         double my = U[2][i][j]; double v = my / r;
         double mz = U[3][i][j]; double w = mz / r;
         double bx = U[4][i][j];
         double by = U[5][i][j];
         double bz = U[6][i][j];
         double e  = U[7][i][j];
         double uu =  u*u   + v*v   + w*w;
         double ub =  u*bx  + v*by  + w*bz;
         double bb = bx*bx + by*by + bz*bz;
         double p = (gamma-1.0)*(e-0.5*r*uu-0.5*bb);
         double ptot = p + 0.5*bb;

         if( r <= 0.0 ){
            return { true, ReturnStatus::ErrorNegativeDensity
                   , std::string{} + "Negative density encountered at "
                   +   "i = " + std::to_string(int(i)-int(bufferWidth))
                   + ", j = " + std::to_string(int(j)-int(bufferWidth))
                   + "\n! SpatialMethodCentralFD2::integrate" };
         }

         if( p <= 0.0 ){
            pressureOK = false;
            pressureI = i;
            pressureJ = j;
         }

         /* rho */ F[0][i][j] = mx;
         /* mx  */ F[1][i][j] = mx*u - bx*bx + ptot;
         /* my  */ F[2][i][j] = my*u - bx*by;
         /* mz  */ F[3][i][j] = mz*u - bx*bz;
         /* bx  */ F[4][i][j] = 0;
         /* by  */ F[5][i][j] = by*u - bx*v;
         /* bz  */ F[6][i][j] = bz*u - bx*w;
         /* e   */ F[7][i][j] = (e+ptot)*u - bx*ub;

         /* rho */ G[0][i][j] = my;
         /* mx  */ G[1][i][j] = mx*v - by*bx;
         /* my  */ G[2][i][j] = my*v - by*by + ptot;
         /* mz  */ G[3][i][j] = mz*v - by*bz;
         /* bx  */ G[4][i][j] = bx*v - by*u;
         /* by  */ G[5][i][j] = 0;
         /* bz  */ G[6][i][j] = bz*v - by*w;
         /* e   */ G[7][i][j] = (e+ptot)*v - by*ub;

         // speeds
         double c2  = gamma*p / r; if( c2 < 0.0 ) c2 = 0.0;
         //double c   = sqrt(c2);    // sound speed

         // x direction
         double ca2 = bx*bx/r;
         double ca  = sqrt(ca2);   // Alfven speed
         double ct2  = 0.5*(bb/r+c2);
         double ctemp = ct2*ct2-c2*ca2; if( ctemp < 0.0 ) ctemp = 0.0;
         double cs2 = ct2 - sqrt(ctemp); if( cs2 < 0.0 ) cs2 = 0.0;
         double cs  = sqrt(cs2);   // slow magnetosonic speed
         double cf2 = ct2 + sqrt(ctemp); if( cf2 < 0.0 ) cf2 = 0.0;
         double cf  = sqrt(cf2);   // fast magnetosonic speed
         cxmax = std::max({ cxmax, u - cf, u - ca, u - cs, u, u + cs, u + ca, u + cf });

         // y direction
         ca2 = by*by/r;
         ca  = sqrt(ca2);   // Alfven speed
         ct2  = 0.5*(bb/r+c2);
         ctemp = ct2*ct2-c2*ca2; if( ctemp < 0.0 ) ctemp = 0.0;
         cs2 = ct2 - sqrt(ctemp); if( cs2 < 0.0 ) cs2 = 0.0;
         cs  = sqrt(cs2);   // slow magnetosonic speed
         cf2 = ct2 + sqrt(ctemp); if( cf2 < 0.0 ) cf2 = 0.0;
         cf  = sqrt(cf2);   // fast magnetosonic speed
         cymax = std::max({ cymax, v - cf, v - ca, v - cs, v, v + cs, v + ca, v + cf });
      }
   }

   // update dt from max wave speeds
   dtIdeal = std::min( dx/cxmax, dy/cymax );

   // use numerical fluxes to calculate dU/dt
   for( auto k = size_t{0}; k < PRB_DIM; k++ ){
      for( auto i = nxFirst; i < nxLast; i++ ){
         for( auto j = nyFirst; j < nyLast; j++ ){
            UL[k][i][j] = -(1.0/(2.0*dx)) * ( F[k][i+1][j] - F[k][i-1][j] )
                          -(1.0/(2.0*dy)) * ( G[k][i][j+1] - G[k][i][j-1] );
         }
      }
   }

   if( !pressureOK ){
      return { true, ReturnStatus::ErrorNegativePressure
             , std::string{} + "Negative pressure encountered at "
             +   "i = " + std::to_string(int(pressureI)-int(bufferWidth))
             + ", j = " + std::to_string(int(pressureJ)-int(bufferWidth))
             + "\n! SpatialMethodCentralFD2::integrate" };
   } else {
      return { false, ReturnStatus::OK, "" };
   }
}
