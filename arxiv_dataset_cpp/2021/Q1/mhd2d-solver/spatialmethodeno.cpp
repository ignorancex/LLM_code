#include "spatialmethodeno.hpp"

#include <stdexcept>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
SpatialMethodEno::SpatialMethodEno
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
      throw std::domain_error( "Buffer is too small for ENO;"
                               " need at least 2 extra grids" );
   }

   F = createMatrices( PRB_DIM, nxTotal, nyTotal );
   G = createMatrices( PRB_DIM, nxTotal, nyTotal );

   _cx = createMatrices( PRB_DIM, nxTotal, nyTotal );
   _cy = createMatrices( PRB_DIM, nxTotal, nyTotal );
   _LUx = createMatrices( PRB_DIM, nxTotal, nyTotal );
   _LUy = createMatrices( PRB_DIM, nxTotal, nyTotal );
   for( auto k = 0; k < PRB_DIM; k++ ){
      for( auto i = size_t{0}; i < nxTotal; i++ ){
         for( auto j = size_t{0}; j < nyTotal; j++ ){
            _cx[k][i][j] = 0.0;
            _cy[k][i][j] = 0.0;
            _LUx[k][i][j] = 0.0;
            _LUy[k][i][j] = 0.0;
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
SpatialMethodEno::~SpatialMethodEno
   (
){
   freeMatrices( F );
   freeMatrices( G );

   freeMatrices( _cx );
   freeMatrices( _cy );
   freeMatrices( _LUx );
   freeMatrices( _LUy );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool SpatialMethodEno::getCharacteristicsX
   ( t_matrices cx
   , t_matrices LUx
){
   for( auto k = 0; k < PRB_DIM; k++ ){
      for( auto i = size_t{0}; i < nxTotal; i++ ){
         for( auto j = size_t{0}; j < nyTotal; j++ ){
            cx[k][i][j]    = _cx[k][i][j];
            LUx[k][i][j]   = _LUx[k][i][j];
         }
      }
   }

   return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool SpatialMethodEno::getCharacteristicsY
   ( t_matrices cy
   , t_matrices LUy
){
   for( auto k = 0; k < PRB_DIM; k++ ){
      for( auto i = size_t{0}; i < nxTotal; i++ ){
         for( auto j = size_t{0}; j < nyTotal; j++ ){
            cy[k][i][j]    = _cy[k][i][j];
            LUy[k][i][j]   = _LUy[k][i][j];
         }
      }
   }

   return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SpatialMethodEno::updateFluxes
   ( const t_matrices U
){
   for( auto i = size_t{0}; i < nxTotal; i++ ){
      for( auto j = size_t{0}; j < nyTotal; j++ ){
         auto r  = U[0][i][j];
         auto mx = U[1][i][j]; auto u = mx / r;
         auto my = U[2][i][j]; auto v = my / r;
         auto mz = U[3][i][j]; auto w = mz / r;
         auto bx = U[4][i][j];
         auto by = U[5][i][j];
         auto bz = U[6][i][j];
         auto e  = U[7][i][j];
         auto uu =  u*u  +  v*v  +  w*w;
         auto ub =  u*bx +  v*by +  w*bz;
         auto bb = bx*bx + by*by + bz*bz;
         auto p = (gamma-1.0)*(e-0.5*r*uu-0.5*bb);
         auto ptot = p + 0.5*bb;

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
      }
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status SpatialMethodEno::getEigenvaluesAt
   ( const double U[PRB_DIM]
   , double       lambda[PRB_DIM]
){
   auto status = t_status{ false, ReturnStatus::OK, std::string{} };

   // rename values
   auto r  = U[0];
   auto mx = U[1]; auto u  = U[1]/U[0];
   auto my = U[2];
   auto mz = U[3];
   auto bx = U[4];
   auto by = U[5];
   auto bz = U[6];
   auto e  = U[7];

   if( r <= 0.0 ){
      return { true, ReturnStatus::ErrorNegativeDensity
             , "negative density encountered\n! getEigenvaluesAt" };
   }

   // vector inner products
   auto mm = mx*mx + my*my + mz*mz;
   auto bb = bx*bx + by*by + bz*bz;

   // pressure
   auto p = (gamma-1.0)*( e - 0.5*mm/r - 0.5*bb );
   if( p < 0.0 ){
      if( breakOnNegativePressure ){
         ERROUT << "ERROR: getEigenvalues: Negative pressure encountered!\n";
         ERROUT << "       Details:\n"
                << "         * rho = " << r << "\n"
                << "         * u = " << mx/r << ", v = " << my/r << ", w = " << mz/r << "\n"
                << "         * bx = " << bx << ", by = " << by << ", bz = " << bz << "\n"
                << "         * e = "  << e << ", bb = " << bb << ", uu = " << mm/r << "\n"
                << "         * p = " << p << "\n";

         return { true, ReturnStatus::ErrorNegativePressure
                , "negative pressure encountered\n! getEigenvaluesAt" };
      } else {
         p = 0.0;

         status.status = ReturnStatus::ErrorNegativePressure;
         status.message = "negative pressure encountered\n! getEigenvaluesAt";
         if( !shownPressureWarning ){
            ERROUT << "WARNING: getEigenvaluesAt: Negative pressure encountered!\n"
                   << "         Results from this point on are suspect.\n"
                   << "         Simulation resumed with pressure forced to zero.\n";
         }
      }
   }

   // speeds
   auto a2  = gamma*p / r;
   if( a2 < 0.0 ) a2 = 0.0;
   //auto a   = sqrt(a2);    // sound speed
   auto ca2 = bx*bx/r;
   auto ca  = sqrt(ca2);   // Alfven speed
   auto c2  = 0.5*(bb/r+a2);
   auto ctemp = c2*c2-a2*ca2; if( ctemp < 0.0 ) ctemp = 0.0;
   auto cs2 = c2 - sqrt(ctemp); if( cs2 < 0.0 ) cs2 = EPS;
   auto cs  = sqrt(cs2);   // slow magnetosonic speed
   auto cf2 = c2 + sqrt(ctemp); if( cf2 < 0.0 ) cf2 = EPS;
   auto cf  = sqrt(cf2);   // fast magnetosonic speed

   /* eigenvalues */
   lambda[0] = u - cf;
   lambda[1] = u - ca;
   lambda[2] = u - cs;
   lambda[3] = u;
   lambda[4] = u + cs;
   lambda[5] = u + ca;
   lambda[6] = u + cf;
   lambda[7] = 0;

   return status;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status SpatialMethodEno::getEigenvaluesBetween
   ( const double U1[PRB_DIM]
   , const double U2[PRB_DIM]
   , double       lambda[PRB_DIM])
{
   auto status = t_status{ false, ReturnStatus::OK, std::string{} };

   // half-point values
   auto r  = 0.5 * ( U1[0] + U2[0] );
   auto u  = 0.5 * ( U1[1]/U1[0] + U2[1]/U2[0] );
   //auto v  = 0.5 * ( U1[2]/U1[0] + U2[2]/U2[0] );
   //auto w  = 0.5 * ( U1[3]/U1[0] + U2[3]/U2[0] );
   auto bx = 0.5 * ( U1[4] + U2[4] );
   auto by = 0.5 * ( U1[5] + U2[5] );
   auto bz = 0.5 * ( U1[6] + U2[6] );

   if( r <= 0.0 ){
      return { true, ReturnStatus::ErrorNegativeDensity
             , "negative density encountered\n! getEigenvaluesBetween" };
   }

   // vector inner products
   //auto uu =  u*u   + v*v   + w*w;
   //auto ub =  u*bx  + v*by  + w*bz;
   auto bb = bx*bx + by*by + bz*bz;

   // half-point pressure
   auto mml = U1[1]*U1[1] + U1[2]*U1[2] + U1[3]*U1[3];
   auto bbl = U1[4]*U1[4] + U1[5]*U1[5] + U1[6]*U1[6];
   auto mmr = U2[1]*U2[1] + U2[2]*U2[2] + U2[3]*U2[3];
   auto bbr = U2[4]*U2[4] + U2[5]*U2[5] + U2[6]*U2[6];
   auto ptotl = (gamma-1.0)*( U1[7] - 0.5*mml/U1[0] - 0.5*bbl ) + 0.5*bbl;
   auto ptotr = (gamma-1.0)*( U2[7] - 0.5*mmr/U2[0] - 0.5*bbr ) + 0.5*bbr;
   auto ptot  = 0.5 * ( ptotl + ptotr );
   auto p = ptot - 0.5*bb;
   if( p < 0.0 ){
      if( breakOnNegativePressure ){
         return { true, ReturnStatus::ErrorNegativePressure
                , "negative pressure encountered\n! getEigenvaluesBetween" };
      } else {
         p = 0.0;

         status.status = ReturnStatus::ErrorNegativePressure;
         status.message = "negative pressure encountered\n! getEigenvaluesBetween";
         if( !shownPressureWarning ){
            ERROUT << "WARNING: getEigenvaluesBetween: Negative pressure encountered!\n"
                   << "         Results from this point on are suspect.\n"
                   << "         Simulation resumed with pressure forced to zero.\n";
         }
      }
   }

   // speeds
   auto a2  = gamma*p / r;
   if( a2 < 0.0 ) a2 = 0.0;
   //auto a   = sqrt(a2);    // sound speed
   auto ca2 = bx*bx/r;
   auto ca  = sqrt(ca2);   // Alfven speed
   auto c2  = 0.5*(bb/r+a2);
   auto ctemp = c2*c2-a2*ca2; if( ctemp < 0.0 ) ctemp = 0.0;
   auto cs2 = c2 - sqrt(ctemp); if( cs2 < 0.0 ) cs2 = EPS;
   auto cs  = sqrt(cs2);   // slow magnetosonic speed
   auto cf2 = c2 + sqrt(ctemp); if( cf2 < 0.0 ) cf2 = EPS;
   auto cf  = sqrt(cf2);   // fast magnetosonic speed

   /* eigenvalues */
   lambda[0] = u - cf;
   lambda[1] = u - ca;
   lambda[2] = u - cs;
   lambda[3] = u;
   lambda[4] = u + cs;
   lambda[5] = u + ca;
   lambda[6] = u + cf;
   lambda[7] = 0;

   return status;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status SpatialMethodEno::getEigensF
   ( double U1[PRB_DIM]
   , double U2[PRB_DIM]
   , double lambda[PRB_DIM]
   , double lv[PRB_DIM][PRB_DIM]
   , double rv[PRB_DIM][PRB_DIM]
){
   auto status = t_status{ false, ReturnStatus::OK, std::string{} };

   // half-point values
   auto r  = 0.5 * ( U1[0] + U2[0] );
   auto u  = 0.5 * ( U1[1]/U1[0] + U2[1]/U2[0] );
   auto v  = 0.5 * ( U1[2]/U1[0] + U2[2]/U2[0] );
   auto w  = 0.5 * ( U1[3]/U1[0] + U2[3]/U2[0] );
   auto bx = 0.5 * ( U1[4] + U2[4] );
   auto by = 0.5 * ( U1[5] + U2[5] );
   auto bz = 0.5 * ( U1[6] + U2[6] );
   //auto e  = 0.5 * ( U1[7] + U2[7] );

   if( r <= 0.0 ){
      return { true, ReturnStatus::ErrorNegativeDensity
             , "negative density encountered\n! getEigens_F" };
   }

   // vector inner products
   auto uu =  u*u   + v*v   + w*w;
   auto bb = bx*bx + by*by + bz*bz;

   // half-point pressure
   auto mml = U1[1]*U1[1] + U1[2]*U1[2] + U1[3]*U1[3];
   auto bbl = U1[4]*U1[4] + U1[5]*U1[5] + U1[6]*U1[6];
   auto mmr = U2[1]*U2[1] + U2[2]*U2[2] + U2[3]*U2[3];
   auto bbr = U2[4]*U2[4] + U2[5]*U2[5] + U2[6]*U2[6];
   auto ptotl = (gamma-1.0)*( U1[7] - 0.5*mml/U1[0] - 0.5*bbl ) + 0.5*bbl;
   auto ptotr = (gamma-1.0)*( U2[7] - 0.5*mmr/U2[0] - 0.5*bbr ) + 0.5*bbr;
   auto ptot  = 0.5 * ( ptotl + ptotr );
   auto p = ptot - 0.5*bb;
   if( p < 0.0 ){
      if( breakOnNegativePressure ){
         return { true, ReturnStatus::ErrorNegativePressure
                , "negative pressure encountered\n! getEigens_F" };
      } else {
         p = 0.0;

         status.status = ReturnStatus::ErrorNegativePressure;
         status.message = "negative pressure encountered\n! getEigenvaluesBetween";
         if( !shownPressureWarning ){
            ERROUT << "WARNING: getEigens_F: Negative pressure encountered!\n"
                   << "         Results from this point on are suspect.\n"
                   << "         Simulation resumed with pressure forced to zero.\n";
         }
      }
   }

   // speeds
   auto a2  = gamma*p / r;
   if( a2 < 0.0 ) a2 = 0.0;
   auto a   = sqrt(a2);    // sound speed
   auto ca2 = bx*bx/r;
   auto ca  = sqrt(ca2);   // Alfven speed
   auto c2  = 0.5*(bb/r+a2);
   auto ctemp = c2*c2-a2*ca2; if( ctemp < 0.0 ) ctemp = 0.0;
   auto cs2 = c2 - sqrt(ctemp); if( cs2 < 0.0 ) cs2 = EPS;
   auto cs  = sqrt(cs2);   // slow magnetosonic speed
   auto cf2 = c2 + sqrt(ctemp); if( cf2 < 0.0 ) cf2 = EPS;
   auto cf  = sqrt(cf2);   // fast magnetosonic speed

#ifdef DEBUG_MIDPOINT_EIGENVALUES
   OUT << "*** DEBUG: get_eigens_f: r, ca, cs, cf = "
       << r << " " << ca << " " << cs << " " << cf << LF;
#endif // DEBUG_MIDPOINT_EIGENVALUES

   // other
   auto rroot   = sqrt( r );
   auto bperp2  = by*by + bz*bz;
   auto sgnbx   = (bx<0.0) ? -1.0 : 1.0;//( (bx>0.0) ? 1 : 0 );
   auto alphaf2 = ( (bperp2<EPS) && EPS_EQUAL(ca2,a2) ) ? 1 : (cf2-ca2) / (cf2-cs2);
        alphaf2 = alphaf2 < 0.0 ? 0.0 : alphaf2; // hack for cf ~ ca
   auto alphaf  = sqrt( alphaf2 );
   auto alphas2 = ( (bperp2<EPS) && EPS_EQUAL(ca2,a2) ) ? 1 : (cf2- a2) / (cf2-cs2);
        alphas2 = alphas2 < 0.0 ? 0.0 : alphas2; // hack for cf ~ a
   auto alphas  = sqrt( alphas2 );
   auto betay   = bperp2 > EPS ? by / sqrt( bperp2 ) : 1.0/sqrt(2.0);
   auto betaz   = bperp2 > EPS ? bz / sqrt( bperp2 ) : 1.0/sqrt(2.0);
   auto theta1  = 1.0 / ( alphaf2*a2*(cf2-(gamma-2.0)/(gamma-1.0)*a2) + alphas2*cf2*(cs2-(gamma-2.0)/(gamma-1.0)*a2) );
   auto theta2  = 1.0 / ( alphaf2*cf*a*sgnbx + alphas2*cs*ca*sgnbx );

#ifdef DEBUG_MIDPOINT_EIGENVALUES
   OUT << "*** DEBUG:  get_eigens_f: cf2-a2, cf2-cs2, alphas, alphas2, betay, th1, th2 = "
       << cf2-a2 << " " << cf2-cs2 << " " << alphas << " " << alphas2
       << " " << betay << " " << theta1 << " " << theta2 << LF;
#endif // DEBUG_MIDPOINT_EIGENVALUES


   /* eigenvalues */
   lambda[0] = u - cf;
   lambda[1] = u - ca;
   lambda[2] = u - cs;
   lambda[3] = u;
   lambda[4] = u + cs;
   lambda[5] = u + ca;
   lambda[6] = u + cf;
   lambda[7] = 0;

   /* right eigenvectors */
   // a1 = u - cf (fast magnetosonic)
   rv[0][0] =   alphaf;
   rv[0][1] =   alphaf * (u-cf);
   rv[0][2] =   alphaf*v + alphas*betay*ca*sgnbx;
   rv[0][3] =   alphaf*w + alphas*betaz*ca*sgnbx;
   rv[0][4] = 0;
   rv[0][5] =   alphas*betay*cf / rroot;
   rv[0][6] =   alphas*betaz*cf / rroot;
   rv[0][7] = 0.5*alphaf*uu + alphaf*cf2/(gamma-1.0) - alphaf*cf*u + alphas*ca*sgnbx*(betay*v+betaz*w) + (gamma-2.0)/(gamma-1.0)*alphaf*(cf2-a2);

   // a2 = u - ca (Alfven)
   rv[1][0] = 0;
   rv[1][1] = 0;
   rv[1][2] =   betaz*sgnbx;
   rv[1][3] = - betay*sgnbx;
   rv[1][4] = 0;
   rv[1][5] =   betaz/rroot;
   rv[1][6] = - betay/rroot;
   rv[1][7] =   (betaz*v-betay*w)*sgnbx;

   // a3 = u - cs (slow magnetosonic)
   rv[2][0] =   alphas;
   rv[2][1] =   alphas * (u-cs);
   rv[2][2] =   alphas*v - alphaf*betay*a*sgnbx;
   rv[2][3] =   alphas*w - alphaf*betaz*a*sgnbx;
   rv[2][4] = 0;
   rv[2][5] = - alphaf*betay*a2 / (cf*rroot);
   rv[2][6] = - alphaf*betaz*a2 / (cf*rroot);
   rv[2][7] = 0.5*alphas*uu + alphas*cs2/(gamma-1.0) - alphas*cs*u - alphaf*a*sgnbx*(betay*v+betaz*w) + (gamma-2.0)/(gamma-1.0)*alphas*(cs2-a2);

   // a4 = u (entropy)
   rv[3][0] = 1;
   rv[3][1] = u;
   rv[3][2] = v;
   rv[3][3] = w;
   rv[3][4] = 0;
   rv[3][5] = 0;
   rv[3][6] = 0;
   rv[3][7] = 0.5*uu;

   // a5 = u + cs (slow magnetosonic)
   rv[4][0] =   alphas;
   rv[4][1] =   alphas * (u+cs);
   rv[4][2] =   alphas*v + alphaf*betay*a*sgnbx;
   rv[4][3] =   alphas*w + alphaf*betaz*a*sgnbx;
   rv[4][4] = 0;
   rv[4][5] = - alphaf*betay*a2 / (cf*rroot);
   rv[4][6] = - alphaf*betaz*a2 / (cf*rroot);
   rv[4][7] = 0.5*alphas*uu + alphas*cs2/(gamma-1.0) + alphas*cs*u + alphaf*a*sgnbx*(betay*v+betaz*w) + (gamma-2.0)/(gamma-1.0)*alphas*(cs2-a2);

   // a6 = u + ca (Alfven)
   rv[5][0] = 0;
   rv[5][1] = 0;
   rv[5][2] = - betaz*sgnbx;
   rv[5][3] = + betay*sgnbx;
   rv[5][4] = 0;
   rv[5][5] =   betaz/rroot;
   rv[5][6] = - betay/rroot;
   rv[5][7] = - (betaz*v-betay*w)*sgnbx;

   // a7 = u + cf (fast magnetosonic)
   rv[6][0] =   alphaf;
   rv[6][1] =   alphaf * (u+cf);
   rv[6][2] =   alphaf*v - alphas*betay*ca*sgnbx;
   rv[6][3] =   alphaf*w - alphas*betaz*ca*sgnbx;
   rv[6][4] = 0;
   rv[6][5] =   alphas*betay*cf / rroot;
   rv[6][6] =   alphas*betaz*cf / rroot;
   rv[6][7] = 0.5*alphaf*uu + alphaf*cf2/(gamma-1.0) + alphaf*cf*u - alphas*ca*sgnbx*(betay*v+betaz*w) + (gamma-2.0)/(gamma-1.0)*alphaf*(cf2-a2);

   // a8 = 0 (dummy)
   rv[7][0] = 0;
   rv[7][1] = 0;
   rv[7][2] = 0;
   rv[7][3] = 0;
   rv[7][4] = 0;
   rv[7][5] = 0;
   rv[7][6] = 0;
   rv[7][7] = 0;

   /* left eigenvectors */
   // a1 = u - cf (fast magnetosonic)
   lv[0][0] = 0.25*theta1*alphaf*a2*uu + 0.5*theta2*( alphaf*a*u*sgnbx - alphas*cs*(betay*v+betaz*w) );
   lv[0][1] = -0.5*theta1*alphaf*a2*u - 0.5*theta2*alphaf*a*sgnbx;
   lv[0][2] = -0.5*theta1*alphaf*a2*v + 0.5*theta2*alphas*betay*cs;
   lv[0][3] = -0.5*theta1*alphaf*a2*w + 0.5*theta2*alphas*betaz*cs;
   lv[0][4] = 0;
   lv[0][5] =  0.5*theta1*alphas*betay*cf*rroot*(cs2-(gamma-2.0)/(gamma-1.0)*a2);
   lv[0][6] =  0.5*theta1*alphas*betaz*cf*rroot*(cs2-(gamma-2.0)/(gamma-1.0)*a2);
   lv[0][7] =  0.5*theta1*alphaf*a2;

   // a2 = u - ca (Alfven)
   lv[1][0] = -0.5*(betaz*v-betay*w)*sgnbx;
   lv[1][1] = 0;
   lv[1][2] =  0.5*betaz*sgnbx;
   lv[1][3] = -0.5*betay*sgnbx;
   lv[1][4] = 0;
   lv[1][5] =  0.5*betaz*rroot;
   lv[1][6] = -0.5*betay*rroot;
   lv[1][7] = 0;

   // a3 = u - cs (slow magnetosonic)
   lv[2][0] = 0.25*theta1*alphas*cf2*uu + 0.5*theta2*( alphas*ca*u*sgnbx + alphaf*cf*(betay*v+betaz*w) );
   lv[2][1] = -0.5*theta1*alphas*cf2*u - 0.5*theta2*alphas*ca*sgnbx;
   lv[2][2] = -0.5*theta1*alphas*cf2*v - 0.5*theta2*alphaf*betay*cf;
   lv[2][3] = -0.5*theta1*alphas*cf2*w - 0.5*theta2*alphaf*betaz*cf;
   lv[2][4] = 0;
   lv[2][5] = -0.5*theta1*alphaf*betay*cf*rroot*(cf2-(gamma-2.0)/(gamma-1.0)*a2);
   lv[2][6] = -0.5*theta1*alphaf*betaz*cf*rroot*(cf2-(gamma-2.0)/(gamma-1.0)*a2);
   lv[2][7] =  0.5*theta1*alphas*cf2;

   // a4 = u (entropy)
   lv[3][0] = 1.0 - 0.5*theta1*uu*(alphaf2*a2+alphas2*cf2);
   lv[3][1] =  theta1*(alphaf2*a2+alphas2*cf2)*u;
   lv[3][2] =  theta1*(alphaf2*a2+alphas2*cf2)*v;
   lv[3][3] =  theta1*(alphaf2*a2+alphas2*cf2)*w;
   lv[3][4] = 0;
   lv[3][5] =  theta1*alphaf*alphas*betay*cf*rroot*(cf2-cs2);
   lv[3][6] =  theta1*alphaf*alphas*betaz*cf*rroot*(cf2-cs2);
   lv[3][7] = -theta1*(alphaf2*a2+alphas2*cf2);

   // a5 = u + cs
   lv[4][0] = 0.25*theta1*alphas*cf2*uu - 0.5*theta2*( alphas*ca*u*sgnbx + alphaf*cf*(betay*v+betaz*w) );
   lv[4][1] = -0.5*theta1*alphas*cf2*u + 0.5*theta2*alphas*ca*sgnbx;
   lv[4][2] = -0.5*theta1*alphas*cf2*v + 0.5*theta2*alphaf*betay*cf;
   lv[4][3] = -0.5*theta1*alphas*cf2*w + 0.5*theta2*alphaf*betaz*cf;
   lv[4][4] = 0;
   lv[4][5] = -0.5*theta1*alphaf*betay*cf*rroot*(cf2-(gamma-2.0)/(gamma-1.0)*a2);
   lv[4][6] = -0.5*theta1*alphaf*betaz*cf*rroot*(cf2-(gamma-2.0)/(gamma-1.0)*a2);
   lv[4][7] =  0.5*theta1*alphas*cf2;

   // a6 = u + ca
   lv[5][0] = +0.5*(betaz*v-betay*w)*sgnbx;
   lv[5][1] = 0;
   lv[5][2] = -0.5*betaz*sgnbx;
   lv[5][3] =  0.5*betay*sgnbx;
   lv[5][4] = 0;
   lv[5][5] =  0.5*betaz*rroot;
   lv[5][6] = -0.5*betay*rroot;
   lv[5][7] = 0;

   // a7 = u + cf
   lv[6][0] = 0.25*theta1*alphaf*a2*uu - 0.5*theta2*( alphaf*a*u*sgnbx - alphas*cs*(betay*v+betaz*w) );
   lv[6][1] = -0.5*theta1*alphaf*a2*u + 0.5*theta2*alphaf*a*sgnbx;
   lv[6][2] = -0.5*theta1*alphaf*a2*v - 0.5*theta2*alphas*betay*cs;
   lv[6][3] = -0.5*theta1*alphaf*a2*w - 0.5*theta2*alphas*betaz*cs;
   lv[6][4] = 0;
   lv[6][5] =  0.5*theta1*alphas*betay*cf*rroot*(cs2-(gamma-2.0)/(gamma-1.0)*a2);
   lv[6][6] =  0.5*theta1*alphas*betaz*cf*rroot*(cs2-(gamma-2.0)/(gamma-1.0)*a2);
   lv[6][7] =  0.5*theta1*alphaf*a2;

   // a8 = 0 (dummy)
   lv[7][0] = 0;
   lv[7][1] = 0;
   lv[7][2] = 0;
   lv[7][3] = 0;
   lv[7][4] = 0;
   lv[7][5] = 0;
   lv[7][6] = 0;
   lv[7][7] = 0;

   /* make vectors continuous */
   auto sgnbt   = ( (by>0.0) || ( EPS_ZERO(by) && (bz>0.0) ) ) ? 1.0 : -1.0;
   auto csfactor = a2 > ca2 ? sgnbt : 1.0;
   auto cffactor = a2 < ca2 ? sgnbt : 1.0;

   for( auto k = size_t{0}; k < PRB_DIM; k++ ){
      rv[0][k] *= cffactor;
      rv[6][k] *= cffactor;
      lv[0][k] *= cffactor;
      lv[6][k] *= cffactor;
      rv[2][k] *= csfactor;
      rv[4][k] *= csfactor;
      lv[2][k] *= csfactor;
      lv[4][k] *= csfactor;
   }

   return status;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status SpatialMethodEno::getEigensG
   ( double U1[PRB_DIM]
   , double U2[PRB_DIM]
   , double lambda[PRB_DIM]
   , double lv[PRB_DIM][PRB_DIM]
   , double rv[PRB_DIM][PRB_DIM]
){
   // invert x and y axis
   std::swap( U1[1], U1[2] ); std::swap( U2[1], U2[2] );
   std::swap( U1[4], U1[5] ); std::swap( U2[4], U2[5] );

   // get eigens for inverted x and y
   auto retval = getEigensF( U1, U2, lambda, lv, rv );
   if( retval.status != ReturnStatus::OK ){
      retval.message += "\n! getEigens_G";
   }

   // return eigenvectors to proper axes
   for( auto k = size_t{0}; k < PRB_DIM; k++ ){
      std::swap( rv[k][1], rv[k][2] ); std::swap( lv[k][1], lv[k][2] );
      std::swap( rv[k][4], rv[k][5] ); std::swap( lv[k][4], lv[k][5] );
   }

   return retval;
}
