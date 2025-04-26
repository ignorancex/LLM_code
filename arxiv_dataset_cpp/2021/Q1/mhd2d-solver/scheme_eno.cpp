#include "mhd2d.hpp"
#include "scheme_eno.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status getEigenvalues
   ( double U[PRB_DIM]
   , double gamma
   , double lambda[PRB_DIM]
   , bool   break_on_neg_pressure
){
   // rename values
   double r  = U[0];
   double mx = U[1]; double u  = U[1]/U[0];
   double my = U[2];
   double mz = U[3];
   double bx = U[4];
   double by = U[5];
   double bz = U[6];
   double e  = U[7];

   if( r <= 0.0 ){
      ERROUT << "ERROR: getEigenvalues: Negative density encountered!\n"
             << "       The simulation will now terminate." << LF;
      return { true, ReturnStatus::ErrorNegativeDensity, "negative density encountered\n! getEigenvalues" };
   }

   // vector inner products
   double mm = mx*mx + my*my + mz*mz;
   double bb = bx*bx + by*by + bz*bz;

   // pressure
   double p = (gamma-1.0)*( e - 0.5*mm/r - 0.5*bb );
   if( p < 0.0 ){
      if( break_on_neg_pressure ){
         ERROUT << "ERROR: getEigenvalues: Negative pressure encountered!" << LF;
         ERROUT << "       Details:" << LF
                << "         * rho = " << r << LF
                << "         * u = " << mx/r << ", v = " << my/r << ", w = " << mz/r << LF
                << "         * bx = " << bx << ", by = " << by << ", bz = " << bz << LF
                << "         * e = "  << e << ", bb = " << bb << ", uu = " << mm/r << LF
                << "         * p = " << p << LF;

         return { true, ReturnStatus::ErrorNegativePressure, "negative pressure encountered\n! getEigenvalues" };
      } else {
         ERROUT << "WARNING: getEigenvalues: Negative pressure encountered!\n"
                << "         Results from this point on are suspect.\n"
                << "         Simulation resumed with pressure forced to zero." << LF;
         p = 0.0;
      }
   }

   // speeds
   double a2  = gamma*p / r;
   if( a2 < 0.0 ) a2 = 0.0;
   //double a   = sqrt(a2);    // sound speed
   double ca2 = bx*bx/r;
   double ca  = sqrt(ca2);   // Alfven speed
   double c2  = 0.5*(bb/r+a2);
   double ctemp = c2*c2-a2*ca2; if( ctemp < 0.0 ) ctemp = 0.0;
   double cs2 = c2 - sqrt(ctemp); if( cs2 < 0.0 ) cs2 = EPS;
   double cs  = sqrt(cs2);   // slow magnetosonic speed
   double cf2 = c2 + sqrt(ctemp); if( cf2 < 0.0 ) cf2 = EPS;
   double cf  = sqrt(cf2);   // fast magnetosonic speed

   /* eigenvalues */
   lambda[0] = u - cf;
   lambda[1] = u - ca;
   lambda[2] = u - cs;
   lambda[3] = u;
   lambda[4] = u + cs;
   lambda[5] = u + ca;
   lambda[6] = u + cf;
   lambda[7] = 0;

   return { false, ReturnStatus::OK, "" };
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status getEigenvalues
   ( double U1[PRB_DIM]
   , double U2[PRB_DIM]
   , double gamma
   , double lambda[PRB_DIM]
   , bool   break_on_neg_pressure
){
   // half-point values
   double r  = 0.5 * ( U1[0] + U2[0] );
   double u  = 0.5 * ( U1[1]/U1[0] + U2[1]/U2[0] );
   //double v  = 0.5 * ( U1[2]/U1[0] + U2[2]/U2[0] );
   //double w  = 0.5 * ( U1[3]/U1[0] + U2[3]/U2[0] );
   double bx = 0.5 * ( U1[4] + U2[4] );
   double by = 0.5 * ( U1[5] + U2[5] );
   double bz = 0.5 * ( U1[6] + U2[6] );

   if( r <= 0.0 ){
      ERROUT << "ERROR: getEigenvalues: Negative density encountered!\n"
             << "       The simulation will now terminate." << LF;
      return { true, ReturnStatus::ErrorNegativeDensity, "negative density encountered\n! getEigenvalues" };
   }

   // vector inner products
   //double uu =  u*u   + v*v   + w*w;
   //double ub =  u*bx  + v*by  + w*bz;
   double bb = bx*bx + by*by + bz*bz;

   // half-point pressure
   double mml = U1[1]*U1[1] + U1[2]*U1[2] + U1[3]*U1[3];
   double bbl = U1[4]*U1[4] + U1[5]*U1[5] + U1[6]*U1[6];
   double mmr = U2[1]*U2[1] + U2[2]*U2[2] + U2[3]*U2[3];
   double bbr = U2[4]*U2[4] + U2[5]*U2[5] + U2[6]*U2[6];
   double ptotl = (gamma-1.0)*( U1[7] - 0.5*mml/U1[0] - 0.5*bbl ) + 0.5*bbl;
   double ptotr = (gamma-1.0)*( U2[7] - 0.5*mmr/U2[0] - 0.5*bbr ) + 0.5*bbr;
   double ptot  = 0.5 * ( ptotl + ptotr );
   double p = ptot - 0.5*bb;
   if( p < 0.0 ){
      if( break_on_neg_pressure ){
         ERROUT << "ERROR: getEigenvalues: Negative pressure encountered!" << LF;
         return { true, ReturnStatus::ErrorNegativePressure, "negative pressure encountered\n! getEigenvalues" };
      } else {
         ERROUT << "WARNING: getEigenvalues: Negative pressure encountered!\n"
                << "         Results from this point on are suspect.\n"
                << "         Simulation resumed with pressure forced to zero." << LF;
         p = 0.0;
      }
   }

   // speeds
   double a2  = gamma*p / r;
   if( a2 < 0.0 ) a2 = 0.0;
   //double a   = sqrt(a2);    // sound speed
   double ca2 = bx*bx/r;
   double ca  = sqrt(ca2);   // Alfven speed
   double c2  = 0.5*(bb/r+a2);
   double ctemp = c2*c2-a2*ca2; if( ctemp < 0.0 ) ctemp = 0.0;
   double cs2 = c2 - sqrt(ctemp); if( cs2 < 0.0 ) cs2 = EPS;
   double cs  = sqrt(cs2);   // slow magnetosonic speed
   double cf2 = c2 + sqrt(ctemp); if( cf2 < 0.0 ) cf2 = EPS;
   double cf  = sqrt(cf2);   // fast magnetosonic speed

   /* eigenvalues */
   lambda[0] = u - cf;
   lambda[1] = u - ca;
   lambda[2] = u - cs;
   lambda[3] = u;
   lambda[4] = u + cs;
   lambda[5] = u + ca;
   lambda[6] = u + cf;
   lambda[7] = 0;

   return { false, ReturnStatus::OK, "" };
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status getEigens_F
   ( double U1[PRB_DIM]
   , double U2[PRB_DIM]
   , double gamma
   , double lambda[PRB_DIM]
   , double lv[PRB_DIM][PRB_DIM]
   , double rv[PRB_DIM][PRB_DIM]
   , bool   break_on_neg_pressure
){
   // half-point values
   double r  = 0.5 * ( U1[0] + U2[0] );
   double u  = 0.5 * ( U1[1]/U1[0] + U2[1]/U2[0] );
   double v  = 0.5 * ( U1[2]/U1[0] + U2[2]/U2[0] );
   double w  = 0.5 * ( U1[3]/U1[0] + U2[3]/U2[0] );
   double bx = 0.5 * ( U1[4] + U2[4] );
   double by = 0.5 * ( U1[5] + U2[5] );
   double bz = 0.5 * ( U1[6] + U2[6] );
   //double e  = 0.5 * ( U1[7] + U2[7] );

   if( r <= 0.0 ){
      ERROUT << "ERROR: getEigens_F: Negative density encountered!\n"
             << "       The simulation will now terminate." << LF;
      return { true, ReturnStatus::ErrorNegativeDensity, "negative density encountered\n! getEigens_F" };
   }

   // vector inner products
   double uu =  u*u   + v*v   + w*w;
   //double ub =  u*bx  + v*by  + w*bz;
   double bb = bx*bx + by*by + bz*bz;

   // half-point pressure
   double mml = U1[1]*U1[1] + U1[2]*U1[2] + U1[3]*U1[3];
   double bbl = U1[4]*U1[4] + U1[5]*U1[5] + U1[6]*U1[6];
   double mmr = U2[1]*U2[1] + U2[2]*U2[2] + U2[3]*U2[3];
   double bbr = U2[4]*U2[4] + U2[5]*U2[5] + U2[6]*U2[6];
   double ptotl = (gamma-1.0)*( U1[7] - 0.5*mml/U1[0] - 0.5*bbl ) + 0.5*bbl;
   double ptotr = (gamma-1.0)*( U2[7] - 0.5*mmr/U2[0] - 0.5*bbr ) + 0.5*bbr;
   double ptot  = 0.5 * ( ptotl + ptotr );
   double p = ptot - 0.5*bb;
   if( p < 0.0 ){
      if( break_on_neg_pressure ){
         ERROUT << "ERROR: getEigens_F: Negative pressure encountered!" << LF;
         return { true, ReturnStatus::ErrorNegativePressure, "negative pressure encountered\n! getEigens_F" };
      } else {
         ERROUT << "WARNING: getEigens_F: Negative pressure encountered!\n"
                << "         Results from this point on are suspect.\n"
                << "         Simulation resumed with pressure forced to zero." << LF;
         p = 0.0;
      }
   }

   // speeds
   double a2  = gamma*p / r;
   if( a2 < 0.0 ) a2 = 0.0;
   double a   = sqrt(a2);    // sound speed
   double ca2 = bx*bx/r;
   double ca  = sqrt(ca2);   // Alfven speed
   double c2  = 0.5*(bb/r+a2);
   double ctemp = c2*c2-a2*ca2; if( ctemp < 0.0 ) ctemp = 0.0;
   double cs2 = c2 - sqrt(ctemp); if( cs2 < 0.0 ) cs2 = EPS;
   double cs  = sqrt(cs2);   // slow magnetosonic speed
   double cf2 = c2 + sqrt(ctemp); if( cf2 < 0.0 ) cf2 = EPS;
   double cf  = sqrt(cf2);   // fast magnetosonic speed

#ifdef DEBUG_MIDPOINT_EIGENVALUES
   OUT << "*** DEBUG: get_eigens_f: r, ca, cs, cf = "
       << r << " " << ca << " " << cs << " " << cf << LF;
#endif // DEBUG_MIDPOINT_EIGENVALUES

   // other
   double rroot   = sqrt( r );
   double bperp2  = by*by + bz*bz;
   double sgnbx   = (bx<0.0) ? -1.0 : 1.0;//( (bx>0.0) ? 1 : 0 );
   double alphaf2 = ( (bperp2<EPS) && EPS_EQUAL(ca2,a2) ) ? 1 : (cf2-ca2) / (cf2-cs2);
          alphaf2 = alphaf2 < 0.0 ? 0.0 : alphaf2; // hack for cf ~ ca
   double alphaf  = sqrt( alphaf2 );
   double alphas2 = ( (bperp2<EPS) && EPS_EQUAL(ca2,a2) ) ? 1 : (cf2- a2) / (cf2-cs2);
          alphas2 = alphas2 < 0.0 ? 0.0 : alphas2; // hack for cf ~ a
   double alphas  = sqrt( alphas2 );
   double betay   = bperp2 > EPS ? by / sqrt( bperp2 ) : 1.0/sqrt(2.0);
   double betaz   = bperp2 > EPS ? bz / sqrt( bperp2 ) : 1.0/sqrt(2.0);
   double theta1  = 1.0 / ( alphaf2*a2*(cf2-(gamma-2.0)/(gamma-1.0)*a2) + alphas2*cf2*(cs2-(gamma-2.0)/(gamma-1.0)*a2) );
   double theta2  = 1.0 / ( alphaf2*cf*a*sgnbx + alphas2*cs*ca*sgnbx );

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
   double sgnbt   = ( (by>0) || ( EPS_ZERO(by) && (bz>0) ) ) ? 1 : -1;
   double csfactor = a2 > ca2 ? sgnbt : 1;
   double cffactor = a2 < ca2 ? sgnbt : 1;

   for( int k = 0; k < PRB_DIM; k++ ){
      rv[0][k] *= cffactor;
      rv[6][k] *= cffactor;
      lv[0][k] *= cffactor;
      lv[6][k] *= cffactor;
      rv[2][k] *= csfactor;
      rv[4][k] *= csfactor;
      lv[2][k] *= csfactor;
      lv[4][k] *= csfactor;
   }

   return { false, ReturnStatus::OK, "" };
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status getEigens_G
   ( double U1[PRB_DIM]
   , double U2[PRB_DIM]
   , double gamma
   , double lambda[PRB_DIM]
   , double lv[PRB_DIM][PRB_DIM]
   , double rv[PRB_DIM][PRB_DIM]
   , bool   break_on_neg_pressure
){
   // invert x and y axis
   std::swap( U1[1], U1[2] ); std::swap( U2[1], U2[2] );
   std::swap( U1[4], U1[5] ); std::swap( U2[4], U2[5] );

   // get eigens for inverted x and y
   auto retval = getEigens_F( U1, U2, gamma, lambda, lv, rv, break_on_neg_pressure );
   if( retval.status != ReturnStatus::OK ){
      retval.message += "\n! getEigens_G";
   }

   // return eigenvectors to proper axes
   for( int i = 0; i < PRB_DIM; i++ ){
      std::swap( rv[i][1], rv[i][2] ); std::swap( lv[i][1], lv[i][2] );
      std::swap( rv[i][4], rv[i][5] ); std::swap( lv[i][4], lv[i][5] );
   }

   return retval;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void applyBoundaryConditions
   ( t_matrices U
   , const t_params &params
){
   // left boundary
   switch( params.boundary[params.b_left] ){
   case BoundaryCondition::Undefined:
      criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                   + "applyBoundaryConditions: Unknown left boundary condition." );
      break;
   case BoundaryCondition::Periodic:
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int j = 0; j < NY; j++ ){
            U[k][NXFIRST-2][j] = U[k][NXLAST-2][j];
            U[k][NXFIRST-1][j] = U[k][NXLAST-1][j];
         }
      }
      break;
   case BoundaryCondition::Dirichlet:
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int j = 0; j < NY; j++ ){
            U[k][NXFIRST-2][j] = params.boundary_dirichlet_U[params.b_left][k][j];
            U[k][NXFIRST-1][j] = params.boundary_dirichlet_U[params.b_left][k][j];
         }
      }
      break;
   case BoundaryCondition::Neumann:
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int j = NYFIRST; j < NYLAST; j++ ){
            U[k][NXFIRST-2][j] = U[k][NXFIRST+2][j];
            U[k][NXFIRST-1][j] = U[k][NXFIRST+1][j];
         }
         // corners
         int dix[4] = {2,2,1,1};
         int diy[4] = {2,1,2,1};
         for( int l = 0; l < 4; l++ ){
            // bottom left (NXFIRST, NYFIRST)
            U[k][NXFIRST-dix[l]][NYFIRST-diy[l]] = U[k][NXFIRST+dix[l]][NYFIRST+diy[l]];
            // top left (NXFIRST, NYLAST)
            U[k][NXFIRST-dix[l]][(NYLAST-1)+diy[l]] = U[k][NXFIRST+dix[l]][(NYLAST-1)-diy[l]];
         }
      }
      // use div B = 0 for dBx/dx and dBy/dy boundary conditions
      {
         double ratio_xy = params.dx/params.dy;
         int i, j;
         for( j = 1; j < NY-1; j++ ){
            i = NXFIRST;   U[4][i-1][j] = U[4][i+1][j] - ratio_xy*( U[5][i][j-1] - U[5][i][j+1] );
            i = NXFIRST-1; U[4][i-1][j] = U[4][i+1][j] - ratio_xy*( U[5][i][j-1] - U[5][i][j+1] );
         }
      }
      break;
   case BoundaryCondition::Open:
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int j = 0; j < NY; j++ ){
            U[k][NXFIRST-1][j] = U[k][NXFIRST+1][j];
            U[k][NXFIRST-2][j] = U[k][NXFIRST+2][j];
         }
      }
      break;
   }

   // right boundary
   switch( params.boundary[params.b_right] ){
   case BoundaryCondition::Undefined:
      criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                   + "applyBoundaryConditions: Unknown right boundary condition." );
      break;
   case BoundaryCondition::Periodic:
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int j = 0; j < NY; j++ ){
            U[k][NXLAST  ][j] = U[k][NXFIRST  ][j];
            U[k][NXLAST+1][j] = U[k][NXFIRST+1][j];
         }
      }
      break;
   case BoundaryCondition::Dirichlet:
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int j = 0; j < NY; j++ ){
            U[k][NXLAST  ][j] = params.boundary_dirichlet_U[params.b_right][k][j];
            U[k][NXLAST+1][j] = params.boundary_dirichlet_U[params.b_right][k][j];
         }
      }
      break;
   case BoundaryCondition::Neumann:
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int j = NYFIRST; j < NYLAST; j++ ){
            U[k][NXLAST  ][j] = U[k][NXLAST-2][j];
            U[k][NXLAST+1][j] = U[k][NXLAST-3][j];
         }
         // corners
         int dix[4] = {2,2,1,1};
         int diy[4] = {2,1,2,1};
         for( int l = 0; l < 4; l++ ){
            // bottom right (NXLAST, NYFIRST)
            U[k][(NXLAST-1)+dix[l]][NYFIRST-diy[l]] = U[k][(NXLAST-1)-dix[l]][NYFIRST+diy[l]];
            // top right (NXLAST, NYLAST)
            U[k][(NXLAST-1)+dix[l]][(NYLAST-1)+diy[l]] = U[k][(NXLAST-1)-dix[l]][(NYLAST-1)-diy[l]];
         }
      }
      // use div B = 0 for dBx/dx and dBy/dy boundary conditions
      {
         double ratio_xy = params.dx/params.dy;
         int i, j;
         for( j = 1; j < NY-1; j++ ){
            i = NXLAST-1; U[4][i+1][j] = U[4][i-1][j] - ratio_xy*( U[5][i][j+1] - U[5][i][j-1] );
            i = NXLAST;   U[4][i+1][j] = U[4][i-1][j] - ratio_xy*( U[5][i][j+1] - U[5][i][j-1] );
         }
      }
      break;
   case BoundaryCondition::Open:
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int j = 0; j < NY; j++ ){
            U[k][NXLAST  ][j] = U[k][NXLAST-2][j];
            U[k][NXLAST+1][j] = U[k][NXLAST-3][j];
         }
      }
      break;
   }

   // bottom boundary
   switch( params.boundary[params.b_bottom] ){
   case BoundaryCondition::Undefined:
      criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                   + "applyBoundaryConditions: Unknown bottom boundary condition." );
      break;
   case BoundaryCondition::Periodic:
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int i = 0; i < NX; i++ ){
            U[k][i][NYFIRST-2] = U[k][i][NYLAST-2];
            U[k][i][NYFIRST-1] = U[k][i][NYLAST-1];
         }
      }
      break;
   case BoundaryCondition::Dirichlet:
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int i = 0; i < NX; i++ ){
            U[k][i][NYFIRST-2] = params.boundary_dirichlet_U[params.b_bottom][k][i];
            U[k][i][NYFIRST-1] = params.boundary_dirichlet_U[params.b_bottom][k][i];
         }
      }
      break;
   case BoundaryCondition::Neumann:
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int i = NXFIRST; i < NXLAST; i++ ){
            U[k][i][NYFIRST-1] = U[k][i][NYFIRST+1];
            U[k][i][NYFIRST-2] = U[k][i][NYFIRST+2];
         }
         // corners
         int dix[4] = {2,2,1,1};
         int diy[4] = {2,1,2,1};
         for( int l = 0; l < 4; l++ ){
            // bottom left (NXFIRST, NYFIRST)
            U[k][NXFIRST-dix[l]][NYFIRST-diy[l]] = U[k][NXFIRST+dix[l]][NYFIRST+diy[l]];
            // bottom right (NXLAST, NYFIRST)
            U[k][(NXLAST-1)+dix[l]][NYFIRST-diy[l]] = U[k][(NXLAST-1)-dix[l]][NYFIRST+diy[l]];
         }
      }
      // use div B = 0 for dBx/dx and dBy/dy boundary conditions
      {
         double ratio_yx = params.dy/params.dx;
         int i, j;
         for( i = 1; i < NX-1; i++ ){
            j = NYFIRST;   U[5][i][j-1] = U[5][i][j+1] - ratio_yx*( U[4][i-1][j] - U[4][i+1][j] );
            j = NYFIRST-1; U[5][i][j-1] = U[5][i][j+1] - ratio_yx*( U[4][i-1][j] - U[4][i+1][j] );
         }
      }
      break;
   case BoundaryCondition::Open:
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int i = 0; i < NX; i++ ){
            U[k][i][NYFIRST-1] = U[k][i][NYFIRST+1];
            U[k][i][NYFIRST-2] = U[k][i][NYFIRST+2];
         }
      }
      break;
   }

   // top boundary
   switch( params.boundary[params.b_top] ){
   case BoundaryCondition::Undefined:
      criticalError( ReturnStatus::ErrorWrongParameter, std::string{}
                   + "applyBoundaryConditions: Unknown top boundary condition." );
      break;
   case BoundaryCondition::Periodic:
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int i = 0; i < NX; i++ ){
            U[k][i][NYLAST  ] = U[k][i][NYFIRST];
            U[k][i][NYLAST+1] = U[k][i][NYFIRST+1];
         }
      }
      break;
   case BoundaryCondition::Dirichlet:
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int i = 0; i < NX; i++ ){
            U[k][i][NYLAST  ] = params.boundary_dirichlet_U[params.b_top][k][i];
            U[k][i][NYLAST+1] = params.boundary_dirichlet_U[params.b_top][k][i];
         }
      }
      break;
   case BoundaryCondition::Neumann:
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int i = NXFIRST; i < NXLAST; i++ ){
            U[k][i][NYLAST  ] = U[k][i][NYLAST-2];
            U[k][i][NYLAST+1] = U[k][i][NYLAST-3];
         }
         // corners
         int dix[4] = {2,2,1,1};
         int diy[4] = {2,1,2,1};
         for( int l = 0; l < 4; l++ ){
            // top left (NXFIRST, NYLAST)
            U[k][NXFIRST-dix[l]][(NYLAST-1)+diy[l]] = U[k][NXFIRST+dix[l]][(NYLAST-1)-diy[l]];
            // top right (NXLAST, NYLAST)
            U[k][(NXLAST-1)+dix[l]][(NYLAST-1)+diy[l]] = U[k][(NXLAST-1)-dix[l]][(NYLAST-1)-diy[l]];
         }
      }
      // use div B = 0 for dBx/dx and dBy/dy boundary conditions
      {
         double ratio_yx = params.dy/params.dx;
         int i, j;
         for( i = 1; i < NX-1; i++ ){
            j = NYLAST-1; U[5][i][j+1] = U[5][i][j-1] - ratio_yx*( U[4][i+1][j] - U[4][i-1][j] );
            j = NYLAST;   U[5][i][j+1] = U[5][i][j-1] - ratio_yx*( U[4][i+1][j] - U[4][i-1][j] );
         }
      }
      break;
   case BoundaryCondition::Open:
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int i = 0; i < NX; i++ ){
            U[k][i][NYLAST  ] = U[k][i][NYLAST-2];
            U[k][i][NYLAST+1] = U[k][i][NYLAST-3];
         }
      }
      break;
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void getFluxes
   ( t_matrices U
   , t_matrices F
   , t_matrices G
   , const t_params &params
){
   // flux F(U), G(U)
   /*#ifdef OPENMP
   // omp-id: scheme_eno:get_fluxes:1
   # pragma omp parallel \
     default( shared )

   # pragma omp for
   #endif*/
   for( int i = 0; i < NX; i++ ){
      for( int j = 0; j < NY; j++ ){
         double r  = U[0][i][j];
         double mx = U[1][i][j]; double u = mx / r;
         double my = U[2][i][j]; double v = my / r;
         double mz = U[3][i][j]; double w = mz / r;
         double bx = U[4][i][j];
         double by = U[5][i][j];
         double bz = U[6][i][j];
         double e  = U[7][i][j];
         double uu =  u*u  +  v*v  +  w*w;
         double ub =  u*bx +  v*by +  w*bz;
         double bb = bx*bx + by*by + bz*bz;
         double p = (params.gamma-1.0)*(e-0.5*r*uu-0.5*bb);
         double ptot = p + 0.5*bb;

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
t_status findNumericalFluxRoe_F
   ( double  U1[PRB_DIM]
   , double  U2[PRB_DIM]
   , double  F[PRB_DIM][8]
   , double  VUF[PRB_DIM][7]
   , double  F_[PRB_DIM]
   , double &a_max
   , int     first_valid
   , int     last_valid
   , const t_params &params
){
   //const int TN      = 8;
   const int TNFIRST = 2;
   //const int TNLAST  = 6;
   // eigenvalues & eigenvectors @ u_(i+1/2)
   double a[PRB_DIM], lv[PRB_DIM][PRB_DIM], rv[PRB_DIM][PRB_DIM];
   // characteristics, differences & fluxes
   double w[PRB_DIM][8], Vw[PRB_DIM][7];
   double wm[PRB_DIM], wp[PRB_DIM], fw[PRB_DIM];
   // maximum wave speed (ie, eigenvalue)
   a_max = 0.0;
   // current index
   int i = TNFIRST+1;

   // eigensystem
   auto retval = getEigens_F( U1, U2, params.gamma, a, lv, rv, params.break_on_neg_pressure );
   if( retval.status != ReturnStatus::OK ){
      ERROUT << "ERROR: findNumericalFluxRoe_F: Flux eigensystem." << LF;
      retval.message += "\n! findNumericalFluxRoe_F";
      return retval;
   }

   // update maximum wave speed
   for( int k = 0; k < PRB_DIM; k++ )
      if( a_max < fabs(a[k]) )
         a_max = fabs(a[k]);

   // local characteristics and undivided differences
   for( int k = 0; k < PRB_DIM; k++ ){
      for( int l = i-1; l <= i+2; l++ ){
         w[k][l] = 0.0;
         for( int m = 0; m < PRB_DIM; m++ )
            w[k][l] += lv[k][m]*F[m][l];
      }
   }
   for( int k = 0; k < PRB_DIM; k++ ){
      for( int l = i-1; l <= i+1; l++ ){
         Vw[k][l] = 0.0;
         for( int m = 0; m < PRB_DIM; m++ )
            Vw[k][l] += lv[k][m]*VUF[m][l];
      }
   }

   // set up boundaries
   for( int k = 0; k < PRB_DIM; k++ ){
      w[k][first_valid-2] = 1e100;
      w[k][first_valid-1] = 1e100;
      w[k][last_valid   ] = 1e100;
      w[k][last_valid+1 ] = 1e100;
      Vw[k][first_valid-2] = 0;
      Vw[k][first_valid-1] = 1e100;
      Vw[k][last_valid-1 ] = 1e100;
      Vw[k][last_valid   ] = 0;
   }

   // minus edge
   for( int k = 0; k < PRB_DIM; k++ ){
      if( ABS(Vw[k][i-1]) < ABS(Vw[k][i]) ) // left (stencil -3/2, -1/2, +1/2)
         wm[k] = -(1.0/2.0)*w[k][i-1] + (3.0/2.0)*w[k][i];
      else // right (stencil -1/2, +1/2, +3/2)
         wm[k] = (1.0/2.0)*w[k][i] + (1.0/2.0)*w[k][i+1];
   }

   // plus edge
   for( int k = 0; k < PRB_DIM; k++ ){
      if( ABS(Vw[k][i]) < ABS(Vw[k][i+1]) ) // left (stencil -3/2, -1/2, +1/2)
         wp[k] = (1.0/2.0)*w[k][i] + (1.0/2.0)*w[k][i+1];
      else // right (stencil -1/2, +1/2, +3/2)
         wp[k] = (3.0/2.0)*w[k][i+1] - (1.0/2.0)*w[k][i+2];
   }

   // fluxes
   for( int k = 0; k < PRB_DIM; k++ ){
      if( a[k] >= 0 && wm[k] < 1.0e10 )
         fw[k] = wm[k];
      else if( a[k] <= 0 && wp[k] < 1.0e10 )
         fw[k] = wp[k];
//      else if( wp[k] > 1.0e10 )
//         fw[k] = wm[k];
//      else if( wm[k] > 1.0e10 )
//         fw[k] = wp[k];
      else
         fw[k] = 0.0;
   }

   // return to physical space fluxes
   for( int k = 0; k < PRB_DIM; k++ ){
      F_[k] = 0.0;
      for( int l = 0; l < PRB_DIM; l++ )
         F_[k] += rv[l][k]*fw[l];
   }

   return { false, ReturnStatus::OK, "" };
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status findNumericalFluxRoe_G
   ( double  U1[PRB_DIM]
   , double  U2[PRB_DIM]
   , double  G[PRB_DIM][8]
   , double  VUG[PRB_DIM][7]
   , double  G_[PRB_DIM]
   , double &a_max
   , int     first_valid
   , int     last_valid
   , const t_params &params
){
   // invert x and y axis
   std::swap( U1[1], U1[2] ); std::swap( U1[4], U1[5] );
   std::swap( U2[1], U2[2] ); std::swap( U2[4], U2[5] );
   for( int i = 0; i < 8; i++ ){
      std::swap( G[1][i], G[2][i] ); std::swap( G[4][i], G[5][i] );
   }
   for( int i = 0; i < 7; i++ ){
      std::swap( VUG[1][i], VUG[2][i] ); std::swap( VUG[4][i], VUG[5][i] );
   }

   // find flux for inverted x and y
   auto retval = findNumericalFluxRoe_F( U1, U2, G, VUG, G_, a_max, first_valid, last_valid, params );
   if( retval.status != ReturnStatus::OK ){
      retval.message += "\n! findNumericalFluxRoe_G";
   }

   // return found flux to proper axes
   std::swap( G_[1], G_[2] ); std::swap( G_[4], G_[5] );

   return retval;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status findNumericalFluxLF_F
   ( double  U1[PRB_DIM]
   , double  U2[PRB_DIM]
   , double  U[PRB_DIM][8]
   , double  F[PRB_DIM][8]
   , double  F_[PRB_DIM]
   , double &a_max
   , double  alpha[PRB_DIM]
   , int     first_valid
   , int     last_valid
   , const t_params &params
){
   //const int TN      = 8;
   const int TNFIRST = 2;
   //const int TNLAST  = 6;
   // eigenvalues & eigenvectors @ u_(i+1/2)
   double a[PRB_DIM], lv[PRB_DIM][PRB_DIM], rv[PRB_DIM][PRB_DIM];
   // characteristics, differences & fluxes
   double RU[PRB_DIM][8], RF[PRB_DIM][8];
   double w[PRB_DIM][8], Vw[PRB_DIM][7];
   double wm[PRB_DIM], wp[PRB_DIM], fw[PRB_DIM];
   // maximum wave speed (ie, eigenvalue)
   a_max = 0.0;
   // current index
   int i = TNFIRST+1;

   // eigensystem
   auto retval = getEigens_F( U1, U2, params.gamma, a, lv, rv, params.break_on_neg_pressure );
   if( retval.status != ReturnStatus::OK ){
      ERROUT << "ERROR: findNumericalFluxLF_F: Flux eigensystem." << LF;
      retval.message += "\n! findNumericalFluxLF_F";
      return retval;
   }

   // update maximum wave speed
   for( int k = 0; k < PRB_DIM; k++ )
      if( a_max < fabs(a[k]) )
         a_max = fabs(a[k]);

   // local characteristics and undivided differences
   for( int k = 0; k < PRB_DIM; k++ ){
      for( int l = i-1; l <= i+2; l++ ){
         RU[k][l] = 0.0;
         RF[k][l] = 0.0;
         for( int m = 0; m < PRB_DIM; m++ ){
            RU[k][l] += lv[k][m]*U[m][l];
            RF[k][l] += lv[k][m]*F[m][l];
         }
      }
   }

   // positive part of the flux splitting
   for( int k = 0; k < PRB_DIM; k++ ){
      for( int l = i-1; l <= i+2; l++ )
         w[k][l] = 0.5*( RF[k][l] + alpha[k]*RU[k][l] );
      for( int l = i-1; l <= i+1; l++ )
         Vw[k][l] = w[k][l+1] - w[k][l];
   }
   // flux on the minus edge
   for( int k = 0; k < PRB_DIM; k++ ){
      if( ABS(Vw[k][i-1]) < ABS(Vw[k][i]) ) // left (stencil -3/2, -1/2, +1/2)
         wm[k] = -(1.0/2.0)*w[k][i-1] + (3.0/2.0)*w[k][i];
      else // right (stencil -1/2, +1/2, +3/2)
         wm[k] = (1.0/2.0)*w[k][i] + (1.0/2.0)*w[k][i+1];
   }

   // negative part of the flux splitting
   for( int k = 0; k < PRB_DIM; k++ ){
      for( int l = i-1; l <= i+2; l++ )
         w[k][l] = 0.5*( RF[k][l] - alpha[k]*RU[k][l] );
      for( int l = i-1; l <= i+1; l++ )
         Vw[k][l] = w[k][l+1] - w[k][l];
   }
   // flux on the plus edge
   for( int k = 0; k < PRB_DIM; k++ ){
      if( ABS(Vw[k][i]) < ABS(Vw[k][i+1]) ) // left (stencil -3/2, -1/2, +1/2)
         wp[k] = (1.0/2.0)*w[k][i] + (1.0/2.0)*w[k][i+1];
      else // right (stencil -1/2, +1/2, +3/2)
         wp[k] = (3.0/2.0)*w[k][i+1] - (1.0/2.0)*w[k][i+2];
   }

   // total numerical flux
   for( int k = 0; k < PRB_DIM; k++ ){
//      fw[k] = wm[k] + wp[k];
      if( a[k] >= 0 && first_valid <= i )
         fw[k] = wm[k] + wp[k];
      else if( a[k] <= 0 && last_valid >= i+2 )
         fw[k] = wm[k] + wp[k];
      else
         fw[k] = 0.0;
   }

   // return to physical space fluxes
   for( int k = 0; k < PRB_DIM; k++ ){
      F_[k] = 0.0;
      for( int l = 0; l < PRB_DIM; l++ )
         F_[k] += rv[l][k]*fw[l];
   }

   return { false, ReturnStatus::OK, "" };
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status findNumericalFluxLF_G
   ( double  U1[PRB_DIM]
   , double  U2[PRB_DIM]
   , double  U[PRB_DIM][8]
   , double  G[PRB_DIM][8]
   , double  G_[PRB_DIM]
   , double &a_max
   , double  alpha[PRB_DIM]
   , int     first_valid
   , int     last_valid
   , const t_params &params
){
   // invert x and y axis
   std::swap( U1[1], U1[2] ); std::swap( U1[4], U1[5] );
   std::swap( U2[1], U2[2] ); std::swap( U2[4], U2[5] );
   for( int i = 0; i < 8; i++ ){
      std::swap( U[1][i], U[2][i] ); std::swap( U[4][i], U[5][i] );
      std::swap( G[1][i], G[2][i] ); std::swap( G[4][i], G[5][i] );
   }

   // find flux for inverted x and y
   auto retval = findNumericalFluxLF_F( U1, U2, U, G, G_, a_max, alpha, first_valid, last_valid, params );
   if( retval.status != ReturnStatus::OK ){
      retval.message += "\n! findNumericalFluxLF_G";
   }

   // return found flux to proper axes
   std::swap( G_[1], G_[2] ); std::swap( G_[4], G_[5] );

   return retval;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status methodENOSystem
   ( t_matrices  U
   , t_matrices  UL
   , double     &dt_step
   , const t_params &params
){
   static char initialized = 0;
   static t_matrices F, G;      // physical fluxes
   static t_matrices VUF, VUG;  // undivided differences
   static t_matrices F_, G_;    // numerical fluxes
   double alpha[PRB_DIM];       // viscosity coefficient in the Lax-Friedrichs flux splitting
   static t_vectors alphaF, alphaG; // alpha for every row, column
   t_status retval;             // for processing function return values
   static int *first_valid_F;
   static int *first_valid_G;
   static int *last_valid_F;
   static int *last_valid_G;

   // track maximum wave speeds (ie, eigenvalues)
   double a_max_f = 0.0;
   double a_max_g = 0.0;

   // temporary storage for function calls
   const int TN      = 8;
   const int TNFIRST = 2;
   const int TNLAST  = 6;
   double tU1[PRB_DIM], tU2[PRB_DIM], tU[PRB_DIM][TN];
   double tF[PRB_DIM][TN], tVUF[PRB_DIM][TN-1], tF_[PRB_DIM];
   double ta_max;
   double talpha[PRB_DIM];

   bool for_break;

   if( !initialized ){
      F   = createMatrices( PRB_DIM, NX, NY );
      G   = createMatrices( PRB_DIM, NX, NY );
      VUF = createMatrices( PRB_DIM, NX, NY );
      VUG = createMatrices( PRB_DIM, NX, NY );
      F_  = createMatrices( PRB_DIM, NX, NY );
      G_  = createMatrices( PRB_DIM, NX, NY );
      alphaF = createVectors( PRB_DIM, NY );
      alphaG = createVectors( PRB_DIM, NX );

      first_valid_F = (int *)malloc( NX * sizeof(*first_valid_F) );
      last_valid_F  = (int *)malloc( NX * sizeof(*last_valid_F)  );
      first_valid_G = (int *)malloc( NY * sizeof(*first_valid_G) );
      last_valid_G  = (int *)malloc( NY * sizeof(*last_valid_G)  );

      for( int i = 0; i < NX; i++ ){
         first_valid_F[i] = TNFIRST;
         last_valid_F [i] = TNLAST;
      }

      if( params.boundary[params.b_left] == BoundaryCondition::Open ){
         first_valid_F[NXFIRST-2] = TNFIRST+3;
         first_valid_F[NXFIRST-1] = TNFIRST+2;
         first_valid_F[NXFIRST  ] = TNFIRST+1;
      }
      if( params.boundary[params.b_right] == BoundaryCondition::Open ){
         last_valid_F[NXLAST-3] = TNLAST;
         last_valid_F[NXLAST-2] = TNLAST-1;
         last_valid_F[NXLAST-1] = TNLAST-2;
         last_valid_F[NXLAST  ] = TNLAST-3;
      }

      for( int j = 0; j < NY; j++ ){
         first_valid_G[j] = TNFIRST;
         last_valid_G [j] = TNLAST;
      }

      if( params.boundary[params.b_bottom] == BoundaryCondition::Open ){
         first_valid_G[NYFIRST-2] = TNFIRST+3;
         first_valid_G[NYFIRST-1] = TNFIRST+2;
         first_valid_G[NYFIRST  ] = TNFIRST+1;
      }
      if( params.boundary[params.b_top] == BoundaryCondition::Open ){
         last_valid_G[NYLAST-3] = TNLAST;
         last_valid_G[NYLAST-2] = TNLAST-1;
         last_valid_G[NYLAST-1] = TNLAST-2;
         last_valid_G[NYLAST  ] = TNLAST-3;
      }

      initialized = 1;
   }

   // first, boundary conditions
   applyBoundaryConditions( U, params );

   // data is ready, calculate fluxes
   getFluxes( U, F, G, params );

   // undivided differences V_UF, V_UG
   for( int k = 0; k < PRB_DIM; k++ ){
      for( int i = NXFIRST-2; i < NXLAST+1; i++ ){
         for( int j = NYFIRST-2; j < NYLAST+1; j++ ){
            VUF[k][i][j] = F[k][i+1][j] - F[k][i][j];
            VUG[k][i][j] = G[k][i][j+1] - G[k][i][j];
         }
      }
   }

   // find the viscosity coefficients required for LF flux splitting
   if( params.scheme == IntegrationMethod::ENO_LF ){
      for( int k = 0; k < PRB_DIM; k++ ){
         for( int i = NXFIRST-1; i < NXLAST; i++ )
            alphaG[k][i] = 0.0;
         for( int j = NYFIRST-1; j < NYLAST; j++ )
            alphaF[k][j] = 0.0;
      }

      for( int i = NXFIRST-1; i < NXLAST; i++ ){
         for( int j = NYFIRST-1; j < NYLAST; j++ ){
            // prepare data
            for( int k = 0; k < PRB_DIM; k++ ){
               tU1[k] = U[k][i][j];         // left point
               tU2[k] = U[k][i+1][j];       // right point
            }
            // get localized eigenvalues
            retval = getEigenvalues( tU1, params.gamma, talpha, params.break_on_neg_pressure );
            if( retval.status != ReturnStatus::OK ){
               ERROUT << "ERROR: methodENOSystem: F eigenvalues." << LF;
               ERROUT << "       Details:" << LF
                      << "         * x = " << params.start_x + (i-NXFIRST)*params.dx << "; i = " << i-NXFIRST << LF
                      << "         * y = " << params.start_y + (j-NYFIRST)*params.dy << "; j = " << j-NYFIRST << LF;

               retval.message += "\n! methodENOSystem -> F eigenvalues";
               return retval;
            }
            // update row, column maximums
            for( int k = 0; k < PRB_DIM; k++ ){
               if( alphaF[k][j] < fabs(talpha[k]) )
                  alphaF[k][j] = fabs(talpha[k]);
            }

            // prepare data
            for( int k = 0; k < PRB_DIM; k++ ){
               tU1[k] = U[k][i][j];         // down point
               tU2[k] = U[k][i][j+1];       // up point
            }
            std::swap( tU1[1], tU1[2] ); std::swap( tU1[4], tU1[5] );
            std::swap( tU2[1], tU2[2] ); std::swap( tU2[4], tU2[5] );

            // get localized eigenvalues
            retval = getEigenvalues( tU1, params.gamma, talpha, params.break_on_neg_pressure );
            if( retval.status != ReturnStatus::OK ){
               ERROUT << "ERROR: methodENOSystem: G eigenvalues." << LF;
               ERROUT << "       Details:" << LF
                      << "         * x = " << params.start_x + (i-NXFIRST)*params.dx << "; i = " << i-NXFIRST << LF
                      << "         * y = " << params.start_y + (j-NYFIRST)*params.dy << "; j = " << j-NYFIRST << LF;

               retval.message += "\n! methodENOSystem -> G eigenvalues";
               return retval;
            }
            // update row, column maximums
            for( int k = 0; k < PRB_DIM; k++ ){
               if( alphaG[k][i] < fabs(talpha[k]) )
                  alphaG[k][i] = fabs(talpha[k]);
            }
         }
      }
   }

   /******************************
    *           F flux           *
    ******************************/
   for_break = false;
   #ifdef OPENMP
   // omp-id: scheme_eno:eno_system:2
   # pragma omp parallel \
     default( shared ) \
     private ( tU1, tU2, tU, tF, tVUF, tF_, ta_max, alpha )

   # pragma omp for reduction ( max : a_max_f )
   #endif
   for( int i = NXFIRST-1; i < NXLAST; i++ ){
      for( int j = NYFIRST; j < NYLAST; j++ ){
         // prepare data
         for( int k = 0; k < PRB_DIM; k++ ){
            if( params.boundary[params.b_left] == BoundaryCondition::Open && i < NXFIRST ){
               tU1[k] = U[k][i+1][j];       // left point
            } else {
               tU1[k] = U[k][i][j];         // left point
            }
            if( params.boundary[params.b_right] == BoundaryCondition::Open && i > NXLAST-2 ){
               tU2[k] = U[k][i][j];         // right point
            } else {
               tU2[k] = U[k][i+1][j];       // right point
            }
            for( int l = 0; l < 4; l++ ){
               tU  [k][TNFIRST+l] = U[k][i-1+l][j]; // point
               tF  [k][TNFIRST+l] = F[k][i-1+l][j]; // physical flux
            }
            for( int l = 0; l < 3; l++ ){
               tVUF[k][TNFIRST+l] = VUF[k][i-1+l][j]; // undivided differences
            }
            alpha[k] = alphaF[k][j];
         }

         // find numerical flux
         if( params.scheme == IntegrationMethod::ENO_Roe ){
            retval = findNumericalFluxRoe_F( tU1, tU2, tF, tVUF, tF_, ta_max, first_valid_F[i], last_valid_F[i], params );
         } else if( params.scheme == IntegrationMethod::ENO_LF ){
            retval = findNumericalFluxLF_F( tU1, tU2, tU, tF, tF_, ta_max, alpha, first_valid_F[i], last_valid_F[i], params );
         }
         if( retval.status != ReturnStatus::OK ){
            ERROUT << "ERROR: methodENOSystem: F numerical flux." << LF;
            retval.message += "\n! methodENOSystem -> F flux";
            for_break = true;
            break;
         }

         //process results
         for( int k = 0; k < PRB_DIM; k++ ){
            F_[k][i][j] = tF_[k];
         }
//         if( params.boundary != BOUNDARY_OPEN || ( i > NXFIRST-1 && i < NXLAST-1 ) ){
         if( ta_max > a_max_f )
            a_max_f = ta_max;
//         }
      }
      if( for_break ){
         break;
      }
   }
   // had to remove return out of the loop so it could work with OpenMP
   if( for_break ){
      return retval;
   }

   /******************************
    *           G flux           *
    ******************************/
   for_break = false;
   #ifdef OPENMP
   // omp-id: scheme_eno:eno_system:3
   # pragma omp parallel \
     default( shared ) \
     private ( tU1, tU2, tU, tF, tVUF, tF_, ta_max, alpha )

   # pragma omp for reduction ( max : a_max_g )
   #endif
   for( int i = NXFIRST; i < NXLAST; i++ ){
      for( int j = NYFIRST-1; j < NYLAST; j++ ){
         // prepare data
         for( int k = 0; k < PRB_DIM; k++ ){
            if( params.boundary[params.b_bottom] == BoundaryCondition::Open && j < NYFIRST ){
               tU1[k] = U[k][i][j+1];       // down point
            } else {
               tU1[k] = U[k][i][j];         // down point
            }
            if( params.boundary[params.b_top] == BoundaryCondition::Open && j > NYLAST-2 ){
               tU2[k] = U[k][i][j];         // up point
            } else {
               tU2[k] = U[k][i][j+1];       // up point
            }
            for( int l = 0; l < 4; l++ ){
               tU  [k][TNFIRST+l] = U[k][i][j-1+l]; // point
               tF  [k][TNFIRST+l] = G[k][i][j-1+l]; // physical flux
            }
            for( int l = 0; l < 3; l++ ){
               tVUF[k][TNFIRST+l] = VUG[k][i][j-1+l]; // undivided differences
            }
            alpha[k] = alphaG[k][i];
         }

         // find numerical flux
         if( params.scheme == IntegrationMethod::ENO_Roe ){
            retval = findNumericalFluxRoe_G( tU1, tU2, tF, tVUF, tF_, ta_max, first_valid_G[j], last_valid_G[j], params );
         } else if( params.scheme == IntegrationMethod::ENO_LF ){
            retval = findNumericalFluxLF_G( tU1, tU2, tU, tF, tF_, ta_max, alpha, first_valid_G[j], last_valid_G[j], params );
         }
         if( retval.status != ReturnStatus::OK ){
            ERROUT << "ERROR: methodENOSystem: G numerical flux." << LF;
            retval.message += "\n! methodENOSystem -> F flux";
            for_break = true;
            break;
         }

         //process results
         for( int k = 0; k < PRB_DIM; k++ ){
            G_[k][i][j] = tF_[k];
         }
//         if( params.boundary != BOUNDARY_OPEN || ( j > NYFIRST-1 && j < NYLAST-1 ) ){
         if( ta_max > a_max_g )
            a_max_g = ta_max;
//         }
      }
      if( for_break ){
         break;
      }
   }
   // had to remove return out of the loop so it could work with OpenMP
   if( for_break ){
      return retval;
   }

#ifdef DEBUG_MAX_VELOCITY
   OUT << "*** DEBUG: a_max_f = " << a_max_f << ", a_max_g = " << a_max_g << "\n";
#endif // DEBUG_MAX_VELOCITY
   // update local dt from max wave speeds
   dt_step = std::min( params.dx/a_max_f, params.dy/a_max_g );

   // use numerical fluxes to calculate dU/dt
   for( int k = 0; k < PRB_DIM; k++ ){
      for( int i = NXFIRST; i < NXLAST; i++ ){
         for( int j = NYFIRST; j < NYLAST; j++ ){
            UL[k][i][j] = -(1.0/params.dx)*( F_[k][i][j] - F_[k][i-1][j] )
                          -(1.0/params.dy)*( G_[k][i][j] - G_[k][i][j-1] );
         }
      }
   }

   return { false, ReturnStatus::OK, "" };
}
