#include "euler1d.hpp"

double riemann_f( double p4, double p1, double p5,
                  double r1, double r5, double gamma ){
   double z, c1, c5, gm1, gp1, g2, fact;

   z = ( p4/p5 - 1.0);
   c1 = sqrt(gamma*p1/r1);
   c5 = sqrt(gamma*p5/r5);

   gm1 = gamma - 1.0;
   gp1 = gamma + 1.0;
   g2  = 2.0 * gamma;

   fact = gm1/g2*(c5/c1)*z / sqrt(1.0+gp1/g2*z);
   fact = pow( 1.0-fact, g2/gm1 );

   return p1*fact - p4;
}

int riemann_solver( t_data &data, const t_params &params ){
   double r1, r3, r4, r5;
   double u1, u3, u4, u5;
   double p1, p3, p4, p5;
   double c1, c3,     c5;

   double rL = params.param_dbl[0];
   double uL = params.param_dbl[1];
   double pL = params.param_dbl[2];
   double rR = params.param_dbl[3];
   double uR = params.param_dbl[4];
   double pR = params.param_dbl[5];

   double t = data.t_current + data.dt;
   int itmax = params.param_int[0];
   double gamma = params.gamma;

   if( pL > pR ){
      r1 = rL; p1 = pL; u1 = uL;
      r5 = rR; p5 = pR; u5 = uR;
   } else {
      r1 = rR; p1 = pR; u1 = uR;
      r5 = rL; p5 = pL; u5 = uL;
   }

   // solve for post-shock pressure by secant method
   // initial guesses
   double p40 = p1;
   double p41 = p5;
   double f0  = riemann_f( p40, p1, p5, r1, r5, gamma );

   // iterate
   p4 = p40;
   for( int i = 1; i <= itmax; i++ ){
      double f1 = riemann_f( p41, p1, p5, r1, r5, gamma );
      if( EPS_EQUAL( f1, f0 ) ) break;
      p4 = p41 - (p41-p40)*f1 / (f1-f0);

      double error = ABS(p4-p41)/p41;
      if( error < EPS ) break;

      p40 = p41;
      p41 = p4;
      f0  = f1;

      if( i == itmax ) // failed to converge
         return RET_RIEMANN_FAILED_TO_CONVERGE;
   }

   // compute post-shock density and velocity
   double z  = p4/p5 - 1.0;
   double gm1 = gamma - 1.0;
   double gp1 = gamma + 1.0;
   double gmfac1 = 0.5 * gm1/gamma;
   double gmfac2 = 0.5 * gp1/gamma;
   double fact = sqrt( 1.0+gmfac2*z );

   c5 = sqrt( gamma*p5/r5 );
   u4 = c5*z / (gamma*fact);
   r4 = r5 * (1.0+gmfac2*z) / (1.0+gmfac1*z);

   // shock speed
   double w = c5 * fact;

   // compute values at foot of rarefaction
   p3 = p4;
   u3 = u4;
   r3 = r1 * pow( p3/p1, 1.0/gamma );

   // compute positions of waves for pL > pR
   if( pL > pR ){
      c1 = sqrt(gamma*p1/r1);
      c3 = sqrt(gamma*p3/r3);

      double xsh = w * t;
      double xcd = u3 * t;
      double xft = (u3-c3) * t;
      double xhd = -c1 * t;

#ifdef DEBUG
      // report
      OUT << "# region  density  speed  pressure  moment  energy" << LF;
      OUT << "#   1    " << r1 << "\t" << p1 << "\t" << u1 << LF;
      OUT << "#   2        R A R E F A C T I O N" << LF;
      OUT << "#   3    " << r3 << "\t" << p3 << "\t" << u3 << LF;
      OUT << "#   4    " << r4 << "\t" << p4 << "\t" << u4 << LF;
      OUT << "#   5    " << r5 << "\t" << p5 << "\t" << u5 << LF;
      OUT << "#" << LF;
      OUT << "# Rarefaction head      @ x = " << xhd << LF;
      OUT << "# Rarefaction foot      @ x = " << xft << LF;
      OUT << "# Contact discontinuity @ x = " << xcd << LF;
      OUT << "# Shock                 @ x = " << xsh << LF;
#endif // DEBUG

      // compute solution as a function of position
      for( int i = 1; i <= params.nx; i++ ){
         double x = (i-2-params.nx/2)*params.dx;
         if( x < xhd ){
            data.U[0][i] = r1;
            data.p[i]    = p1;
            data.u[0][i] = u1;
         } else if( x < xft ){
            data.u[0][i] = 2.0/gp1 * (c1+x/t);
            fact = 1.0 - 0.5*gm1*data.u[0][i]/c1;
            data.U[0][i] = r1 * pow( fact, 2.0/gm1 );
            data.p[i]    = p1 * pow( fact, 2.0*gamma/gm1 );
         } else if( x < xcd ){
            data.U[0][i] = r3;
            data.p[i]    = p3;
            data.u[0][i] = u3;
         } else if( x < xsh ){
            data.U[0][i] = r4;
            data.p[i]    = p4;
            data.u[0][i] = u4;
         } else {
            data.U[0][i] = r5;
            data.p[i]    = p5;
            data.u[0][i] = u5;
         }
      }
   }

   // compute positions of waves for pL < pR
   if( pL < pR ){
      c1 = sqrt(gamma*p1/r1);
      c3 = sqrt(gamma*p3/r3);

      double xsh = -w * t;
      double xcd = -u3 * t;
      double xft = -(u3-c3) * t;
      double xhd = c1 * t;

#ifdef DEBUG
      // report
      OUT << "# region  density  speed  pressure  moment  energy" << LF;
      OUT << "#   1    " << r5 << "\t" << p5 << "\t" << u5 << LF;
      OUT << "#   2    " << r4 << "\t" << p4 << "\t" << u4 << LF;
      OUT << "#   3    " << r3 << "\t" << p3 << "\t" << u3 << LF;
      OUT << "#   4        R A R E F A C T I O N" << LF;
      OUT << "#   5    " << r1 << "\t" << p1 << "\t" << u1 << LF;
      OUT << "#" << LF;
      OUT << "# Shock                 @ x = " << xsh << LF;
      OUT << "# Contact discontinuity @ x = " << xcd << LF;
      OUT << "# Rarefaction foot      @ x = " << xft << LF;
      OUT << "# Rarefaction head      @ x = " << xhd << LF;
#endif // DEBUG

      for( int i = 1; i <= params.nx; i++ ){
         double x = (i-2-params.nx/2)*params.dx;
         if( x > xhd ){
            data.U[0][i] = r1;
            data.p[i]    = p1;
            data.u[0][i] = -u1;
         } else if( x > xft ){
            data.u[0][i] = -2.0/gp1 * (c1-x/t);
            fact = 1.0 + 0.5*gm1*data.u[0][i]/c1;
            data.U[0][i] = r1 * pow( fact, 2.0/gm1 );
            data.p[i]    = p1 * pow( fact, 2.0*gamma/gm1 );
         } else if( x > xcd ){
            data.U[0][i] = r3;
            data.p[i]    = p3;
            data.u[0][i] = -u3;
         } else if( x > xsh ){
            data.U[0][i] = r4;
            data.p[i]    = p4;
            data.u[0][i] = -u4;
         } else {
            data.U[0][i] = r5;
            data.p[i]    = p5;
            data.u[0][i] = -u5;
         }
      }
   }

   toConservation( params, data );

   return RET_OK;
}
