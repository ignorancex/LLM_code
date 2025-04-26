#include <cstdio>
#include <cstdlib>
#include <cmath>

double rk3_step( double t
               , double un
               , double dt
               , double (*f)(double,double) );
double euler_step( double t
                 , double un
                 , double dt
                 , double (*f)(double,double) );

int main
   ( int argc
   , char *argv[]
){
   if( argc != 2 ){
      printf( "Usage: %s steps\n", argv[0] );
      return 0;
   }

   const auto nmax = std::atoi( argv[1] );
   if( nmax <= 0 ){
      printf( "Error: number of steps has to be an integer greater than zero.\n" );
      return 0;
   }

   const auto tmax  = int{1};
   const auto dt    = double{tmax/(double)nmax};

   auto ur = double{1.0};
   auto ue = double{1.0};
   printf( "# n t euler rk3\n" );
   printf( "%d %.6f %.6e %.6e\n", 0, 0.0, ue, ur );

   for( auto n = 1; n <= nmax; n++ ){
      const auto t = dt*n;
      ur = rk3_step( t, ur, dt, []( auto t, auto u ){ return u; } );
      ue = euler_step( t, ue, dt, []( auto t, auto u ){ return u; } );
      printf( "%d %.6f %.6e %.6e\n", n, t, ue, ur );
   }

   const auto exact = exp(1.0);
   printf( "\n# [Stats]\n" );
   printf( "# nmax err_euler err_rk3\n" );
   printf( "# %d %.8e %.8e\n", nmax, std::fabs(exact-ue), std::fabs(exact-ur) );

   return 0;
}

double euler_step
   ( double t
   , double un
   , double dt
   , double (*f)(double,double)
){
   const auto unp1 = un + dt*f(t,un);

   return unp1;
}

double rk3_step
   ( double t
   , double un
   , double dt
   , double (*f)(double,double)
){
   const auto u1   = un + dt*f(t,un);
   const auto u2   = 3.0/4.0 * un + 1.0/4.0 * u1 + 1.0/4.0 * dt * f(t,u1);
   const auto unp1 = 1.0/3.0 * un + 2.0/3.0 * u2 + 2.0/3.0 * dt * f(t,u2);

   return unp1;
}
