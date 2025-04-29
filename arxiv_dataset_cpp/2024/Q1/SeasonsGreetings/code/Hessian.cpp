#include "f2.h"

#include <iostream>
#include <chrono>

int main() {
  using namespace std;
  using namespace std::chrono;
  // increase to compare run times of various approaches
  const int N=240101; // e.g., YYMMDD
  // code to be executed in integer arithmetic
  using T=char;
  // size of x yields squared number of Hessian entries
   T x[N];
  // pairs of custom input values distributed uniformly
  T xv[]={72,97,112,112,121,32,50,48,50,52};
  for (int i=0,j=0;i<N;++i) {
    if ((!(i%(N/5)))&&(j<10)) {
      x[i]=xv[j++];
      x[i+++1]=xv[j++];
    } else {
      x[i]=0;
    }
  }

  // Season's Greetings ...
  cout << "Second-Order Tangent AD wishes ";
  // start time measurement
  auto t_begin=system_clock::now();
  // compute diagonal of the Hessian in tangent of tangent mode
  ddfdxx_t1_t2_sparse<N>(x);
  // finish time measurement
  auto t_end=system_clock::now();
  // report run time 
  cout << "! (taking " 
    << duration_cast<milliseconds>(t_end-t_begin).count()
    << "ms)" << endl;
  cout << "Second-Order Adjoint AD wishes ";
  // start time measurement
  t_begin=system_clock::now();
  // compute diagonal of the Hessian in tangent of adjoint mode
  ddfdxx_a1_t2_compressed<N>(x);
  // finish time measurement
  t_end=system_clock::now();
  // report run time 
  cout << "! (taking " 
    << duration_cast<milliseconds>(t_end-t_begin).count()
    << "ms)" << endl;
  return 0;
}
