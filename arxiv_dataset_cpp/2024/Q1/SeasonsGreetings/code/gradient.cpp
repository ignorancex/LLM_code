#include "f1.h"

#include <iostream>
#include <chrono>

int main() {
  using namespace std;
  using namespace std::chrono;
  // increase to experience superior run time of adjoint
  const int N=231224; // e.g., YYMMDD; required to be even
  // code to be executed in integer arithmetic
  using T=char;
  // size of x equal to size of gradient
  T x[N];
  // pairs of custom input values distributed uniformly
  T xv[]={101,77,114,114,32,121,109,88,115,97};
  for (int i=0,j=0;i<N;++i) {
    if ((!(i%(N/5)))&&(j<10)) { 
      x[i]=xv[j++];
      x[i+++1]=xv[j++];
    } else {
      x[i]=0;
    }
  }
  // Season's Greetings by Tangent AD ...
  cout << "Tangent AD wishes ";
  // start time measurement
  auto t_begin=system_clock::now();
  // compute gradient in tangent mode
  dfdx_t1<N>(x);
  // finish time measurement
  auto t_end=system_clock::now();
  // report run time
  cout << "! (taking "
    << duration_cast<milliseconds>(t_end-t_begin).count()
    << "ms)" << endl;
  // Season's Greetings by Tangent AD ...
  cout << "Adjoint AD wishes ";
  // start time measurement
  t_begin=system_clock::now();
  // compute gradient in adjoint mode
  dfdx_a1<N>(x);
  // finish time measurement
  t_end=system_clock::now();
  // report run time
  cout << "! (taking "
    << duration_cast<milliseconds>(t_end-t_begin).count()
    << "ms)" << endl;
  return 0;
}
