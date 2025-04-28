/*
 *  PIXON
 *  A Pixon-based method for reconstructing velocity-delay map in reverberation mapping.
 * 
 *  Yan-Rong Li, liyanrong@mail.ihep.ac.cn
 * 
 */
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <random>
#include <nlopt.hpp>
#include <fftw3.h>

#include "proto.hpp"

using namespace std;

int main(int argc, char ** argv)
{
  Config config; 
  /* if input param file */
  if(argc >= 2)
  {
    config.load_cfg(argv[1]);
  }
  else 
  {
    cout<<"Please input param file."<<endl;
    exit(0);
  }
  
  config.print_cfg();
  cout<<"Pixon basis type: "<<config.pixon_basis_type<<", "<<PixonBasis::pixonbasis_name[config.pixon_basis_type]<<endl;
  
  run(config);

  return 0;
}