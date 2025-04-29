#include "test.h"

#include <iostream>

using namespace std;

int main(int argc, char * argv[]) {

  // Status flag.
  int status = 0;

  cerr << "===================================" << endl;
  cerr << "    ANNz: Photo-z determination    " << endl;
  cerr << "===================================" << endl;

  //==========================================
  // Process arguments.
  //==========================================
  
  // Check for minimum number of arguments
  if (argc < 4) {
    cerr << "ERROR: Insufficient arguments" << endl;
    cerr << "Usage: annz_test <input_cat> <output_file> <wts_file1> [<wts_file2> ...]" << endl;
    return -1;
  }
  
  // Determine committee size from number of wts files provided.
  int n_nets = argc - 3;
  
  //------------------------------------------------
  // Copy file net file names to new array.
  //------------------------------------------------
  char* wts_file[n_nets];
  for (int ii = 0; ii < n_nets; ++ii) {
    wts_file[ii] = argv[3 + ii];
  }

  // Create Testing object.
  Testing myTesting(n_nets, wts_file);
  
  //std::cerr << "Created testing object." << std::endl;

  // Pass testing set to committee.
  myTesting.test(argv[1], argv[2]);

}
