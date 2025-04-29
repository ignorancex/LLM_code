#include "train.h"
#include <iostream>

using namespace std;

// This is used instead of main() because we need arguments from the command line.
int main(int argc, char * argv[]) {
  
  //sets the output to whatever locale (number time date format) the person is using.
  cerr.imbue(locale("")); 


  cerr << endl << "===================================" << endl;
  cerr <<         "      ANNz: Network training       " << endl;
  cerr <<         "===================================" << endl << endl;
  
  // Check for minimum number of arguments
  if (argc < 6) {
    cerr << "ERROR: Insufficient arguments"                                                  << endl;
    cerr << "Usage: annz_train <arch_file> <train_file> <valid_file> <out_file> <rand_seed>" << endl;
    return -1;
  }

  // Seed random number generator (used to initialise the weights).
  // srand initializes the random number. 
  // atol converts string to a long number.
  // argv[5] is <rand_seed>
  srand(atol(argv[5]));
    
   
  // Create Training object, which depends on the <architecture file>, <training file> and <validation file>.
  // This part does a lot of things:
  // * It takes the information of the architecture file, and initializes
  //   the number of inputs, outputs, node IDs, number of weights, activations etc
  // * It reads the data set from the training and validating sets, records them,
  // * It finds the rms error of the validation set by finding the sum of squares
  // * It updates the activation values too.
  Training myTraining(argv[1], argv[2], argv[3]);
  
  // Loop until no further iterations are requested.
  // niter is the number of iterations.
  int niter;
  do {
    cout << "Maximum iterations: ";
    cin  >> niter;
    cout << endl;
    
	if (niter == 0) 
	break;
    
	myTraining.train(niter);
  } while (true);
  
  // Save best weight values to file. argv[4] is output filename.
  myTraining.saveNetState(argv[4]);

}

// So the Training class has functions .train(int) and .saveNetState(str).
