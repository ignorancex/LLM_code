#include "test.h"
#include "util.h"

#include <iostream>
#include <fstream>
#include <iomanip>

//------------------------------------------------------------
// Committee class methods.
//------------------------------------------------------------

//------------------------------------------------------------------------------
// Create a new committee.
Committee::Committee(int n_nets, char** net_files) : n_nets(n_nets){
    
  // Instantiate networks from files.
  for (int i = 0; i < n_nets; ++i) {
    nets.push_back(new TestNetwork(net_files[i]));
  }
    
  // Check committee members all agree on input and output numbers.
  nInputs = nets[0]->getNInputs();
  nOutputs = nets[0]->getNOutputs();

  for (int i = 1; i < n_nets; ++i) {
    if ((nets[i]->getNInputs() != nInputs) || (nets[i]->getNOutputs() != nOutputs)) {
      std::cerr << "ERROR: Committee members incompatible ";
      std::cerr << "(must agree on numbers of inputs and outputs)." << std::endl;
    }
  } 
}

//----------------------------------------------------------------------------------------
// Present given input vector to the committee.
void Committee::pass(const std::vector<double> & inputs, 
		     const std::vector<double> & errors, 
		     Result & result) {
    
  // Storage for results from the network.
  std::vector< std::vector< std::vector<double> > > jacobian(n_nets, std::vector< std::vector<double> >());
  std::vector< std::vector<double> > outputs(n_nets, std::vector<double>());

  // Present to each network in turn.
  for (int i = 0; i < n_nets; ++i) {
    nets[i]->norm_pass(inputs, outputs[i], jacobian[i]);
  }
    
  // Compute mean of each output.
  result.outputs.assign(nOutputs, 0.0);
  for (int j = 0; j < nOutputs; ++j) {
    for (int i = 0; i < n_nets; ++i) {
      result.outputs[j] += outputs[i][j];
    }
    result.outputs[j] /= n_nets;
  }

  // Errors...
  std::vector<double> sigma_netvar(nOutputs, 0.0), sigma_noise(nOutputs, 0.0);
  result.errors.assign(nOutputs, 0.0);

  // Compute network variance.
  if (n_nets > 2) {
    for (int j = 0; j < nOutputs; ++j) {
	
      for (int k = 0; k < n_nets; ++k)
	sigma_netvar[j] += outputs[k][j] * outputs[k][j]; // Sum squares.
	
      // Calculate standard error.
      sigma_netvar[j] = n_nets/(n_nets - 1.0) *
	(sigma_netvar[j]/n_nets - result.outputs[j] * result.outputs[j]);
    }
  }

  // Compute noise at outputs.
  for (int j = 0; j < nOutputs; ++j) {
    for (int i = 0; i < nInputs; ++i) {
	
      // Average Jacobian over committee members.
      double Javg = 0.0;
      for (int k = 0; k < n_nets; ++k)
	Javg += jacobian[k][j][i];
      Javg /= n_nets;
	
      // Sum (dm * J)^2 -- combine contributions from each input in quadrature.
      sigma_noise[j] += pow(Javg * errors[i], 2);
    }

    // Save combined error to Result struct.
    result.errors[j] = sqrt(sigma_noise[j] + sigma_netvar[j]);
  }
}



//================================================================================
// Testing class methods.
//=================================================================================

//--------------------------------------------------------------------------------------
// Constructor.
Testing::Testing(int n_nets, char** net_files) 
  : myCommittee(n_nets, net_files), n_nets(n_nets) {
        
  std::cerr << "Using a committee of " << n_nets << " networks." << std::endl;
  if (n_nets < 3)
    std::cerr << "WARNING: Error estimates not sensible with fewer than 3 networks in committee." << std::endl;
    
  std::cerr << " # of inputs  : " << (nInputs = myCommittee.nInputs) << std::endl;
  std::cerr << " # of outputs : " << (nOutputs = myCommittee.nOutputs) << std::endl;    
}

//-------------------------------------------------------------
// Present given testing set to committee.
bool Testing::test(char test_file[], char out_file[]) {
    
  // Open test data file and output file.
  std::ifstream test_file_stream(test_file);
  std::ofstream output_file(out_file);

  // Temporary buffer for reading into.
  std::string buffer;
  
  // True if spectroscopic evaluation values are provided.
  bool specs;
  
  // Data.
  std::vector<double> inputs, errors, trues;

  // Accuracy stats.
  std::vector<double> sum(nOutputs, 0.0), sum_sq(nOutputs, 0.0);

  // Number of records in test data.
  int nObjects = 0;
    
  // Result struct.
  Committee::Result myResult;
  
  // Scan through test data.
  do {
      
    // Get next non-blank line.
    do {
      annz_util::skip_comments(test_file_stream);
      getline(test_file_stream, buffer);
    } while ((buffer.find_first_not_of(' ') == std::string::npos) && (!test_file_stream.fail()));
    
    // Exit if we've reached end of file.
    if (test_file_stream.fail())
      break;
    
    // Extract data from string.
    bool thisSp;
    if (!annz_util::parse_input_data(buffer, nInputs, nOutputs, thisSp, inputs, errors, trues))
      return false;
    
    // Check for consistency of specs.
    if (nObjects == 0) {
      specs = thisSp;
      if (specs == true)
	std::cerr << "Found spectroscopic redshifts." << std::endl;
      else
	std::cerr << "Spectroscopic redshifts not present." << std::endl;
      
      // Putting this here provides a more logical flow of reporting to the user.
      std::cerr << "Calculating photometric redshifts..." << std::endl;

    } else if (thisSp != specs) {
      std::cerr << "ERROR: Inconsistent data: must provide z_spec for all or none." << std::endl;
      return false;
    }
      
    nObjects++; // Track sample size.

    // Present case to the committee.
    myCommittee.pass(inputs, errors, myResult);

    // Output results to file.
    for (int n = 0; n < nOutputs; ++n) {
      if (specs)
	output_file << trues[n] << " ";
      output_file << myResult.outputs[n] << " " << myResult.errors[n] << " ";
    }
    output_file << std::endl;
  
    if (specs) {
      // Tally stats for accuracy reporting.
      for (int j = 0; j < nOutputs; ++j) {
	double dif = myResult.outputs[j] - trues[j];
	sum[j] += dif;
	sum_sq[j] += dif * dif;
      }
    }

  } while (true);
  
  std::cerr << "Sample size: " << nObjects << std::endl;
  
  //-----------------------------------------------------------------------
  // Report accuracy stats.
  //-----------------------------------------------------------------------
  std::cerr << "================================================" << std::endl;
  if (specs) {
    for (int i = 0; i < nOutputs; ++i) {
      std::cerr << "Output " << i + 1 << ": Mean: " << std::setprecision(3) << sum[i]/nObjects;
      std::cerr << " rms: " << std::setprecision(6) << sqrt(sum_sq[i]/nObjects) << std::endl;
    }
  }
  std::cerr << "Photometric redshifts saved to \"" << out_file << "\"" << std::endl;
  std::cerr << "================================================" << std::endl;
  
  return true;

}
