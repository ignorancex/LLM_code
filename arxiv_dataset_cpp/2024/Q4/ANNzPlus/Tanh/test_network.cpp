#include "test_network.h"
#include "util.h"

#include <fstream>
#include <iostream>

// As Network::pass(), but normalises the inputs before passing, and renormalises outputs before returning.
void TestNetwork::norm_pass(std::vector<double> inputs, std::vector<double>& outputs, 
			    std::vector< std::vector<double> >& jacobian) {
  
  // Renormalise inputs.
  for (int i = 0; i < nInputs; ++i) {
    inputs[i] = (inputs[i] - meanIn[i])/sigmaIn[i];
  }
  
  // Pass through network, to obtain outputs.
  pass(inputs, outputs);
  
  // Renormalise outputs.
  for (int n = 0; n < nOutputs; ++n) {
    outputs[n] = outputs[n] * sigmaOut[n] + meanOut[n];
  }
  
  // Update Jacobian in order to allow propagation of input errors.
  updateJacobian();
  
  jacobian.assign(nOutputs, std::vector<double>(nInputs, 0.0));
  
  for (int j = 0; j < nOutputs; ++j) {
    for (int i = 0; i < nInputs; ++i) {
      // Copy and renormalise Jacobian.
      jacobian[j][i] = sigmaOut[j]/sigmaIn[i] * nodes[outputLayer[j]]->jacob[i];
    }
  }

}

//------------------------------------------------------------------------------
// Restore network weights from file.
bool TestNetwork::restore_wts(const char file_name[]) {
  
  // Consistency checks.
  bool wts = false;
  int inNorm = 0, outNorm = 0;
  
  // Attach input stream to network file.
  std::ifstream net_file(file_name);
  
  // Storage.
  std::string buffer, flag;
  std::vector<std::string> data;
  
  // Read a line at a time from net file.
  while (annz_util::get_next_dataline(net_file, buffer)) {
    
    // Extract flag and data from input line.
    annz_util::parse_param_line(buffer, flag, data);
    
    //-------------------------------------------
    // Identify flag.
    if (flag == "INORM") {
      //--------------------------------------------
      // Input normalisation.
      if (data.size() != 3) {
	annz_util::file_error(file_name, buffer);
	return false;
      }
      
      int ix = annz_util::string_to_int(data[0]);
      if (meanIn.size() <= ix)
	meanIn.resize(ix + 1);
      
      meanIn[ix] = annz_util::string_to_double(data[1]);

      if (sigmaIn.size() <= ix)
	sigmaIn.resize(ix + 1);

      sigmaIn[ix] = annz_util::string_to_double(data[2]);
      inNorm++;
      
    } else if (flag == "ONORM") {
      //--------------------------------------------
      // Output normalisation.
      if (data.size() != 3) {
	annz_util::file_error(file_name, buffer);
	return false;
      }
      
      int ix = annz_util::string_to_int(data[0]);
      if (meanOut.size() <= ix)
	meanOut.resize(ix + 1);
      
      meanOut[ix] = annz_util::string_to_double(data[1]);
      
      if (sigmaOut.size() <= ix)
	sigmaOut.resize(ix + 1);
      
      sigmaOut[ix] = annz_util::string_to_double(data[2]);
      outNorm++;
      
    } else if (flag == "WEIGHTS") {
      //----------------------------------------------
      // Weights.
      if (data.size() != 1) {
	annz_util::file_error(file_name, buffer);
	return false;
      }
      
      // Read number of weights.
      int nwts = annz_util::string_to_int(data[0]);

      // Read weights from file.
      int f,t;
      double v;
      for (int i = 0; i < nwts; ++i) {
	annz_util::skip_comments(net_file);
	
	if (net_file >> f >> t >> v) {
	  weights[wt_lu[t][f]]->wt = v;
	} else {
	  annz_util::file_error(file_name, buffer);
	  return false;
	}
      }
      
      wts = true;
      
    } else {
      // Flag not recognised.
      // Quietly ignore, since this will always happen, 
      // when the network architecture is read
      // provided all the normalisation and weight info is present, 
      // we don't care what other junk may be in the file.    
    }
  }

  //----------------------------------------------------
  // Check that all required info was found.
  if (!wts) {
    std::cerr << "ERROR: network weights not found" << std::endl;
    return false;
  } else if ((inNorm != nInputs) || (outNorm != nOutputs)) {
    std::cerr << "ERROR: normalisation specification inconsistent with architecture" << std::endl;
    return false;
  }

  return true;
}

