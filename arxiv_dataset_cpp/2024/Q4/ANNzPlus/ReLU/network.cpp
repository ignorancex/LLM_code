//---------------------------------------------------------
// Definition of methods for Network and derived classes.
//----------------------------------------------------------

#include "network.h"
#include "util.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

//==================================================================
// Network operations.
//==================================================================

//---------------------------------------------------------------
// Set the network inputs to the given values, and propagate through the network.
void Network::pass(const std::vector<double>& inputs, std::vector<double>& outputs) {
  setInputs(inputs);
  updateActivations();
  outputs = getOutputs();
}

//--------------------------------------------------------------------------
// Update activations of all nodes, to reflect new inputs/weights.
void Network::updateActivations() {

  // Initialise feeding layer to be the inputs.
  // feed is a vector of integers, contains list of node IDs of the inputLayer.
  Layer feed = inputLayer;

  // Update activations for each hidden layer in turn (feeding forward).
  for (int l = 0; l < hiddenLayer.size(); ++l) {
      
    // Scan over nodes in current hidden layer.
    for (int n = 0; n < hiddenLayer[l].size(); ++n) {

      // Store node ID.
      int node = hiddenLayer[l][n];

      // Compute activity by summing over feeding layer's nodes: w_ji * z_i.
      // weights is a Connection vector (to, from, wt).
      // nodes is a Node vector (activation, delta).
      double activity = 0.0;
      for (int f = 0; f < feed.size(); ++f) {
	    activity += weights[(wt_lu[node][feed[f]])]->wt * nodes[feed[f]]->activation;
      }

      // Add on bias (which always has unit activation).
      activity += weights[(wt_lu[node][BIAS_ID])]->wt;

      // Compute activation of this node (sigmoid for hidden nodes).
      nodes[node]->activation = logistic(activity);

    }

    // This layer now becomes the feeder for the next.
    feed = hiddenLayer[l];
  }

  // Output layer.
  for (int n = 0; n < outputLayer.size(); ++n) {

    // Store node ID.
    int node = outputLayer[n];

    // Compute activity by summing over previous layer's nodes: w_ji * z_i.
    double activity = 0.0;
    for (int f = 0; f < feed.size(); ++f) {
      activity += weights[(wt_lu[node][feed[f]])]->wt * nodes[feed[f]]->activation;
    }

    // Bias.
    activity += weights[(wt_lu[node][BIAS_ID])]->wt;

    // Compute activation of this node (linear for output nodes).
    nodes[node]->activation = activity;
  }
  
}


// Compute Jacobian at each node.
// This is the derivative of the activation of the node wrt each of the inputs.
// Assumes activations have already been updated for this input vector.
void Network::updateJacobian() {
  
  // Start with inputs for which dz/dx is just kronecker's delta.
  for (int j = 0; j < nInputs; ++j) {
    for (int i = 0; i < nInputs; ++i) {
      nodes[inputLayer[j]]->jacob[i] = (j==i ? 1 : 0);
    }
  }

  // Hidden layers.
  Layer feed = inputLayer;
  for (int l = 0; l < nLayers; ++l) {
    
    // Compute for each node in this layer.
    for (int j = 0; j < nHidden[l]; ++j) {
      int node = hiddenLayer[l][j]; // ID of node currently being computed for.
      
      // Compute for each input in turn.
      for (int i = 0; i < nInputs; ++i) {
	// Sum over feeding nodes.	
	double sum = 0.0;
	for (int f = 0; f < feed.size(); ++f) {
	  sum += weights[wt_lu[node][feed[f]]]->wt * nodes[feed[f]]->jacob[i];
	}
	
	double a = nodes[node]->activation; // Get node's activation.
	// jacob[i] = g'(a) * sum[wt * jacob] over feeding nodes.
	double max_dev;
	{
		if (a>0)
		{
	 	max_dev=1;
		}
		else 
		{
		max_dev=0;
		}
       }
	nodes[node]->jacob[i] = max_dev * sum;
      }
    }
    feed = hiddenLayer[l]; // This layer becomes the feeder for the next.
  }

  // Output layer.
  for (int j = 0; j < nOutputs; ++j) {
    // Compute for each input in turn.
      for (int i = 0; i < nInputs; ++i) {
	// Sum over feeding nodes.	
	double sum = 0.0;
	for (int f = 0; f < feed.size(); ++f) {
	  sum += weights[wt_lu[outputLayer[j]][feed[f]]]->wt * nodes[feed[f]]->jacob[i];
	}
	// g'(a) * sum[wt * jacob] over feeding nodes, with g'(a) = 1.
	nodes[outputLayer[j]]->jacob[i] = sum; 
      }
  }
  
}



// Compute delta at each node.
// Note that this implementation assumes a sum-of-squares cost function.
// Assumes activations have already been computed for this input vector.
void Network::updateDeltas(const std::vector<double>& trues) {

  // Compute deltas in output layer.
  // Assumes linear activation function.
  // delta = 2(activation - spectros)
  for (int n = 0; n < outputLayer.size(); ++n) {
    nodes[outputLayer[n]]->delta = 2.0 * (nodes[outputLayer[n]]->activation - trues[n]);
  }
  
  // This is the layer to which the current one sends connections. 
  Layer fed = outputLayer;

  // Scan hidden layers, starting at output end.
  for (int l = hiddenLayer.size() - 1; l >= 0; --l) {
    for (int n = 0; n < hiddenLayer[l].size(); ++n) {
      
	  int node      = hiddenLayer[l][n];
      double sumdel = 0.0;
      
	  for (int f = 0; f < fed.size(); ++f) {
      	sumdel += nodes[fed[f]]->delta * weights[wt_lu[fed[f]][node]]->wt;
      }
      
	  double a           = nodes[node]->activation;
	  double max_dev;
	{
		if (a>0)
		{
	 	max_dev=1;
		}
		else 
		{
		max_dev=0;
		}
       }

      sumdel            *= max_dev; // Derivative of logistic activation function.
      nodes[node]->delta = sumdel;
    }
    fed = hiddenLayer[l]; // This layer is now the 'fed' layer for the previous one.
  }

}

//=======================================================
// Network initialisation.
//=======================================================

//------------------------------------------------------------------------------
// Constructor. Sets up nodes and connections. Weights remain uninitialised.
Network::Network(const char file_name[]) {

  // Restore architecture from file.
  restore_arch(file_name);
  
  // Create and connect the network nodes; 
  // setup the weights vector and lookup table.
  setup(nInputs, nOutputs, nLayers, nHidden);
}

//-------------------------------------------------------------------------------
// Create and connect network nodes given architecture specification. This function updates the values
// inputLayer, hiddenLayer and outputLayer (all vectors).
void Network::setup(int nIn, int nOut, int nLay, std::vector<int> nHid) {
  
  //-----------------------------------------------------
  // Create nodes, and assign to layers.
  // cerr << "Creating nodes and assigning to layers" << endl;
  
  int nextID = 0; // Keeps count of the number of nodes created.

  nodes.push_back(new Node(1.0)); // Bias node. Here nodes is a vector of Node class.
  nextID++;
  
  // Create input layer nodes. inputLayer is a vector<int>.
  for (int i = 0; i < nIn; ++i) {
    inputLayer.push_back(nextID++); // Add node ID to inputLayer.
    nodes.push_back(new Node()); // Create node.
  }

  // Track the number of weights.
  int wtCount = 0;

  // Number of nodes feeding into next layer. Should be 5 at this point.
  int feeders = inputLayer.size();

  // Create hidden Layers.
  for (int l = 0; l < nLay; ++l) {
    std::vector<int> newLayer;
    // Create nodes for lth hidden layer.
    for (int j = 0; j < nHid[l]; ++j) {
      newLayer.push_back(nextID++);
      nodes.push_back(new Node());
      wtCount += feeders + 1; // hidden layer 1 wts: 6 each, hidden layer 2 wts: 11 each.
    }
    hiddenLayer.push_back(newLayer); // Add new layer to hiddenLayer vector.
    feeders = hiddenLayer[l].size(); // This layer becomes the feeder for the next.
  }
  
  // Create output layer.
  for (int i = 0; i < nOut; ++i) {
    outputLayer.push_back(nextID++);
    nodes.push_back(new Node());
    wtCount += feeders + 1;
  }
  
  // So at this point, we have
  // (1) a vector nodes, with elements of uninitialized Nodes class for each node
  // (2) inputLayer (a vector), with elements each having a specific ID number
  // (3) hiddenLayer (a vector of vectors), elements of each hidden layer having their own ID numbers
  // (4) outputLayer (a vector), same like inputLayer
  // (5) weight count, which is like quite a lot man...

  //cerr << "Created " << nodes.size() << " nodes" << endl;
  //----------------------------------------------------------------
  // Make connections.
  //cerr << "Connecting nodes; creating weights vector and lookup table." << endl;
  
  // Ensure weights vector is clear and initialise weights lookup table.
  // weights is a vector of Connections, which has 3 numbers (from, to and weight value).
  weights.clear();    
  // wt_lu is a vector of the size of 27 (no. of nodes), each element is a vector of 27 elements with value -1.
  // In other words, it is a matrix of nodes x nodes.
  wt_lu = std::vector< std::vector<int> >(nodes.size(), std::vector<int>(nodes.size(), -1));

  // Input layer feeds connections to first hidden layer.
  // feed is a vector (1,2,3,4,5) (5 inputs) in the example.
  Layer feed = inputLayer; 
  
  // Track number of weights created.
  int wtix = 0; 
  
  // Scan over hidden layers. 2 hidden layers in the example.
  for (int l = 0; l < hiddenLayer.size(); ++l) {
    
    // Scan over nodes in this hidden layer. 10 nodes each layer.
    for (int j = 0; j < hiddenLayer[l].size(); ++j) {

      int myID = hiddenLayer[l][j]; // Note current node ID for convenience.

      // Connection to bias node. Connection(to,from)
      // From what I understand, the bias node connects to every single node in the layers!
	  weights.push_back(new Connection(myID, 0)); 
      wt_lu[myID][0] = wtix++;
      
      // Scan over feeding layer.
      // This process creates a weight connection from nodes of each previous layer to this layer.
      for (int f = 0; f < feed.size(); ++f) {
      	weights.push_back(new Connection(myID, feed[f]));
      	wt_lu[myID][feed[f]] = wtix++;
      }
    }

    feed = hiddenLayer[l]; // This layer now becomes the feeder for the next.
  }

  // Scan over output layer.
  for (int j = 0; j < outputLayer.size(); ++j) {
    int myID = outputLayer[j];
    
    weights.push_back(new Connection(myID, 0)); // Connection to bias node.
    wt_lu[myID][0] = wtix++; 
    
    // Scan over feeding layer,
    for (int f = 0; f < feed.size(); ++f) {
      weights.push_back(new Connection(myID, feed[f]));
      wt_lu[myID][feed[f]] = wtix++;
    }
  }
  // So at this point, we have the addition of
  //(6) a weights vector of connections, which has to, from, and weight value for each weight;
  // weight values are zero at the moment...
}



//======================================================================
// File reading, for network setup.
//======================================================================

//------------------------------------------------------------------
// Restore network architecture from description in given file.
// This could be either a .net file (output from annz_net) or a 
// .wts file (output from annz_train). It will return true, and
// update the values of nInputs, nOutputs, nLayers and nHidden.
bool Network::restore_arch(const char file_name[]) {
  
  // For completeness check.
  bool ins = false, outs = false, lays = false;
  
  // Attach input stream to network file.
  std::ifstream net_file(file_name);
     
  // Temporary storage.
  std::string buffer, flag;
  std::vector<std::string> data;
   


  //------------------------------------------------
  // Read a line at a time from net file.
  while (annz_util::get_next_dataline(net_file, buffer)) {
    
    // Extract flag and data from input line. From now on flag and data have values!
    annz_util::parse_param_line(buffer, flag, data);
    //std::cerr << ins << outs << lays << " ";

    //-------------------------------------------
    // Identify flag.
    if (flag == "N_INPUTS") {
      //------------------------------------------
      // Number of inputs.
      // to ensure that the data has one and only one value
      if (data.size() != 1) {
	annz_util::file_error(file_name, buffer);
	return false;
      }
      
      nInputs = annz_util::string_to_int(data[0]);
      ins     = true;

    } else if (flag == "N_OUTPUTS") {
      //-------------------------------------------
      // Number of outputs.      
      if (data.size() != 1) {
	annz_util::file_error(file_name, buffer);
	return false;
      }
      
      nOutputs = annz_util::string_to_int(data[0]);
      outs     = true;
      
    } else if (flag == "N_LAYERS") {
      //--------------------------------------------
      // Hidden layer specifications. N_LAYERS refer to the number of hidden layers.
      nLayers = annz_util::string_to_int(data[0]);
      
      if (data.size() != nLayers + 1) {
	  annz_util::file_error(file_name, buffer);
	  return false;
	  }
      
      // add the number of nodes in the respective hidden layers to the data vector.
      for (int i = 1; i <= nLayers; ++i)
	  nHidden.push_back(annz_util::string_to_int(data[i]));
       
      lays = true;
      
    } else {
      // Flag not recognised.
      // Quietly ignore, since this will happen whenever a .wts file is read -- provided 
      // the architecture is fully specified we don't care what other junk may be in the file.
    }
  }
  
  // Check that all required info was found.
  if (!ins || !outs || !lays) {
    std::cerr << "ERROR: architecture specification incomplete" << std::endl;
    return false; 
  } else {
    return true;
  }
  
}

//=======================================================
// Output ANN information.
//========================================================

//--------------------------------------------
// Dump weights to given output stream.
void Network::dump_weights(std::ostream& out) {
  out << "WEIGHTS " << weights.size() << std::endl;
  for (int i = 0; i < weights.size(); ++i)
    out << weights[i]->from << " " << weights[i]->to << " " << weights[i]->wt << std::endl;
}

//-----------------------------------------------------
// Dump network architecture to given output stream.
void Network::dump_arch(std::ostream& out) {
  out << "N_INPUTS "  << nInputs  << std::endl;
  out << "N_OUTPUTS " << nOutputs << std::endl;
  out << "N_LAYERS "  << nLayers;

  for (int i = 0; i < nLayers; ++i)
  out << " " << nHidden[i];
  
  out << std::endl;
}
