#include "PDAG.h"

PDAG::PDAG(size_t nodes,StringVector node_names,
           NumericMatrix adj,bool verbose) :
  Graph::Graph (nodes,node_names,adj,verbose)
  {}

// Makes a graph without any edges
PDAG::PDAG(size_t nodes,bool verbose) : 
  Graph::Graph(nodes,verbose) {
    emptyGraph();
  }

// Returns the parents, children, and spouses of node i along with undirected edges
// Differs from Graph::getAdjacent b/c it returns spouse nodes
// Differs from DAG::getNeighbors b/c it returns undirected edge neighbors
NumericVector PDAG::getNeighbors(const size_t &i,bool verbose){
  validateIndex(i);
  NumericVector neighbors;
  NumericVector parents;
  NumericVector children;
  NumericVector undirected;
  size_t p = Graph::size();
  if (verbose){
    Rcout << "FUNCTION PDAG::getNeighbors. Node " << i << std::endl;
  }
  // Obtain parents and children of i
  for (size_t j=0;j<p;++j){
    if (amat(j,i)==1 && amat(i,j)==0){ 
      parents.push_back(j);
      if (verbose){
        Rcout << "Call from PDAG::getNeighbors. Node " << j << " is a parent.\n";
      }
    } else if (amat(i,j)==1 && amat(j,i)==0){
      children.push_back(j);
      if (verbose){
        Rcout << "Call from PDAG::getNeighbors. Node " << j << " is a child.\n";
      }
    } else if (amat(i,j)==1 && amat(j,i)==1){
      undirected.push_back(j);
      if (verbose){
        Rcout << "Call from PDAG::getNeighbors. Node " << j << " is a neighbor (undirected).\n";
      }
    }
  }
  // Create vector to store parents U children U undirected neighbors
  neighbors = union_(parents,children);
  neighbors = union_(neighbors,undirected);
  
  // Find nonadjacent common parents with i (spouses)
  NumericVector spouses;
  size_t current_val;
  size_t other_val;
  for (size_t child : children ){
    if (verbose){
      Rcout << "Call from PDAG::getNeighbors. ";
      Rcout << "We are evaluating the following child: " << child << std::endl;
    }
    for (size_t j=0;j<p;++j){
      current_val = amat(j,child);
      other_val = amat(child,j);
      // We don't check for adjacency b/c those nodes will be picked up
      // in the parent or the child sets already.
      if (current_val == 1 && other_val ==0 && i != j){
        spouses.push_back(j);
        if (verbose){
          Rcout << "Call from PDAG::getNeighbors. Node " << j;
          Rcout << " is a spouse of node " << i << ".\n";
        }
      }
    }
  }
  neighbors = union_(neighbors,spouses);
  
  
  std::sort(neighbors.begin(),neighbors.end());
  if (verbose){
    Rcout << "Neighbors of node " << i;
    printVecElementsNoNames(neighbors,": ","",", ");
    Rcout << "\n\n";
  }
  
  return neighbors;
}

// Retrieves the neighbors of all the target nodes, and excludes the target nodes
// themselves for the final result.
NumericVector PDAG::getNeighborsMultiTargets(const NumericVector &targets,
                                             bool verbose){
  NumericVector neighbors;
  
  for (auto t : targets){
    validateIndex(t);
    if (verbose){
      Rcout << "Target: " << t << std::endl;
    }
    neighbors = union_(neighbors,getNeighbors(t,verbose));
  }
  
  // Remove target nodes from final vector
  neighbors = setdiff(neighbors,targets);
  std::sort(neighbors.begin(),neighbors.end());
  if (verbose){
    Rcout << "Total Neighborhood:\n";
    printVecElementsNoNames(neighbors,"","\n",", ");
  }
  
  return neighbors;
}

// Returns true if i and j are in the same neighborhood 
// (i.e. one is the parent of another, or they share a common child)
// for this function, we consider a node in the same nbhd as itself
bool PDAG::inNeighborhood(const size_t &i,
                          const size_t &j){
  validateIndex(i); validateIndex(j);
  // We consider a node in the same neighborhood as itself
  if (i==j){
    return true;
  }
  // This checks for basic adjacency
  if (Graph::areAdjacent(i,j)){
    return true;
  }
  // Check for spouses 
  
  // Find children of both nodes
  size_t p = Graph::size();
  bool parent_i; bool parent_j;
  // Determining if i and j are spouses
  // Don't need to check for adjacency since that has already been checked
  for (size_t k = 0;k<p;++k){
    parent_i = amat(i,k)==1 && amat(k,i)==0;
    parent_j = amat(j,k)==1 && amat(k,j)==0;
    if (parent_i && parent_j){
      return true;
    }
  }
  return false;
}



