#include "DAG.h"

DAG::DAG(size_t nodes,StringVector node_names,
         NumericMatrix adj,bool verbose) :
  Graph::Graph (nodes,node_names,adj,verbose)
  {}

// Makes a graph without any edges
DAG::DAG(size_t nodes,bool verbose) : 
  Graph::Graph(nodes,verbose) {
    emptyGraph();
}

// Helpers for function determining acyclicity
// Adds nodes to v which have no incoming edges in the DAG
void getNonIncidentNodes(DAG* d,std::vector<size_t> &v){
  size_t p = d->size();
  for (size_t j=0;j<p;++j){
    if (is_true(all(d->getAmatCol(j)==0))){
      v.push_back(j);
    }
  }
}

// Returns true if there are no edges in adj. matrix A
bool checkAllZeros(NumericMatrix &A){
  size_t p = A.nrow();
  for (size_t i=0;i<p;++i){
    for (size_t j=0;j<p;++j){
      if (A(i,j)!=0){
        return false;
      }
    }
  }
  return true;
}

// Kahn's algorithm for a topological sort
// If the graph has a topological sort, then there are no cycles
// Otherwise, the graph has at least one cycle.
bool DAG::isAcyclic(){
  size_t p = size();
  size_t n; // takes the value of a node w/o an incoming edge
  if (p==0){
    return true;
  }
  NumericMatrix tmp = clone(getAmat());
  std::vector<size_t> L; // records checked nodes
  std::vector<size_t> S; // records nodes without incoming edges
  
  getNonIncidentNodes(this,S);
  
  while (!S.empty()){
    n = S.back();
    S.pop_back();
    L.push_back(n);
    // Remove all edges from n to another node
    for (size_t i=0;i<p;++i){
      if (tmp(n,i)==1){
        tmp(n,i)=0;
        // if i has no other incoming edges, insert into S
        if (is_true(all(tmp(_,i)==0))){
          S.push_back(i);
        }
      }
    }
  }
  // If there are still edges, then there is a cycle
  return checkAllZeros(tmp);
}

NumericVector DAG::getParents(const size_t &i){
  validateIndex(i);
  size_t p = Graph::size();
  NumericVector parents;
  for (size_t j=0;j<p;++j){
    if (amat(j,i)==1 && amat(i,j)==0){
      parents.push_back(j);
    }
  }
  return parents;
}

NumericVector DAG::getChildren(const size_t &i){
  validateIndex(i);
  size_t p = Graph::size();
  NumericVector children;
  for (size_t j=0;j<p;++j){
    if (amat(i,j)==1 && amat(j,i)==0){
      children.push_back(j);
    }
  }
  return children;
}

// Returns the parents, children, and spouses of node i
// Differs from Graph::getAdjacent b/c it returns spouse nodes
NumericVector DAG::getNeighbors(const size_t &i,bool verbose){
  validateIndex(i);
  NumericVector neighbors;
  NumericVector parents;
  NumericVector children;
  size_t p = Graph::size();
  if (verbose){
    Rcout << "FUNCTION DAG::getNeighbors. Node " << i << std::endl;
  }
  // Obtain parents and children of i
  for (size_t j=0;j<p;++j){
    if (amat(j,i)==1){ 
      parents.push_back(j);
      if (verbose){
        Rcout << "Call from DAG::getNeighbors. Node " << j << " is a parent.\n";
      }
    } else if (amat(i,j)==1){
      children.push_back(j);
      if (verbose){
        Rcout << "Call from DAG::getNeighbors. Node " << j << " is a child.\n";
      }
    }
  }
  // Create vector to store parents U children
  neighbors = union_(parents,children);
  // If we have a DAG with estimated Markov Blankets encoded,
  // We should not include spouses in the set of neighbors
  
  // Find nonadjacent common parents with i (spouses)
  NumericVector potential_spouses;
  size_t current_val;
  for (size_t child : children ){
    if (verbose){
      Rcout << "Call from DAG::getNeighbors. ";
      Rcout << "We are evaluating the following child: " << child << std::endl;
    }
    for (size_t j=0;j<p;++j){
      current_val = amat(j,child);
      // We don't check for adjacency b/c those nodes will be picked up
      // in the parent or the child sets already.
      if (current_val == 1 && i != j){
        potential_spouses.push_back(j);
        if (verbose){
          Rcout << "Call from DAG::getNeighbors. Node " << j;
          Rcout << " is a potential spouse of node " << i << ".\n";
        }
      }
    }
  }
  neighbors = union_(neighbors,potential_spouses);
  
  
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
NumericVector DAG::getNeighborsMultiTargets(const NumericVector &targets,
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
bool DAG::inNeighborhood(const size_t &i,
                         const size_t &j){
  validateIndex(i); validateIndex(j);
  // We consider a node in the same neighborhood as itself
  if (i==j){
    return true;
  }
  // This checks for adjacency
  if (Graph::areAdjacent(i,j)){
    return true;
  }
  // Check for spouses 
  
  // Find children of both nodes
  size_t p = Graph::size();
  // Determining if i and j are spouses
  // Don't need to check for adjacency since that has already been checked
  for (size_t k = 0;k<p;++k){
    if (amat(i,k)==1 && amat(j,k)==1){
      return true;
    }
  }
  return false;
}

bool DAG::isAncestor(const size_t &desc,const size_t &anc){
  validateIndex(desc); validateIndex(anc);
  if (desc==anc){
    return false;
  }
  StringVector node_names = getNodeNames();
  NumericVector current_ancestors;
  NumericVector next_level_ancestors;
  current_ancestors = getParents(desc);
  size_t level=1;
  while (current_ancestors.size()>0){
    if (isMember(current_ancestors,anc)){
      if (verbose){
        Rcout << node_names(anc) << " is an ancestor of " << node_names(desc);
        Rcout << " " << level << " levels up.\n";
      }
      return true;
    }
    std::for_each(current_ancestors.begin(),current_ancestors.end(),
                  [&next_level_ancestors,this](size_t node){
                    next_level_ancestors = union_(next_level_ancestors,getParents(node));
                  });
    current_ancestors = next_level_ancestors;
    next_level_ancestors = NumericVector::create();
    ++level;
  }
  
  return false;
}

