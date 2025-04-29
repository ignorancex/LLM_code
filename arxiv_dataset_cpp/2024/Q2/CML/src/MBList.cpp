#include "MBList.h"

// Any node i in the Markov Blanket of node x must be identified
// by mb_mat[i,x]=mb_mat[x,i]=1, though only one of these equivalences
// with 1 will be necessary for this function
NumericVector getMBFromMat(NumericMatrix mb_mat,size_t x){
  size_t p = mb_mat.nrow();
  NumericVector mb_vec = NumericVector::create();
  for (size_t i=0;i<p;++i){
    // (Sample) Any adjacency in provided matrix implies presence in MB
    if (mb_mat(x,i)!=0 || mb_mat(i,x)!=0){
      mb_vec.push_back(i);
    }
  }
  return mb_vec;
}

// Returns true if node i is in the MB of node target
// false otherwise
bool MBList::inMB(size_t target,size_t i){
  if (mb_list.count(i)==0){
    stop("%i is not an element of the map.\n",i);
  }
  return isMember(mb_list.find(target)->second,i);
}

// Sample version
// Node i is in MB(j) iff mb_mat[i,j]=mb_mat[j,i]=1
MBList::MBList(NumericVector node_vec,NumericMatrix mb_mat,bool verbose):
  verbose(verbose){
  if (verbose){
    Rcout << "Using estimated MB's to form list.\n";
  }
  // Sort all of the nodes in order
  std::sort(node_vec.begin(),node_vec.end());
  for (size_t node : node_vec){
    // Add pairs of <node,MB(node)> to the map data structure
    mb_list.insert(pair<size_t,NumericVector>(node,getMBFromMat(mb_mat,node)));
  }
  size = node_vec.size();
  if (verbose){
    printMBs();
  }
}

// Population version
// Node i is in MB(j) iff mb_mat[i,j]=1 or mb_mat[j,i]=1 or if i and j share a child
// Deletion of true_DAG should be taken care of outside this class
MBList::MBList(NumericVector node_vec,DAG* true_DAG,bool verbose):
  verbose(verbose){
  if (verbose){
    Rcout << "Using the true DAG for the MB List.\n";
  }
  // We must find the MB's for first-order neighbors as well
  NumericVector first_order_neighbors = NumericVector::create();
  NumericVector mb;
  // Add the MB for each node given
  for (size_t node : node_vec){
    mb = true_DAG->getNeighbors(node,false);
    mb_list.insert(pair<size_t,NumericVector>(node,mb));
    first_order_neighbors = union_(first_order_neighbors,mb);
  }
  // remove nodes already considered
  first_order_neighbors = setdiff(first_order_neighbors,node_vec);
  // Add the MB for each first-order neighbor
  for (size_t node : first_order_neighbors){
    mb_list.insert(pair<size_t,NumericVector>(node,true_DAG->getNeighbors(node,false)));
  }
  size = node_vec.size() + first_order_neighbors.size();
  if (verbose){
    printMBs();
  }
}

/*
 * Returns the union of the target neighborhoods (Markov Blankets)
 */
NumericVector MBList::getMBMultipleTargets(NumericVector targets,
                                           bool include_targets,
                                           bool exclude_targets){
  if (targets.size()==0){
    return(NumericVector::create());
  }
  if (include_targets && exclude_targets){
    warning("Inclusion/Exclusion of targets are conflicting. Exclusion takes precedence.");
  }
  // Unless we are to include targets, begin with empty set
  NumericVector neighbors;
  if (include_targets){
    if (verbose){
      Rcout << "We are including the target variables in the final set.\n";
    }
    neighbors = targets;
  }
  for (auto t : targets){
    if (verbose){
      Rcout << "Node: " << t << std::endl;
    }
    neighbors = union_(neighbors,getMB(t));
  }
  if (exclude_targets){
    if (verbose){
      Rcout << "Removing targets from final set\n";
    }
    neighbors = setdiff(neighbors,targets);
  }
  std::sort(neighbors.begin(),neighbors.end());
  if (verbose){
    Rcout << "All nodes from neighborhoods:\n";
    printVecElementsNoNames(neighbors,"","\n",", ");
  }
  return neighbors;
}

void MBList::printMBs(){
  Rcout << "MBList Size: " << size << endl;
  Rcout << "Markov Blankets:\n";
  
  for (auto const &it : mb_list){
    Rcout << it.first << ": ";
    printVecElementsNoNames(it.second);
    Rcout << endl;
  }
}
