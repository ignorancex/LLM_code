#include "SepSetList.h"

/*
 * Efficient numbering is the position of the node in the neighbors vector
 * used to initialize the separating set list
 * 
 * True numbering is the position of the node in the vector of the nodes from
 * the entire graph
 * 
 * Efficient numbering is used to index the SepSetList object
 * True numbering is used for the separating sets
 */

/*
 * Initializes a list containing separating sets for all pairs of nodes
 * Each separating set is set to NA initially
 */
SepSetList::SepSetList(NumericVector &neighbors):nodes(neighbors){
  N = neighbors.size();
  for (size_t i : neighbors){
    List subset = List::create();
    for (size_t j : neighbors){
      subset.push_back(NumericVector::create(NA_REAL));
    }
    S.push_back(subset);
  }
}

void checkInputValues(size_t i,size_t j, 
                      size_t N){
  if (i<0 || i>=N){
    stop("Input i=%i is invalid",i);
  }
  if (j<0 || j>=N){
    stop("Input j=%i is invalid",j);
  }
}

// Manually set Sep(i,j) to sep
// If no vector is provided, it is automatically set to -1 (empty set)
void SepSetList::changeList(size_t i,size_t j,
                            NumericVector sep){
  checkInputValues(i,j,N);
  NumericVector sep_new;
  // since the vector is passed in by reference automatically,
  // We must clone it to make any changes to `sep` in further steps
  // without affecting the current separating set
  sep_new = clone(sep); 
  List sublist;
  sublist = S[i];
  sublist[j] = sep_new;
}

/*
 * Takes the efficient numbering for two nodes (i and j)
 * and returns the nodes in the sep set (these nodes are given using true numbering)
 * We use true numbering because not all possible separating nodes are included
 * in the efficient numbering scheme (e.g. second-order neighbors)
 */
NumericVector SepSetList::getSepSet(size_t i,size_t j){
  // Identify separating sets for nodes i and j
  List sublist_i = S[i];
  return sublist_i[j];
}

// Checks if node k is in the separating set of nodes i and j
// k is assumed to be according to the true graph numbering
bool SepSetList::isSepSetMember(size_t i,size_t j,size_t k){
  checkInputValues(i,j,N);
  NumericVector sepset_ji;
  NumericVector sepset_ij;
  bool is_member;
  bool is_member_check;
  
  sepset_ji = getSepSet(j,i);
  sepset_ij = getSepSet(i,j);
  // Verify if k is in separating set for i and j
  is_member = isMember(sepset_ij,k);
  // Check other way just in case
  is_member_check = isMember(sepset_ji,k);
  
  if (is_member != is_member_check){
    warning("Separation sets for %i and %i disagree",i,j);
  }
  
  return is_member && is_member_check;
}

/*
 * Checks if node k (true numbering) is in the separating sets 
 * for nodes i and j (efficient numbering)
 */
bool SepSetList::isPotentialVStruct(size_t i,size_t j,size_t k) {
  return !isSepSetMember(i,j,k);
}

void SepSetList::printSepSetList(){
  List sublist;
  NumericVector sepLabels;
  for (size_t i=0;i<S.length();++i){
    sublist = S[i]; 
    for (size_t j=0;j<S.length();++j){
      Rcout << "S_{" << nodes(i) << "," << nodes(j) << "} = ";
      sepLabels = sublist[j];
      printVecElementsNoNames(sepLabels,"",""," ");
      Rcout << " ";
    }
    Rcout << std::endl;
  }
}

