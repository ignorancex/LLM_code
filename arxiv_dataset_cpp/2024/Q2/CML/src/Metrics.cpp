#include <algorithm>
#include "DAG.h"
#include "PDAG.h"
#include "SharedFunctions.h"
#include <string>
using namespace Rcpp;

// Ensure the adjacency matrices have the same dimensions and are square
void validateInputs(NumericMatrix est,NumericMatrix truth){
  if (est.ncol() != est.nrow() || truth.ncol() != truth.nrow()){
    stop("Inputted matrices are not both square matrices");
  }
  
  if (est.ncol() != truth.ncol()){
    stop("Inputted matrices do not have matching dimensions");
  }
}

// Ensure that the inputted target values are valid
void validateTargets(NumericMatrix g,NumericVector targets){
  if ( is_true(any(targets > g.ncol()-1)) ){
    stop("Invalid target index: value(s) greater than size of graph");
  } else if ( is_true(any(targets < 0)) ){
    stop("Invalid target index: negative value(s)");
  }
}

// Returns the number of edges in the graph
// Tested
// [[Rcpp::export]]
int getEdgeNumber(NumericMatrix G){
  int p = G.nrow();
  int total_edges=0;
  for (int i=0;i<p;++i){
    for (int j=i+1;j<p;++j){
      if (G(i,j)!=0 || G(j,i)!=0){
        ++total_edges;
      }
    }
  }
  return total_edges;
}

/*
 * In order to be considered for the within-neighborhood metrics,
 * both nodes must be in the same target neighborhood for at least
 * one target. If not, even if they are in the same neighborhood as each other,
 * we will not consider them in a shared neighborhood
 * Tested
 */
// [[Rcpp::export]]
bool sharedNeighborhood(NumericMatrix reference,NumericVector targets,
                        int i,int j,bool verbose = false){
  int p = reference.nrow();
  StringVector node_names;
  for (int i=0;i<p;++i){
    String node("V");
    node += i;
    node_names.push_back(node);
  }
  
  PDAG g_ref(p,node_names,reference);
  for (NumericVector::iterator it=targets.begin();it<targets.end();++it){
    // Check if i and j share a neighborhood with a particular target
    if (g_ref.inNeighborhood(*it,i) && g_ref.inNeighborhood(*it,j)){
      if (verbose){
        Rcout << i << " and " << j << " are in the neighborhood of target " << *it << std::endl;
      }
      return true;
    }
  }
  
  // Otherwise, return false
  if (verbose){
    Rcout << "Nodes " << i << " and " << j << " don't share the same target neighborhood\n";
  }
  return false;
}

/*
 * Checks if node i is in any of the target neighborhoods
 */
// [[Rcpp::export]]
bool inTargetNeighborhood(NumericMatrix reference,NumericVector targets,
                          int i,bool verbose = false){
  validateTargets(reference,targets);
  int p = reference.nrow();
  StringVector node_names;
  for (int j=0;j<p;++j){
    String node("V");
    node += j;
    node_names.push_back(node);
  }
  PDAG g_ref(p,node_names,reference);
  for (NumericVector::iterator it=targets.begin();it<targets.end();++it){
    // Check if i is in target neighborhood
    if (g_ref.inNeighborhood(*it,i)){
      if (verbose){
        Rcout << i << " is in the neighborhood of target node " << *it << std::endl;
      }
      return true;
    }
  }
  // Otherwise, return false
  return false;
}

// Checks whether nodes i and j in graph G are adjacent but not ancestral
// This is useful for estimated adjacency matrices where we have mixed notation
bool isAdjNotAncestral(NumericMatrix G,size_t i,size_t j,bool verbose=false){
  bool check1 = (G(i,j) == 0) && (G(j,i) == 0);
  bool check2 = (G(i,j)>1) || (G(j,i)>1);
  if (check1){
    if (verbose){
      Rcout << "Nodes " << i << " and " << j << " are ";
      Rcout << "not adjacent.\n";
    }
    return false; // i and j are not adjacent
  } else if (check2){
    if (verbose){
      Rcout << "Nodes " << i << " and " << j << " are ";
      Rcout << "connected by an ancestral edge.\n";
    }
    return false; // this is an ancestral edge
  } else {
    if (verbose){
      Rcout << "Nodes " << i << " and " << j << " are ";
      Rcout << "adjacent.\n";
    }
    return true; // at least one entry is 1 and the other is either 0 or 1
  }
}

/*
 * Skeleton comparison goes through every combination of nodes in the graph
 * and compares whether or not we have an accurate skeleton
 * Only non-ancestral edges from the estimated graph are included
 */
// [[Rcpp::export]]
List compareSkeletons(NumericMatrix est,NumericMatrix truth,bool verbose=false){
  validateInputs(est,truth);
  int fp=0;
  int fn=0;
  int correct=0;
  bool est_adjacent;
  int p = est.nrow();
  for (int i=0;i<p;++i){
    for (int j=i+1;j<p;++j){
      est_adjacent = isAdjNotAncestral(est,i,j,verbose);
      if (est_adjacent && (truth(i,j)==0 && truth(j,i)==0)){
        if (verbose){
          Rcout << "False positive edge for nodes " << i << " and " << j << std::endl;
        }
        ++fp;  // Est: i,j adjacent and not ancestral, but truth not adjacent
      } else if (!est_adjacent && (truth(i,j)!=0 || truth(j,i)!=0)){
        if (verbose){
          Rcout << "False negative edge for nodes " << i << " and " << j << std::endl;
        }
        ++fn; // Est: i,j not adjacent but adjacent in truth
      } else if (est_adjacent && (truth(i,j)!=0 || truth(j,i)!=0)){
        if (verbose){
          Rcout << "True positive edge for nodes " << i << " and " << j << std::endl;
        }
        ++correct; // i,j adjacent in both graphs and i and j not ancestral in estimated
      }
    }
  }
  return List::create(
    _["skel_fp"]=fp,
    _["skel_fn"]=fn,
    _["skel_correct"]=correct
  );
}

/*
 * These functions serve to find the differences in the v-structures in two mixed graphs
 * 
 * checkTriple determines if there is a v-structure with nodes i,j,k and returns them in order
 * Note: (i,j,k) refers to i->j<-k
 */
List checkTriple(NumericMatrix g,int i,int j,int k,bool verbose=false){
  // check if i *-> k <-* j, then k *-> j <-* i, and j *-> i <-* k with each set of parents nonadjacent
  //Rcout << "i: " << i << " | j: " << j << " | k: " << k << " | Conclusion: ";
  
  if (g(i,k)==1 && g(k,i)==0 && g(j,k)==1 && g(k,j)==0 && g(i,j)==0 && g(j,i)==0)
  {
    
    if (verbose){
      Rcout << i << " -> " << k << " <- " << j << std::endl;  
    }
    
    return List::create(_["result"]=true,_["order"]=NumericVector::create(i,k,j));  
  }
  else if (g(k,j)==1 && g(j,k)==0 && g(i,j)==1 && g(j,i)==0 && g(i,k)==0 && g(k,i)==0)
  {
    if (verbose){
      Rcout << i << " -> " << j << " <- " << k << std::endl;  
    }
    
    return List::create(_["result"]=true,_["order"]=NumericVector::create(k,j,i));
  }
  else if (g(j,i)==1 && g(i,j)==0 && g(k,i)==1 && g(i,k)==0 && g(j,k)==0 && g(k,j)==0)
  {
    if (verbose){
      Rcout << j << " -> " << i << " <- " << k << std::endl;  
    }
    
    return List::create(_["result"]=true,_["order"]=NumericVector::create(j,i,k));
  }
  else 
  {
    if (verbose){
      Rcout << "No unshielded triple\n";
      Rcout << std::endl;
    }
    
    return List::create(_["result"]=false,_["order"]=NA_REAL);
  }
}

/*
 * Checks if graph g2 has the same v-structure as it is found in another graph
 */
bool checkOtherTriple(NumericMatrix g2,NumericVector v,bool verbose=false){
  // check v(0) -> v(1) <- v(2) with v(0) and v(2) nonadjacent
  if (g2(v(0),v(1))==1 && g2(v(1),v(0))==0 && 
      g2(v(2),v(1))==1 && g2(v(1),v(2))==0 && 
      g2(v(0),v(2))==0 && g2(v(2),v(0))==0){
    if (verbose){
      Rcout << "v-structure correctly recovered\n"; 
    }
    return true;
  } else {
    if (verbose){
      Rcout << "v-structure *not* present in other graph\n";
    }
    return false;
  }
}
/*
 * We go through every triple of nodes in the vertex set and compare the v-structures
 * we obtain. V-Structure identification only counts for edges that are non-ancestral
 * in the estimated graph
 */
// [[Rcpp::export]]
List compareVStructures(NumericMatrix est,NumericMatrix truth,bool verbose=false){
  validateInputs(est,truth);
  int num_missing=0;
  int num_added=0;
  int num_correct=0;
  
  int p = est.nrow();
  List triple_check;
  bool continue_checking;
  
  for (int i=0;i<p-2;++i){
    for (int j=i+1;j<p-1;++j){
      for (int k=j+1;k<p;++k){
        continue_checking=true;
        if (verbose){
          Rcout << "Checking: " << i << ", " << j << ", and " << k << std::endl;
          Rcout << "Estimated Graph:\n";
        }
        // v-structure in estimated graph only counts if they are 
        // identified in same neighborhood and properly oriented
        // we do not consider ancestral v-structures for our metrics
        triple_check = checkTriple(est,i,j,k,verbose);
        if (triple_check["result"]){
          // Does contain a v-structure with nodes i,j,k
          if (verbose){
            Rcout << "True Graph:\n";
          }
          // Check if the same v-structure appears in ground truth
          if (checkOtherTriple(truth,triple_check["order"],verbose)){
            // We don't need to check for false negatives since v-structure was correctly identified
            continue_checking=false; 
            ++num_correct;
          } else {
            // We have a false positive
            ++num_added; 
          }
        }
        
        // Now, we check if we missed any v-structures
        if (continue_checking){
          if (verbose){
            Rcout << "True Graph:\n";
          }
          // Identify any v-structures in ground truth for nodes i,j,k
          triple_check = checkTriple(truth,i,j,k,verbose);
          if (triple_check["result"]){
            if (verbose){
              Rcout << "V-structure *not* present in estimated graph\n";
            }
            // Since we have already checked the estimated graph
            // Any additional v-structure in the ground truth not previously
            // accounted for is regarded as a false negative in the estimated graph
            ++num_missing;
          }
        }
      }
    }  
  }
  
  return List::create(
    _["missing"]=num_missing,
    _["added"]=num_added,
    _["correct"]=num_correct
  );
}

/*
 * Tracks the recovery of parents of target nodes in the estimated graph
 * using the ground truth to identify true parents
 */
void oneTargetPRA(NumericMatrix est,NumericMatrix truth,int t,NumericVector &targets,
                  int &correct,int &missing,int &added,int &potential,bool verbose){
  int p = est.nrow();
  for (int i=0;i<p;++i){
    if (i==t){
      continue;
    }
    if (verbose){
      Rcout << "t: " << t << " | i: " << i; 
    }
    
    /*
     * We first deal with the case where the estimated graph has i as a parent of t
     * We can have:
     * a) False positive: not adjacent in true graph or t -> i
     * b) Correct
     * c) False positive: undirected edge for t - i in the true graph
     * 
     * We then deal with the case where the estimated graph does not have i as a parent of t
     * We can have:
     * a) False negative: i -> t in true graph
     * b) Correct: Not noted since this is a true negative
     * c) Potential: undirected edge i - t in estimated graph and i -> t in true graph
     * d) Correct: undirected edge for both; also not noted
     */
    if (est(i,t)==1 && est(t,i)==0){ // Est: i -> t
      if (verbose){
        Rcout << " | Est. Graph: Parent | True graph: ";
      }
      // Truth: Either i <- t or i and t are not adjacent
      if (truth(i,t)==0){
        ++added;
        if (verbose){
          Rcout << "Not a parent | ";
          Rcout << "Added: " << added << std::endl; 
        }
      }
      else if (truth(i,t)==1 && truth(t,i)==0) { // Truth: i -> t
        ++correct;
        if (verbose){
          Rcout << "Parent | ";
          Rcout << "Correct: " << correct << std::endl;
        }
        
      } else if (truth(i,t)==1 && truth(t,i)==1){ // Truth: i - t
        ++added;
        if (verbose){
          Rcout << "Not a parent | ";
          Rcout << "Undirected edge in True Graph | Added: " << added << std::endl;  
        }
      }
    } else { 
      // Est: Either i - t or i and t are not adjacent 
      // or some ancestral marking (not regarded as in the same neighborhood)
      if (truth(i,t)==1 && truth(t,i)==0){ // Truth: i -> t
        if (est(i,t)==1 && est(t,i)==1){ // Est: i - t
          ++potential;
          if (verbose){
            Rcout << " | Undirected edge in Est. Graph | Potential: " << potential;
          }
        } else {
          ++missing;
          if (verbose){
            Rcout << " | Est. Graph: Not a parent | True graph: Parent | ";
            Rcout << "Missing: " << missing;
          }
        }
      }
      if (verbose) Rcout << std::endl;
    }
  }
}

// [[Rcpp::export]]
List parentRecoveryAccuracy(NumericMatrix est,NumericMatrix truth,
                            NumericVector targets,bool verbose=false){
  validateInputs(est,truth);
  validateTargets(est,targets);
  int num_correct=0;
  int num_added=0;
  int num_missing=0;
  int num_potential=0;
  // Add all of the parent recovery statistics for each of the targets
  std::for_each(targets.begin(),
                targets.end(),
                [&](int t) {oneTargetPRA(est,truth,t,targets,
                    num_correct,num_missing,num_added,num_potential,verbose);});
  
  return List::create(
    _["missing"]=num_missing,
    _["added"]=num_added,
    _["correct"]=num_correct,
    _["potential"]=num_potential
  );
}

// Returns true if i is an ancestor of j
bool idAncestors(NumericMatrix reference,int desc,int anc,bool verbose=true){
  int p = reference.nrow();
  StringVector node_names;
  makeNodeNames(p,node_names);
  DAG g_ref(p,node_names,reference,false);
  return g_ref.isAncestor(desc,anc);
}

// Returns true if there is an ancestral path between anc and desc
// that is not mediated by any node in the target neighborhood of either node
// bool checkAncestralPath(NumericMatrix reference,NumericVector targets,
//                         int anc,int desc,bool verbose=true){
//   if (verbose){
//     Rcout << "Checking if " << anc << " is an unmediated ancestor of " << desc << std::endl;
//   }
//   int p = reference.nrow();
//   StringVector node_names;
//   makeNodeNames(p,node_names);
//   DAG g_ref(p,node_names,reference,verbose);
//   
//   // If anc is not an ancestor of desc, return false
//   if (!g_ref.isAncestor(desc,anc)){
//     if (verbose){
//       Rcout << "Node " << desc << " is not a descendant of " << anc << std::endl;
//     }
//     return false;
//   }
//   
//   // Get parents of the potential descendant node
//   NumericVector parent_set = g_ref.getParents(desc);
//   NumericVector::iterator parent=parent_set.begin();
//   for (;parent<parent_set.end();++parent){
//     // Check if there is a parent node in the same target neighborhood
//     // as the descendant node which is also a descendant of the ancestral node
//     if (g_ref.isAncestor(*parent,anc) && 
//         sharedNeighborhood(reference,targets,*parent,desc,verbose)){
//       if (verbose){
//         Rcout << "Node " << *parent << " is a parent of " << desc;
//         Rcout << " and is in the same target neighborhood, while also being a ";
//         Rcout << "descendant of " << anc << std::endl;
//       }
//       return false;
//     }
//   }
//   
//   NumericVector children_set = g_ref.getChildren(anc);
//   NumericVector::iterator child=children_set.begin();
//   for (;child<children_set.end();++child){
//     // Check if there is a parent node in the same target neighborhood
//     // as the descendant node which is also a descendant of the ancestral node
//     if (g_ref.isAncestor(desc,*child) && 
//         sharedNeighborhood(reference,targets,*child,anc,verbose)){
//       if (verbose){
//         Rcout << "Node " << *child << " is a child of " << anc;
//         Rcout << " and is in the same target neighborhood, while also being an ";
//         Rcout << "ancestor of " << desc << std::endl;
//       }
//       return false;
//     }
//   }
//   // There are no mediating nodes in the path from anc to desc
//   // which share the same target neighborhood as desc
//   if (verbose){
//     Rcout << "There is an unmediated ancestral path between " << anc << " and " << desc << std::endl;
//   }
//   return true;
//   
// }

// // [[Rcpp::export]]
// List interNeighborhoodEdgeMetrics(NumericMatrix est,NumericMatrix reference,
//                                   NumericVector targets,bool verbose=true){
//   int p = est.nrow();
//   
//   int true_ancestor = 0;
//   int missing_ancestor = 0;
//   int missing_orientation = 0;
//   int reverse_orientation = 0;
//   int added_connection = 0;
//   int false_positive_arrow = 0;
//   
//   bool connected; // tracks whether two nodes are connected in estimated graph
//   bool is_i_ancestor_j; // tracks whether i is an ancestor of j
//   bool is_j_ancestor_i; // tracks whether j is an ancestor of i
//   bool is_anc_path_unmediated_ij; // tracks whether the ancestral path is mediated between i and j
//   bool is_anc_path_unmediated_ji; // tracks whether the ancestral path is mediated between j and i
//   bool i_arrow_j;
//   bool j_arrow_i;
//   bool undirected;
//   bool bidirected;
//   
//   for (int i=0;i<p-1;++i){
//     for (int j=i+1;j<p;++j){
//       // check the following:
//       // i and j each belong to a target neighborhood 
//       // both belong to different target neighborhoods
//       if (!sharedNeighborhood(reference,targets,i,j,verbose) && 
//           inTargetNeighborhood(reference,targets,i,verbose) && 
//           inTargetNeighborhood(reference,targets,j,verbose)){
//         if (verbose){
//           Rcout << "Looking at nodes " << i << " and " << j << std::endl;
//         }
//         
//         // Check if i is connected to j in estimated graph
//         connected = est(i,j) != 0 || est(j,i) != 0;
//         if (est(i,j)==3 && est(j,i)==3){
//           warning("Adjacency matrix entries for (%i,%i) and (%i,%i) are both 3.",i,j,j,i);
//         } 
//         // Check if i is an ancestor of j in the ground truth graph
//         is_i_ancestor_j = idAncestors(reference,j,i,verbose);
//         // Check if there is another node in the same neighborhood of i or j that mediates 
//         // The ancestral path between the two
//         is_anc_path_unmediated_ij = checkAncestralPath(reference,targets,i,j,verbose);
//         // Repeat same procedure as above, but switch i and j
//         is_j_ancestor_i = idAncestors(reference,i,j,verbose);
//         is_anc_path_unmediated_ji = checkAncestralPath(reference,targets,j,i,verbose);
//         if (!connected){
//           if (verbose){
//             Rcout << "Nodes are unconnected in estimated graph\n";
//           }
//           if ((is_j_ancestor_i && is_anc_path_unmediated_ji) || 
//               (is_i_ancestor_j && is_anc_path_unmediated_ij)){
//             if (verbose){
//               Rcout << "Missing ancestral relationship\n";
//             }
//             ++missing_ancestor;
//           }
//         } else { // i and j are connected in estimated graph
//           if (verbose){
//             Rcout << "Nodes are connected in estimated graph\n";
//           }
//           i_arrow_j = (est(i,j)==2 && est(j,i)!=2) || (est(i,j)==1 && est(j,i)==0);
//           j_arrow_i = (est(j,i)==2 && est(i,j)!=2) || (est(i,j)==0 && est(j,i)==1);
//           undirected = est(i,j)==1 && est(j,i)==1;
//           bidirected = est(i,j)==2 && est(j,i)==2;
//           if (is_i_ancestor_j && i_arrow_j && is_anc_path_unmediated_ij){
//             if (verbose){
//               Rcout << "True ancestral relationship (i->j)\n";
//             }
//             ++true_ancestor;
//           } else if (is_j_ancestor_i && j_arrow_i && is_anc_path_unmediated_ji){
//             if (verbose){
//               Rcout << "True ancestral relationship (j->i)\n";
//             }
//             ++true_ancestor;
//           } else if (is_i_ancestor_j && (undirected || bidirected) && is_anc_path_unmediated_ij){
//             if (verbose){
//               Rcout << "Missing orientation of ancestral relationship (i->j)\n";
//             }
//             ++missing_orientation;
//           } else if (is_j_ancestor_i && (undirected || bidirected) && is_anc_path_unmediated_ji){
//             if (verbose){
//               Rcout << "Missing orientation of ancestral relationship (j->i)\n";
//             }
//             ++missing_orientation;
//           } else if (is_i_ancestor_j && j_arrow_i && is_anc_path_unmediated_ij){
//             if (verbose){
//               Rcout << "Orientation Reversed: j->i instead of i->j\n";
//             }
//             ++reverse_orientation;
//           } else if (is_j_ancestor_i && i_arrow_j && is_anc_path_unmediated_ji){
//             if (verbose){
//               Rcout << "Orientation Reversed: i->j instead of j->i\n";
//             }
//             ++reverse_orientation;
//           } else if ((i_arrow_j || j_arrow_i) && !(is_i_ancestor_j || is_j_ancestor_i)){
//             if (verbose){
//               Rcout << "False positive ancestral relationship\n";
//             }
//             ++false_positive_arrow;
//           } else {
//             if (verbose){
//               Rcout << "False positive connection\n"; 
//             }
//             ++added_connection;
//           }
//         }
//       }
//     }
//   }
//   return List::create(
//     _["CorrectAncestors"]=true_ancestor,
//     _["MissingAncestors"]=missing_ancestor,
//     _["MissingOrientation"]=missing_orientation,
//     _["ReverseOrientation"]=reverse_orientation,
//     _["FPOrientedEdge"]=false_positive_arrow,
//     _["AddedConnection"]=added_connection
//   );
// }
 // [[Rcpp::export]]
List interNeighborhoodEdgeMetrics(NumericMatrix est,NumericMatrix reference,
                                  NumericMatrix true_dag,
                                  NumericVector nbhd,bool verbose=false){
  int p = est.nrow();
  size_t true_anc = 0;
  size_t incorrect_anc = 0;
  size_t total_anc_edges = 0;
  for (int i=0;i<p-1;++i){
    for (int j=i+1;j<p;++j){
      if (est(i,j)==3 && est(j,i)==3){
        warning("Adjacency matrix entries for (%i,%i) and (%i,%i) are both 3.",i,j,j,i);
      }
      // Only consider edges that are ancestral
      if (est(i,j)>1 && est(j,i)>1){
        if (verbose){
          Rcout << "Looking at nodes " << i << " and " << j << "...";
        }
        ++total_anc_edges;
        // i is estimated to be an ancestor of j
        if (est(i,j)==2 && est(j,i)==3){
          if (idAncestors(true_dag,nbhd(j),nbhd(i))){
            if (verbose){
              Rcout << "true ancestor" << std::endl;
            }
            ++true_anc;
          } else {
            if (verbose){
              Rcout << "incorrect ancestor" << std::endl;
            }
            ++incorrect_anc;
          }
        } else if (est(i,j)==3 && est(j,i)==2){
          // j is estimated to be an ancestor of i
          if (idAncestors(true_dag,nbhd(i),nbhd(j))){
            if (verbose){
              Rcout << "true ancestor" << std::endl;
            }
            ++true_anc;
          } else {
            if (verbose){
              Rcout << "incorrect ancestor" << std::endl;
            }
            ++incorrect_anc;
          }
        } else {
          if (verbose){
            Rcout << "nothing" << std::endl;
          }
        }
      }
    }
  }
  return List::create(
    _["CorrectAncestors"]=true_anc,
    _["IncorrectAncestors"]=incorrect_anc,
    _["TotalAncestralEdges"]=total_anc_edges
  );
}


/*
 * A true positive is when the orientation exactly matches in both graphs
 * A true negative is when there is no edge in both graphs
 * A false positive is when there is an edge in the est. graph but no edge in true
 * A false negative is whenever there is an edge in the true graph 
 * which does not exactly match the edge in the estimated graph
 */
// [[Rcpp::export]]
double overallF1(NumericMatrix est,NumericMatrix ref,
                 NumericVector targets,
                 bool verbose=false){
  validateInputs(est,ref);
  validateTargets(ref,targets);
  // Different edge comparison categories
  double tp = 0;
  double fp = 0;
  double fn = 0.;
  double incorrect_orientation = 0;
  
  int p = est.nrow();
  // Placeholders for adjacency matrix values
  // Estimated Graph
  int e_ij;
  int e_ji;
  // True (Reference) Graph
  int t_ij;
  int t_ji;
  // Loop through the upper triangular of the adj. matrix
  for (int i=0;i<p;++i){
    for (int j=i+1;j<p;++j){
      // Both i and j must be in the same target neighborhood to be considered
      if (sharedNeighborhood(ref,targets,i,j)){
        e_ij = est(i,j); e_ji = est(j,i); t_ij = ref(i,j); t_ji = ref(j,i);
        // Both entries match in each graph and there is an edge present -> TP
        if ((e_ij==t_ij) && (e_ji==t_ji) && (e_ij!=0 || e_ji!=0)){
          tp += 1;
          if (verbose){
            Rcout << "Edge between " << i << " and " << j << " match. TP=";
            Rcout << tp << std::endl;
          }
        } else if ((e_ij!=t_ij) || (e_ji!=t_ji)){
          // At least one of the entries differs from est. to true graph
          
          // There is an edge in the true graph
          if (t_ij!=0 || t_ji != 0){
            // There is also an edge in the estimated graph -> Incorrect orient.
            if (e_ij==1 || e_ji==1){
              incorrect_orientation += 1;
              if (verbose){
                Rcout << "Edge between " << i << " and " << j;
                Rcout << " appears in true graph and in the estimated graph,";
                Rcout << " but they have different orientations.";
                Rcout << " Incorrect Orientation=" << incorrect_orientation << std::endl;
              }
            } else {
              // There is no corresponding edge in the estimated graph -> FN
              fn += 1;
              if (verbose){
                Rcout << "Edge between " << i << " and " << j;
                Rcout << " appears in true graph but not in the estimated graph.";
                Rcout << " FN=" << fn << std::endl;
              }
            }
          } else {
            // An edge is present in the est. graph, but there is no edge
            // in the true graph -> FP
            fp += 1;
            if (verbose){
              Rcout << "Edge between " << i << " and " << j;
              Rcout << " appears in estimated graph but not in the true graph.";
              Rcout << " FP=" << fp << std::endl;
            }
          }
        }
      }
    }
  }
  double f1 = (2.0 * tp) / (2.0*tp + fp + fn + incorrect_orientation);
  
  return f1;
}


// [[Rcpp::export]]
DataFrame allMetrics(NumericMatrix est,NumericMatrix ref_graph,
                     NumericVector targets,NumericMatrix true_dag,
                     NumericVector nbhd,bool verbose=false,
                     std::string algo="pc",std::string which_nodes=""){
  validateInputs(est,ref_graph);
  validateTargets(ref_graph,targets);
  // Find the metrics for comparing the graphs
  List est_skeleton = compareSkeletons(est,ref_graph);
  List est_vstruct = compareVStructures(est,ref_graph);
  List est_pra = parentRecoveryAccuracy(est,ref_graph,targets);
  List est_ancestors = interNeighborhoodEdgeMetrics(est,ref_graph,true_dag,nbhd);
  return DataFrame::create(
    _[algo+"_"+which_nodes+"_skel_fp"]=est_skeleton["skel_fp"],
    _[algo+"_"+which_nodes+"_skel_fn"]=est_skeleton["skel_fn"],
    _[algo+"_"+which_nodes+"_skel_tp"]=est_skeleton["skel_correct"],
    _[algo+"_"+which_nodes+"_v_fn"] = est_vstruct["missing"],
    _[algo+"_"+which_nodes+"_v_fp"] = est_vstruct["added"],
    _[algo+"_"+which_nodes+"_v_tp"] = est_vstruct["correct"],
    _[algo+"_pra_fn"]=est_pra["missing"],
    _[algo+"_pra_fp"] = est_pra["added"],
    _[algo+"_pra_tp"] = est_pra["correct"],
    _[algo+"_pra_potential"] = est_pra["potential"],
    _[algo+"_ancestors_correct"]=est_ancestors["CorrectAncestors"],
    _[algo+"_ancestors_incorrect"]=est_ancestors["IncorrectAncestors"],
    _[algo+"_ancestors_total"]=est_ancestors["TotalAncestralEdges"],
    _[algo+"_overall_f1"]=overallF1(est,ref_graph,targets)                                             
  );
}

// [[Rcpp::export]]
DataFrame getNeighborhoodMetrics(NumericMatrix G){
  validateInputs(G,G);
  return DataFrame::create(
    _["size"]=G.ncol(),
    _["num_edges"]=getEdgeNumber(G)
  );
  
}
