#include "CML.h"

/*
 * NOTE: Code developed for the FCI rules adapted from the implementation
 * provided in the source code of the pcalg package:
 * https://rdrr.io/cran/pcalg/src/R/udag2pag.R
 */

// Sample Version
CML::CML(NumericMatrix true_dag,arma::mat df,
                   NumericVector targets,
                   NumericVector nodes_interest,
                   StringVector names,
                   int lmax,
                   double signif_level,
                   bool verbose,
                   std::string test,
                   bool estDAG) : 
  ConstrainedAlgo(true_dag,df,targets,nodes_interest,names,lmax,
                  signif_level,verbose,test,estDAG){
  
  // Make a map to relate efficient numbering to true numbering
  // Map: true numbering -> efficient numbering
  for (size_t i=0;i<N;++i){
    node_numbering.insert(std::pair<size_t,size_t>(neighborhood(i),i));
  }
  if (verbose){
    Rcout << "Element mapping for efficient ordering (True -> Efficient):\n";
    for(auto it = node_numbering.cbegin(); it != node_numbering.cend(); ++it)
    {
      Rcout << it->first << " " << it->second  << "\n";
    }
  }
}

CML::CML(NumericMatrix true_dag,
                   NumericVector targets,
                   NumericVector nodes_interest,
                   StringVector names,
                   int lmax,
                   bool verbose) : 
  ConstrainedAlgo(true_dag,targets,nodes_interest,names,
                  lmax,verbose){
  pop = true;
  // Make a map to relate efficient numbering to true numbering
  // Map: true numbering -> efficient numbering
  for (size_t i=0;i<N;++i){
    node_numbering.insert(std::pair<int,int>(neighborhood(i),i));
  }
  if (verbose){
    Rcout << "Element mapping for efficient ordering:\n";
    for(auto it = node_numbering.cbegin(); it != node_numbering.cend(); ++it)
    {
      Rcout << it->first << " " << it->second  << "\n";
    }
  }
}


void CML::getSkeletonTotal(){
  auto total_skeleton_start = high_resolution_clock::now();
  NumericVector neighbors;
  NumericVector edges_i;
  
  NumericMatrix kvals;
  
  int l = -1;
  bool cont = true;
  // Finding the skeleton for the complete undirected graph on U_t (X_t U N_t)
  while ((l < lmax) && cont){
    l += 1;
    if (verbose){
      Rcout << "The value of l is " << l << std::endl;
    }
    if (l>0){
      // only continue if there are enough nodes for potential sep. sets
      cont = false; 
    }
    for (size_t i=0;i<N;++i){
      if (verbose){
        Rcout << "The value of i is " << i;
        Rcout << " (" << names(neighborhood(i)) << ")"<< std::endl;
      }
      // Work through potential neighbors with separating set of size l
      // These potential neighbors are those currently connected to node i
      // in the current iteration's estimated graph
      edges_i = C_tilde->getAdjacent(i);
      for (auto j : edges_i){
        if (j > i){
          if (verbose){
            Rcout << "The value of j is " << j;
            Rcout << " (" << names(neighborhood(j)) << ")"<< std::endl;
          }
          // Find neighbors of i and j from the current graph C
          neighbors = setdiff(union_(edges_i,C_tilde->getAdjacent(j)),
                              NumericVector::create(i,j));
          std::sort(neighbors.begin(),neighbors.end());
          // If there are enough potential neighbors to match the current 
          // separating set size, we continue
          cont = false;
          if (neighbors.length()>= l){
            cont = true;
            if (verbose && l>0){
              if (neighbors.length() > 1){
                Rcout << "There are " << neighbors.length() << " neighbors.\n";
              } else {
                Rcout << "There is " << neighbors.length() << " neighbor.\n";
              }
            }
            kvals = combn_cpp(neighborhood[neighbors],l);
            // check whether nodes i and j are separated by any of the 
            // potential separating sets in kvals
            checkSeparation(l,i,j,kvals);
          }
        }
      }
    }
  }
  if (verbose){
    Rcout << "\n\nValues after Total Skeleton Run\n\n";
    print_elements();
  }
  auto total_skeleton_end = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(total_skeleton_end-total_skeleton_start);
  total_skeleton_time = duration.count() / 1e6; // time in seconds
}

void CML::getSkeletonTarget(const size_t &t){
  auto target_skeleton_start = high_resolution_clock::now();
  // We should start with l=1 because we've already done l=0 previously
  size_t l = 0; 
  bool cont = true;
  NumericVector neighbors;
  NumericVector edges_i;
  NumericMatrix kvals;
  
  // Making an adjustment to target based on efficient numbering 
  size_t target_efficient = node_numbering.find(t)->second;
  // Find neighborhood just surrounding the target node
  NumericVector target_neighborhood = mb_list->getMB(t);
  target_neighborhood.push_back(t);
  std::transform(target_neighborhood.begin(),target_neighborhood.end(),
                 target_neighborhood.begin(),
                 [this](size_t a) { return node_numbering.find(a)->second; });
  std::sort(target_neighborhood.begin(),target_neighborhood.end());
  
  if (verbose){
    Rcout << "\n\nFinding skeleton for the neighborhood of target " << t;
    Rcout << " (Efficient Number: " << target_efficient << ")"<< std::endl;
    Rcout << "Neighborhood nodes under consideration: ";
    printVecElementsNoNames(target_neighborhood,""," | (");
    for (size_t i=0;i<target_neighborhood.length();++i){
      if (i<target_neighborhood.length()-1){
        Rcout << names(neighborhood(target_neighborhood(i))) << "(";
        Rcout << neighborhood(target_neighborhood(i)) << ")" << " ";  
      } else {
        Rcout << names(neighborhood(target_neighborhood(i)));
        Rcout << "(" << neighborhood(target_neighborhood(i)) << ")" << ")\n";
      } 
    }
  }
  
  while ((l < lmax) && cont){
    l += 1;
    cont = false;
    if (verbose){
      Rcout << "The value of l is " << l << std::endl;
    }
    
    // We only consider the target and its neighborhood in the graph currently
    for (auto i : target_neighborhood){
      if (verbose){
        Rcout << "The value of i is " << i << std::endl;
      }
      // Work through potential neighbors with separating set of size l
      // These potential neighbors are those currently connected to node i 
      // in the current iteration's estimated graph
      // We do not want to separate the connections between two different cliques
      edges_i = intersect(C_tilde->getAdjacent(i),target_neighborhood);
      for (auto j : edges_i){
        if (j > i){
          if (verbose){
            Rcout << "The value of j is " << j << std::endl;
          }
          // Find neighbors of i and j from the true DAG (or they are estimated)
          // These neighbors are using the true node numbers 
          // (check documentation for this function)
          bool tmp = mb_list -> silencer();
          neighbors = mb_list->getMBMultipleTargets(
            NumericVector::create(neighborhood(i),neighborhood(j)),
            false, // not including i and j
            true); // We include the last argument to remove i and j
          mb_list -> removeSilencer(tmp);
          if (verbose){
            printVecElementsNoNames(neighbors,"Potential separating nodes: ","\n");  
          }
          // If there are enough potential neighbors to match the current 
          // separating set size, we continue
          if (neighbors.length()>=l){
            cont = true;
            if (verbose){
              Rcout << "There are " << neighbors.length() << " neighbor(s).\n";
            }
            kvals = combn_cpp(neighbors,l);
            checkSeparation(l,i,j,kvals);
          }
        }
      }
    }
  }
  if (verbose){
    Rcout << "The final C matrix:\n";
    C_tilde -> printAmat();
    Rcout << "Conclusion of algorithm.\n";
  }
  // Save the amount of time taken for the algorithm
  auto target_skeleton_end = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(target_skeleton_end-target_skeleton_start);
  double total_time = duration.count() / 1e6; // time in seconds
  
  target_skeleton_times.push_back(total_time);
}

// We are trying to identify structures i -> k <- j
// Where i and j are not adjacent, and k is not in the separating set of i and j
int CML::getVStructures() {
  size_t k_eff;
  int times_used=0;
  bool no_neighbors;
  bool j_invalid;
  NumericVector placeholder;
  
  NumericVector i_adj;
  NumericVector j_adj;
  NumericVector j_vals;
  NumericVector k_vals;
  
  List sublist_i;
  List sublist_j;
  
  String node_i;
  String node_j;
  
  NumericVector sepset_ij;
  NumericVector sepset_ji;
  
  if (verbose){
    Rcout << "Beginning loops to find v-structures.\n";
  }
  // We are searching for i-k-j where i and j are not adjacent and k is 
  // not in the separating set for i and j
  for (size_t i=0;i<N;++i){
    // We will search this vector for nodes connected to node i
    placeholder = C_tilde->getAmatRow(i); 
    no_neighbors = (all(placeholder==0)).is_true();
    // If there are neighbors to consider
    if (!no_neighbors){ 
      if (verbose){
        Rcout << "i: "<< i << " (" << names(neighborhood(i)) << ")" << std::endl;
      }
      i_adj = C_tilde->getAdjacent(i); // potential values of k
      j_vals = C_tilde->getNonAdjacent(i); // potential values of j
      // Iterate over possible j values
      for (auto j : j_vals){
        // We move on if:
        // Node j has no children,
        // j is parent to i, 
        // or we are repeating an analysis and this j should not be considered
        placeholder = C_tilde->getAmatRow(j);
        j_invalid = (all(placeholder==0)).is_true();
        j_invalid = j_invalid || C_tilde->getAmatVal(j,i)!= 0 || j <= i;
        if (!j_invalid){
          if (verbose){
            Rcout << "j: " << j << " (" << names(neighborhood(j)) << ")"<< std::endl;
          }
          j_adj = C_tilde->getAdjacent(j);
          k_vals = intersect(i_adj,j_adj); // k must be a neighbor of i and j
          std::sort(k_vals.begin(),k_vals.end());
          if (verbose && k_vals.length()>0){
            Rcout << "Potential k values: ";
            printVecElements(k_vals,names[neighborhood],"","\n");
          }
          // If there are no common neighbors, move to next j
          if (k_vals.length()!=0){
            // We loop through all of the common neighbors
            sepset_ij = S->getSepSet(i,j);
            sepset_ji = S->getSepSet(j,i);
            for (auto k : k_vals){
              if (verbose){
                Rcout << "k: " << k << " (" << names(neighborhood(k)) << ")\n"; 
              }
              // Verify if k is in separating set for i and j
              k_eff = k;
              k = neighborhood(k); // Switch k to true numbering
              if (S->isPotentialVStruct(i,j,k)){ 
                if (verbose){
                  Rcout << "Separation Set: ";
                  printVecElementsNoNames(sepset_ij);
                  Rcout << " | V-Structure (True Numbering): ";
                  Rcout << neighborhood(i) << "*->" << k << "<-*";
                  Rcout << neighborhood(j) << std::endl; 
                }
                C_tilde->setAmatVal(i,k_eff,2); // An arrow is denoted by "2"
                C_tilde->setAmatVal(j,k_eff,2); // i and j are separated ("0")
                ++times_used;
                if (verbose){
                  Rcout << "Rule 0 has been used " << times_used << " times.\n";
                }
              }
            }
          }
        }
      }
    }
  }
  return times_used;
}

/*
 * 
 * Define functions for the rules used in the FCI algorithm
 * Definitions:
 * 0: No edge
 * 1: -o
 * 2: -> (arrowhead)
 * 3: - (tail)
 * 
 * Note that a * represents a wild card that can represent either an open circle or an arrow
 * When we are considering the PC algorithm, for nodes i and j, if we have i -> j,
 * we regard G(i,j) = 2 and G(j,i) = 3. Therefore, in keeping with this convention, for FCI we will consider the arrowhead
 * information stored in G to be related to the arrowhead incident on the second node for the edge in consideration.
 * For example, if i o-> j is the edge in consideration, G(i,j)=2, since the arrow is incident on j, the second node
 * Similarly, G(j,i)=1, since the circle is incident on node i, which is the second node for the edge between node
 * j and node i.
 */

/*
 * For R1, we search for alpha *-> beta o-* gamma. 
 * If alpha and gamma are not adjacent, then orient the triple: alpha *-> beta -> gamma
 */

void CML::rule1search(size_t beta,size_t alpha,bool &track_changes){
  // Search for beta o-* gamma (beta (1) (!=0) gamma)
  //verbose = true;
  for (size_t gamma=0;gamma<N;++gamma){
    if ((C_tilde->operator()(gamma,beta)==1) && (C_tilde->operator()(beta,gamma)!= 0)){ 
      if ((C_tilde->getAmatVal(alpha,gamma)==0) && 
          (C_tilde->getAmatVal(gamma,alpha)==0)){
        if (C_tilde->getAmatVal(beta,gamma)==3){
          warning("Contradiction in Rule 1! G(%i,%i)=3 when it must be changed to 2",
                  beta,gamma);
        }
        C_tilde->operator()(beta,gamma)=2;
        C_tilde->operator()(gamma,beta)=3;
        if (verbose){
          Rcout << "Rule 1:\n";
          Rcout << "Orient: " << alpha << " *-> " << beta << " o-* " << gamma;
          Rcout << " as " << beta << " -> " << gamma << "\n";
        }
        track_changes=true;
        ++rules_used(1);
        if (verbose){
          Rcout << "Rule 1 has been used " << rules_used(1) << " times.\n";
        }
      }
    }
  }
}

bool CML::rule1(bool &track_changes) {
  // Outer loops: Go through every node to find an asterisk. 
  // The node it is incident on is alpha
  // The connected node is beta, which must have an arrowhead pointing into it.
  for (size_t alpha = 0;alpha<N;++alpha){
    for (size_t beta = 0;beta<N;++beta){
      if (C_tilde->getAmatVal(alpha,beta)==2 && C_tilde->getAmatVal(beta,alpha)!=0){ // alpha *-> beta
        rule1search(beta,alpha,track_changes);
      }
    }
  }
  return track_changes;
}

/*
 * Rule 2: Search for alpha -> beta *-> gamma (alpha (3) (2) beta (*) (2) gamma)
 * OR Search for alpha *-> beta -> gamma (alpha (*) (2) beta (3) (2) gamma)
 * For either case, if alpha *-o gamma, then orient alpha *-o gamma as alpha *-> gamma
 */
void CML::rule2search(size_t beta,size_t alpha,
                           bool condition1,bool condition2,
                           bool &track_changes){
  // Condition 1 refers to alpha -> beta *-> gamma
  if (condition1){
    for (size_t gamma=0;gamma<N;++gamma){
      if (C_tilde->getAmatVal(gamma,beta)!=0 && 
          C_tilde->getAmatVal(beta,gamma)==2){ // beta *-> gamma
        if (C_tilde->getAmatVal(alpha,gamma)==1 && 
            C_tilde->getAmatVal(gamma,alpha)!=0){ // alpha *-o gamma
          C_tilde->setAmatVal(alpha,gamma,2); // alpha *-> gamma
          track_changes = true;
          ++rules_used(2);
          if (verbose){
            Rcout << "Rule 2:\n";
            Rcout << "Orient: " << alpha << " -> " << beta << " *-> " << gamma;
            Rcout << " and " << alpha << "*-o" << gamma;
            Rcout << " as: " << alpha << " *-> " << gamma << std::endl;
            Rcout << "Rule 2 has been used " << rules_used(2) << " times.\n";
          }
        }
      }
    }
  } else if (condition2){ // Condition 2 refers to alpha *-> beta -> gamma
    for (size_t gamma=0;gamma<N;++gamma){
      if ((C_tilde->getAmatVal(gamma,beta)==3) && 
          (C_tilde->getAmatVal(beta,gamma)==2)){ // beta -> gamma
        if ((C_tilde->getAmatVal(alpha,gamma)==1) && 
            (C_tilde->getAmatVal(gamma,alpha)!=0)){ // alpha *-o gamma
          C_tilde->setAmatVal(alpha,gamma,2); // alpha *-> gamma
          if (verbose){
            Rcout << "Rule 2:\n";
            Rcout << "Orient: " << alpha << " *-> " << beta << " -> " << gamma;
            Rcout << " and " << alpha << "*-o" << gamma;
            Rcout << " as: " << alpha << " *-> " << gamma << std::endl;
          }
          track_changes = true;
          ++rules_used(2);
          if (verbose){
            Rcout << "Rule 2 has been used " << rules_used(2) << " times.\n";
          }
        }
      }
    }
  }
}

bool CML::rule2(bool &track_changes) {
  bool condition1;
  bool condition2;
  // Searching for alpha -> beta OR alpha *-> beta
  for (size_t alpha = 0;alpha<N;++alpha){
    for (size_t beta = 0;beta<N;++beta){
      condition1 = (C_tilde->getAmatVal(alpha,beta)==2) && 
        (C_tilde->getAmatVal(beta,alpha)==3); // alpha -> beta
      condition2 = (C_tilde->getAmatVal(alpha,beta)==2) && 
        (C_tilde->getAmatVal(beta,alpha)!=0); // alpha *-> beta
      if (condition1 | condition2){
        rule2search(beta,alpha,condition1,condition2,track_changes);
      }
    }
  }
  return track_changes;
}

/*
 * Rule 3: Search for alpha *-> beta <-* gamma (alpha (*) (2) beta (2) (*) gamma)
 * AND alpha *-o theta o-* gamma (alpha (*) (1) theta (1) (*) gamma)
 * If alpha and gamma are not adjacent and theta *-o beta, 
 * then orient theta *-o beta as theta *-> beta
 */
List CML::rule3asearch(size_t beta,size_t alpha){
  // There may be multiple values of gamma for which this holds. 
  // This assures we get them all.
  NumericVector gammafinal;
  bool success = false;
  for (size_t gamma=0;gamma<N;++gamma){
    // Search for beta <-* gamma
    if (C_tilde->getAmatVal(gamma,beta)==2 && 
        C_tilde->getAmatVal(beta,gamma)!=0 && 
        gamma!= alpha){
      gammafinal.push_back(gamma);
      if (!success){
        success = true;
      }
    }
  }
  
  return List::create(
    _["gamma"] = gammafinal,
    _["rule3success"] = success
  );
}

void CML::rule3bsearch(const size_t &alpha,const size_t &beta,
                            const size_t &gamma,bool &track_changes){
  bool condition1;
  bool condition2;
  //verbose = true;
  // We are searching for alpha (*) (1) theta (1) (*) gamma
  for (size_t theta = 0;theta<N;++theta){
    condition1 = (C_tilde->getAmatVal(alpha,theta)==1) && 
      (C_tilde->getAmatVal(theta,alpha)!=0); // alpha *-o theta
    condition2 = (C_tilde->getAmatVal(theta,gamma)!=0) && 
      (C_tilde->getAmatVal(gamma,theta)==1); // theta o-* gamma
    if (condition1 && condition2){
      if ((C_tilde->getAmatVal(alpha,gamma)==0) && 
          (C_tilde->getAmatVal(gamma,alpha)==0)){ // alpha and gamma are not adjacent
        if ((C_tilde->getAmatVal(theta,beta)==1) && 
            (C_tilde->getAmatVal(beta,theta)!=0)){ // theta *-o beta
          C_tilde->setAmatVal(theta,beta,2); // theta *-> beta
          if (verbose){
            Rcout << "Rule 3:\n";
            Rcout << "Orient: " << theta << " *-> " << beta << std::endl;
          }
          track_changes = true;
          ++rules_used(3);
          if (verbose){
            Rcout << "Rule 3 has been used " << rules_used(3) << " times.\n";
          }
        }
      }
    }
  }
}

bool CML::rule3(bool &track_changes) {
  List searchResults;
  NumericVector gammaVals;
  // (alpha (*) (2) beta (2) (*) gamma)
  for (size_t alpha = 0;alpha<N;++alpha){
    for (size_t beta = 0;beta<N;++beta){
      if ((C_tilde->getAmatVal(alpha,beta)==2) && 
          (C_tilde->getAmatVal(beta,alpha)!=0)){ // alpha *-> beta <-* gamma
        searchResults = rule3asearch(beta,alpha); // Search for gamma
        if (searchResults["rule3success"]){
          // Iterate over all values of gamma to find values of theta
          gammaVals = searchResults["gamma"];
          for (auto gamma : gammaVals){
            rule3bsearch(alpha,beta,gamma,track_changes);
          }
        }
      }
    }
  }
  return track_changes;
}


/*
 * Rule 4: If we have a discriminating path between theta and gamma for beta,
 * <theta,...,alpha,beta,gamma>
 * and beta o-* gamma; then if beta is in the separating set for theta and gamma
 * then we orient beta -> gamma; otherwise orient (alpha,beta,gamma) as 
 * alpha <-> beta <-> gamma
 */
bool CML::check_sep_r4(size_t beta,NumericVector md_path){
  if (verbose) Rcout << "Checking separation";
  size_t n = md_path.length();
  size_t theta = md_path(0);
  size_t gamma = md_path(n-1);
  
  if (verbose){
    Rcout << " of " << neighborhood(theta) << " and ";
    Rcout << neighborhood(gamma) << " by " << neighborhood(beta);
  }
  bool cond1 = S->isSepSetMember(theta,gamma,neighborhood(beta)); 
  if (verbose) {
    Rcout << "...finished\n"; 
  }
  return (cond1);
}

bool CML::rule4(bool &track_changes){
  bool cond1;
  bool cond2;
  NumericVector md_path;
  // Looking for beta o-* gamma
  for (size_t beta=0;beta<N;++beta){
    for (size_t gamma=0;gamma<N;++gamma){
      if (C_tilde->getAmatVal(beta,gamma)!=0 && 
          C_tilde->getAmatVal(gamma,beta)==1){
        // Begin looking for disc. path <theta,...,alpha,beta,gamma>
        // First, look for possible values of alpha
        for (size_t alpha=0;alpha<N;++alpha){
          // Need alpha <-* beta
          cond1 = C_tilde->getAmatVal(beta,alpha)==2 && 
            C_tilde->getAmatVal(alpha,beta)!=0;
          // triangle structure exists but is not oriented
          // alpha -> gamma
          cond2 = C_tilde->getAmatVal(gamma,alpha)==3 && 
            C_tilde->getAmatVal(alpha,gamma)==2; 
          if (cond1 && cond2){
            // Only continue to check if the edge between gamma and beta hasn't
            // already been modified
            if(C_tilde->getAmatVal(gamma,beta)==1){
              // Check for a discriminating path <theta,...,alpha,beta,gamma>
              md_path = C_tilde->minDiscPath(alpha,beta,gamma);
              if (md_path(0)==-1){
                Rcout << "No discriminating path for these nodes.\n";
              } else {
                // Check if beta separates theta and gamma
                if(check_sep_r4(beta,md_path)){
                  if (verbose){
                    Rcout << "\nRule 4\nThere is a discriminating path between ";
                    Rcout << md_path(0) << " and " << gamma << " for " << beta;
                    Rcout << " and " << beta << " is in the SepSet of " << gamma;
                    Rcout << " and " << md_path(0) << ". Orient: ";
                    Rcout << beta << " -> " << gamma << std::endl;
                  }
                  C_tilde->setAmatVal(beta,gamma,2);
                  C_tilde->setAmatVal(gamma,beta,3);
                } else {
                  // beta is not in the separating set
                  if (verbose){
                    Rcout << "\nRule 4\nThere is a discriminating path between ";
                    Rcout << md_path(0) << " and " << gamma << " for " << beta;
                    Rcout << " and " << beta << " is NOT in the SepSet of ";
                    Rcout << gamma << " and " << md_path(0) << ". Orient: ";
                    Rcout << alpha << " <-> " << beta << " <-> " << gamma << std::endl;
                  }
                  C_tilde->setAmatVal(beta,gamma,2);
                  C_tilde->setAmatVal(gamma,beta,2);
                  if (C_tilde->getAmatVal(alpha,beta)==3){
                    // This shouldn't be a problem if it comes up.
                    warning("Contradiction in Rule 4b"); 
                  }
                  C_tilde->setAmatVal(alpha,beta,2);
                }
                track_changes = true;
                ++rules_used(4);
                if (verbose){
                  Rcout << "Rule 4 has been used " << rules_used(4) << " times.\n";
                }
              }
              
            } 
          }  
        }
      }
    }  
  }
  return track_changes;
}


bool CML::rule8(bool &track_changes){
  bool cond1;
  bool cond2;
  bool cond3;
  for (size_t alpha=0;alpha<N;++alpha){
    for (size_t gamma=0;gamma<N;++gamma){
      if (C_tilde->getAmatVal(alpha,gamma)==2 && 
          C_tilde->getAmatVal(gamma,alpha)==1){ // alpha o-> gamma
        for (size_t beta=0;beta<N;++beta){
          cond1 = C_tilde->getAmatVal(beta,alpha)==3 && 
            C_tilde->getAmatVal(alpha,beta)==2; // alpha -> beta
          cond2 = C_tilde->getAmatVal(beta,alpha)==3 && 
            C_tilde->getAmatVal(alpha,beta)==1; // alpha -o beta
          if (cond1 || cond2){
            cond3 = C_tilde->getAmatVal(beta,gamma)==2 && 
              C_tilde->getAmatVal(gamma,beta)==3; // beta -> gamma
            if (cond3){
              C_tilde->setAmatVal(gamma,alpha,3); // alpha -> gamma
              if (cond1){ // alpha -> beta
                if (verbose){
                  Rcout << "\nRule 8\nOrient: " << alpha << " -> ";
                  Rcout << beta << " -> " << gamma;
                  Rcout << " with " << alpha << " o-> " << gamma << " as ";
                  Rcout << alpha << " -> " << gamma << std::endl;
                }
              } else { // alpha -o beta
                if (verbose){
                  Rcout << "\nRule 8\nOrient: " << alpha;
                  Rcout << " -o " << beta << " -> " << gamma;
                  Rcout << " with " << alpha << " o-> " << gamma << " as ";
                  Rcout << alpha << " -> " << gamma << std::endl;  
                }
              }
              track_changes = true;
              ++rules_used(8);
              if (verbose){
                Rcout << "Rule 8 has been used " << rules_used(8) << " times.\n";
              }
            }
          }
        }
      }  
    }  
  }
  return track_changes;
}


bool CML::rule9(bool &track_changes){
  bool cond1;
  bool cond2;
  bool cond3;
  bool cond4;
  bool cond_final;
  std::vector<int> beta_vals;
  size_t beta_current;
  NumericVector upd;
  
  for (size_t alpha=0;alpha<N;++alpha){
    for (size_t gamma=0;gamma<N;++gamma){
      if (C_tilde->getAmatVal(alpha,gamma)==2 && 
          C_tilde->getAmatVal(gamma,alpha)==1){ // alpha o-> gamma
        if (verbose){
          Rcout << "Potential alpha: " << alpha;
          Rcout << " | Potential gamma: " << gamma << std::endl;
        }
        beta_vals.clear();
        // Find all beta such that alpha (o-)-(o>) beta, 
        // and beta and gamma are not connected
        for (size_t beta=0;beta<N;++beta){
          cond1 = C_tilde->getAmatVal(alpha,beta) == 2 || 
            C_tilde->getAmatVal(alpha,beta) == 1;
          cond2 = C_tilde->getAmatVal(beta,alpha) == 1 || 
            C_tilde->getAmatVal(beta,alpha) == 3;
          cond3 = C_tilde->getAmatVal(gamma,beta) == 0 && 
            C_tilde->getAmatVal(beta,gamma) == 0;
          cond4 = beta != gamma;
          cond_final = cond1 && cond2 && cond3 && cond4;
          if (cond_final){
            if (verbose) {
              Rcout << "Potential beta: " << beta << std::endl; 
            }
            beta_vals.push_back(beta);
          }
        }
        while (beta_vals.size() > 0 && C_tilde->getAmatVal(gamma,alpha)==1){
          beta_current = beta_vals[beta_vals.size()-1];
          beta_vals.pop_back();
          upd = C_tilde->minUncovPdPath(alpha,beta_current,gamma);
          if (upd.length()>1){
            C_tilde->setAmatVal(gamma,alpha,3);
            if (verbose){
              Rcout << "Rule 9: There exists an uncovered potentially directed ";
              Rcout << "path between " << alpha << " and " << gamma << std::endl;
              Rcout << "Orient: " << alpha << " -> " << gamma << std::endl;
            }
            track_changes = true;
            ++rules_used(9);
            if (verbose){
              Rcout << "Rule 9 has been used " << rules_used(9) << " times.\n";
            }
          }
        }
      }  
    }
  }
  return track_changes;
}

// Check if rule 10 holds in the simple case
bool CML::rule10simple(const size_t &alpha,
                            const size_t &beta,
                            const size_t &gamma,
                            const size_t &d){
  bool cond1 = C_tilde->getAmatVal(alpha,beta)==1 ||
    C_tilde->getAmatVal(alpha,beta)==2;
  bool cond2 = C_tilde->getAmatVal(beta,alpha)==1 || 
    C_tilde->getAmatVal(beta,alpha)==3;
  bool cond3 = C_tilde->getAmatVal(alpha,d)==1 || 
    C_tilde->getAmatVal(alpha,d)==2;
  bool cond4 = C_tilde->getAmatVal(d,alpha)==1 || 
    C_tilde->getAmatVal(d,alpha)==3;
  bool cond5 = C_tilde->getAmatVal(d,beta)==0 && 
    C_tilde->getAmatVal(beta,d)==0;
  if (cond1&cond2&cond3&cond4&cond5){
    C_tilde->setAmatVal(gamma,alpha,3);
    if (verbose){
      Rcout << "\nRule 10 [easy]:\nOrient: ";
      Rcout << alpha << " -> " << gamma << std::endl;
    }
    ++rules_used(10);
    if (verbose){
      Rcout << "Rule 10 has been used " << rules_used(10) << " times.\n";
    }
    return true;
  } else {
    return false;
  }
}

NumericVector betaSearch(Graph *C_tilde,const size_t &gamma){
  NumericVector beta_vals = NumericVector::create();
  size_t N = C_tilde -> size();
  for (size_t b=0;b<N;++b){ // search for beta -> gamma
    bool cond1 = C_tilde->getAmatVal(gamma,b)==3 && 
      C_tilde->getAmatVal(b,gamma)==2;
    if (cond1){
      beta_vals.push_back(b);  
    }
  }
  return beta_vals;
}

// Identify nodes adjacent to alpha not equal to gamma such that 
// <alpha,x> is a p.d. path
NumericVector nextPathNodeSearch(Graph *C_tilde,
                                 const size_t &alpha,const size_t &gamma){
  // Find all x s.t. alpha (-o)-(o>) x  
  NumericVector x_vals = NumericVector(0); // Creates an empty vector
  size_t N = C_tilde -> size();
  for (size_t x=0;x<N;++x){ // first node on the potential path
    bool cond1 = C_tilde->getAmatVal(alpha,x)==1 || 
      C_tilde->getAmatVal(alpha,x)==2;
    bool cond2 = C_tilde->getAmatVal(x,alpha)==1 || 
      C_tilde->getAmatVal(x,alpha)==3;
    bool cond3 = x != gamma;
    if (cond1 && cond2 && cond3){
      x_vals.push_back(x);
    }
  }
  return x_vals;
}

/* 
 * Suppose alpha o-> gamma, beta -> gamma <- theta, p_1 is uncovered p.d. from alpha
 * to beta, and p_2 is uncovered p.d. from alpha to theta. Let mu be 
 * adjacent to alpha on p_1, and omega adjacent to alpha on p_2. If mu and omega are 
 * distinct, and not adjacent, then orient alpha -> gamma.
 */
bool CML::rule10(bool &track_changes){
  
  size_t counter_x1;
  size_t counter_x2;
  size_t x1;
  size_t x2;
  // search for alpha o-> gamma
  for (size_t alpha=0;alpha<N;++alpha){
    for (size_t gamma=0;gamma<N;++gamma){ 
      if (C_tilde->getAmatVal(alpha,gamma)==2 && 
          C_tilde->getAmatVal(gamma,alpha)==1){
        // Search for possible values of beta and theta: beta->gamma<-theta
        NumericVector beta_vals = betaSearch(C_tilde,gamma);
        if (beta_vals.length()>=2){ // Need at least two for beta and theta
          size_t counter_b=0;
          while (counter_b<beta_vals.length() && 
                 C_tilde->getAmatVal(gamma,alpha)==1){
            size_t beta = beta_vals(counter_b);
            if (verbose){
              Rcout << "Potential alpha: " << alpha << " | ";
              Rcout << "Potential gamma: " << gamma << " | ";
              Rcout << "Potential beta: " << beta << " | " << std::endl;
            }
            ++counter_b;
            size_t counter_theta=0;
            // possible values of theta
            NumericVector theta_vals = setdiff(beta_vals,
                                               NumericVector::create(beta)); 
            while ((counter_theta < theta_vals.length()) && 
                   (C_tilde->getAmatVal(gamma,alpha)==1)){
              size_t theta = theta_vals(counter_theta);
              if (verbose){
                Rcout << "Potential theta: " << theta << " | ";
              }
              ++counter_theta;
              // Check conditions for very simple case
              // mu = beta and omega=theta
              if (rule10simple(alpha,beta,gamma,theta)){
                track_changes = true;
              } else { // search for two minimal uncovered p.d. paths
                // Identify nodes such that <alpha,x> is p.d.
                NumericVector x_vals = nextPathNodeSearch(C_tilde,alpha,gamma);
                // Need starting values for p_1 *and* p_2
                if (x_vals.length()>=2){ 
                  counter_x1 = 0;
                  while ((counter_x1 < x_vals.length()) && 
                         C_tilde->getAmatVal(gamma,alpha)==1){
                    x1 = x_vals(counter_x1);
                    if (verbose){
                      Rcout << "Potential mu: " << x1 << " | ";
                    }
                    ++counter_x1;
                    NumericVector x2_vals = setdiff(x_vals,NumericVector::create(x1));
                    counter_x2 = 0;
                    while (counter_x2 < x2_vals.length() && 
                           C_tilde->getAmatVal(gamma,alpha)==1){
                      x2 = x2_vals(counter_x2);
                      if (verbose){
                        Rcout << "Potential omega: " << x2 << std::endl;
                      }
                      ++counter_x2;
                      // See if we can find uncovered p.d. path 
                      // t1 = <alpha,first_pos,...,beta>
                      NumericVector t1 = C_tilde->minUncovPdPath(alpha,x1,beta);
                      if (verbose) {
                        printVecElementsNoNames(t1,"t1: ","\n");
                      }
                      // Only search for t2 if t1 is found
                      if (t1.length()>1){
                        // See if we can find uncovered p.d. path 
                        // t1 = <alpha,second_pos,...,theta>
                        NumericVector t2 = C_tilde->minUncovPdPath(alpha,x2,theta);
                        if (verbose) {
                          printVecElementsNoNames(t2,"t2: ","\n");
                        }
                        if (t2.length()>1 && x1!=x2 && 
                            C_tilde->getAmatVal(x1,x2)==0){
                          C_tilde->setAmatVal(gamma,alpha,3);
                          if (verbose) {
                            Rcout << "\nRule 10\nOrient: " << alpha;
                            Rcout << " -> " << gamma << std::endl;
                          }
                          track_changes = true;
                          ++rules_used(10);
                          if (verbose){
                            Rcout << "Rule 10 has been used ";
                            Rcout << rules_used(10) << " times.\n";
                          }
                        } 
                      }
                    }
                  }
                }
                if (verbose) Rcout << std::endl;
              }
            }
          }
        }
      }
    }  
  }
  return track_changes;
}

void CML::allRules(){
  bool track_changes = true;
  while (track_changes){
    track_changes = false;
    track_changes = rule1(track_changes);
    track_changes = rule2(track_changes);
    track_changes = rule3(track_changes);
    track_changes = rule4(track_changes);
    track_changes = rule8(track_changes);
    track_changes = rule9(track_changes);
    track_changes = rule10(track_changes);
  }
}

/*
 * Because of the special circumstances of our algorithm,
 * we are able to reassign certain edges because of our knowledge of targets
 * and their neighborhoods.
 * <-> => - ([2,2] to [1,1]) DONE | 4/22/24: We are now keeping this as is. TODO: TEST AND UPDATE
 * o-> => -> (decrement both i and j by 1) DONE
 * -> => -> (no change, but adj. matrix has to account differently, [3,2] to [0,1]) DONE 
 * o-o => - (no change in adj. matrix) DONE
 * -o => -> ([3,1] to [0,1]) DONE
 * 
 * Also, to avoid confusion, any ancestral edges with circles will be relabeled
 * with a "4" instead of a 1
 */
void CML::convertMixedGraph(){
  size_t G_ij; // (i,j) element of adj. mat
  size_t G_ji; // (j,i) element of adj. mat
  bool sep_nbhd; // checks if the considered nodes are in separate neighborhoods
  for (size_t i=0;i<N;++i){
    for (size_t j=0;j<N;++j){
      sep_nbhd = false;
      // First check to see if i and j are in the same neighborhood
      // If i and j are not neighbors, 
      // then we should not change the orientations from the ancestral graph
      sep_nbhd = !(mb_list -> inMB(neighborhood(i),neighborhood(j)));
      G_ij = C_tilde -> getAmatVal(i,j);
      G_ji = C_tilde -> getAmatVal(j,i);
      if (!sep_nbhd){ // only convert edge notation w/in neighborhoods
        // if (G_ij==2 && G_ji==2){
        //   // Convert bidirected edge to undirected
        //   C_tilde->setAmatVal(i,j,1);
        //   C_tilde->setAmatVal(j,i,1);
        // } 
        if (G_ij==2 && G_ji==1){ 
          // Convert o-> to -> if i and j are in same nbhd
          C_tilde->setAmatVal(i,j,1);
          C_tilde->setAmatVal(j,i,0);
        } else if (G_ij==2 && G_ji==3){
          // Convert -> [3,2] to -> [0,1]
          C_tilde->setAmatVal(i,j,1);
          C_tilde->setAmatVal(j,i,0);
        } else if (G_ij==1 && G_ji==3){
          // Convert -o [3,1] to -> [0,1]
          C_tilde->setAmatVal(j,i,0);
        }
      } else {
        // For nodes that are in different neighborhoods
        // Convert circle markings label from "1" to "4"
        if (G_ij==1){
          C_tilde->setAmatVal(i,j,4);
          //C_tilde->setAmatVal(j,i,4); // TODO: WHY DID WE HAVE THIS BEFORE? 
        }
      }
      
    }
  }
}

// Convert the final graph after using efficient notation to the full
// adjacency matrix
void CML::convertFinalGraph(){
  // Create a graph with the full number of nodes
  Graph* g = new Graph(p);
  // Start with an empty graph
  g -> emptyGraph();
  size_t current_val = 0;
  for (size_t i=0;i<N;++i){
    for (size_t j=0;j<N;++j){
      current_val = C_tilde -> getAmatVal(i,j);
      g -> setAmatVal(neighborhood(i),neighborhood(j),current_val);  
    }
  }
  delete C_tilde;
  C_tilde = g;
  g = nullptr;
}

// TODO: Need to test this to make sure within-neighborhood bidirected edges don't produce warning
void CML::checkNotation(){
  size_t G_ij; size_t G_ji;
  
  for (size_t i=0;i<C_tilde->size();++i){
    for (size_t j=i+1;j<C_tilde->size();++j){
      G_ij = C_tilde->getAmatVal(i,j);
      G_ji = C_tilde->getAmatVal(j,i);
      if ((G_ij==0 || G_ij==1) && G_ji>1){
        warning("Ancestral marking mixed with neighborhood marking.");
      } else if ((G_ji==0 || G_ji==1) && G_ij>1){
        warning("Ancestral marking mixed with neighborhood marking.");
      }
    }
  }
}

void CML::run(){
  if (verbose){
    Rcout << "Beginning the";
    if (pop){
      Rcout << " population";
    }
    Rcout << " Local FCI algorithm over all neighborhoods.\n";
  }
  // Finding the skeleton for the complete undirected graph on X_T U N_T
  getSkeletonTotal(); 
  
  if (verbose){
    Rcout << "Beginning algorithm over each individual neighborhood.\n";
  }
  // Get the skeleton for each target node and its neighborhood
  
  std::for_each(targets.begin(),
                targets.end(),
                [this](size_t t){ getSkeletonTarget(t); });
  
  // Rule 0: Obtain V Structures
  rules_used(0) = getVStructures();
  
  // Remaining FCI Rules
  allRules();
  
  convertMixedGraph();
  
  convertFinalGraph();
  
  total_time = total_skeleton_time;
  total_time += std::accumulate(target_skeleton_times.begin(),
                                target_skeleton_times.end(),0);
}

void CML::run_mag(){
  if (verbose){
    Rcout << "Beginning the";
    if (pop){
      Rcout << " population";
    }
    Rcout << " Local FCI algorithm over all neighborhoods, not including the mixed graph conversion rules within neighborhoods.\n";
  }
  // Finding the skeleton for the complete undirected graph on X_T U N_T
  getSkeletonTotal(); 
  
  if (verbose){
    Rcout << "Beginning algorithm over each individual neighborhood.\n";
  }
  // Get the skeleton for each target node and its neighborhood
  
  std::for_each(targets.begin(),
                targets.end(),
                [this](size_t t){ getSkeletonTarget(t); });
  
  // Rule 0: Obtain V Structures
  rules_used(0) = getVStructures();
  
  // Remaining FCI Rules
  allRules();
  
  convertFinalGraph();
  
  total_time = total_skeleton_time;
  total_time += std::accumulate(target_skeleton_times.begin(),
                                target_skeleton_times.end(),0);
}

