#include "ConstrainedAlgo.h"

// Sample version
ConstrainedAlgo::ConstrainedAlgo(NumericMatrix true_dag,arma::mat df,
                                 NumericVector targets,
                                 NumericVector nodes_interest,
                                 StringVector names,
                                 int lmax,
                                 double signif_level,
                                 bool verbose,
                                 std::string test,
                                 bool estDAG) :
  lmax(lmax),targets(targets),names(names),df(df),
  signif_level(signif_level),test(test),verbose(verbose){
  
  num_targets = targets.length();
  if (verbose){
    Rcout << "There is (are) " << num_targets << " target(s).\n";
    printVecElements(targets,names,"Targets: ","\n");
  }
  p = true_dag.ncol();
  if (test=="testIndFisher"){
    R = arma::cor(df); 
  }
  n = df.n_rows;
  num_tests=0;
  
  // Set DAG object (We use this object to obtain neighborhoods)
  true_DAG = new DAG(p,names,true_dag);
  
  if (estDAG){
    // We are given a matrix containing Markov Blanket information
    mb_list = new MBList(nodes_interest,true_dag,verbose);
  } else {
    // We are given the true DAG to find actual Markov Blankets
    mb_list = new MBList(nodes_interest,true_DAG,verbose);
  }
  
  // Find the neighborhoods of the target nodes (targets U nbhd(targets))
  neighborhood = mb_list -> getMBMultipleTargets(targets); 
  neighborhood = union_(neighborhood,targets);
  std::sort(neighborhood.begin(),neighborhood.end());
  N = neighborhood.size();
  
  if (verbose){
    Rcout << "There are " << p << " nodes in the DAG.\n";
    Rcout << "There are " << N << " nodes in the neighborhoods we are considering.\n";
    Rcout << "All nodes being considered: ";
    printVecElementsNoNames(neighborhood,"","\n"," ");
  }
  
  // Initial graph that will be modified through the process of the algorithm
  C_tilde = new Graph(N);
  
  if (verbose){
    Rcout << "Our starting matrix is " << C_tilde->getNRow();
    Rcout << "x" << C_tilde->getNCol() << ".\n";
    C_tilde -> printAmat();
    Rcout << "\n\n";
  }
  
  // Create the list that will store separating sets
  S = new SepSetList(neighborhood);
  if (verbose){
    Rcout << "\nOur initial separating sets:\n";
    S -> printSepSetList();
    Rcout << std::endl;
    Rcout << "We are using test: " << test << std::endl;
  }
}

// Population version of the algorithm object
ConstrainedAlgo::ConstrainedAlgo(NumericMatrix true_dag,
                                 NumericVector targets,
                                 NumericVector nodes_interest,
                                 StringVector names,
                                 int lmax,
                                 bool verbose) : 
  lmax(lmax),targets(targets),
  names(names),verbose(verbose) {
  num_targets = targets.length();
  if (verbose){
    Rcout << "Population Version (C++):\n";
    Rcout << "There are " << num_targets << " targets.\n";
    printVecElements(targets,names,"Targets: ","\n");
  }
  p = true_dag.ncol();
  pop = true;
  num_tests=0;
  
  // Set DAG object
  true_DAG = new DAG(p,names,true_dag);
  // Use the true DAG to find neighborhoods
  mb_list = new MBList(nodes_interest,true_DAG,verbose);
  
  // Find the neighborhoods of the target nodes (targets U nbhd(targets))
  neighborhood = mb_list -> getMBMultipleTargets(targets);
  neighborhood = union_(neighborhood,targets);
  std::sort(neighborhood.begin(),neighborhood.end());
  
  N = neighborhood.size();
  
  if (verbose){
    Rcout << "There are " << p << " nodes in the DAG.\n";
    Rcout << "There are " << N << " nodes in the neighborhood.\n";
    Rcout << "All nodes being considered: ";
    printVecElementsNoNames(neighborhood,"","\n"," ");
  }
  
  // Initial graph that will be modified through the process of the algorithm
  C_tilde = new Graph(N);
  
  if (verbose){
    Rcout << "Our starting matrix is ";
    Rcout << C_tilde->getNRow() << "x" << C_tilde->getNCol() << ".\n";
    C_tilde -> printAmat();
    Rcout << "\n\n";
  }
  
  // Create the list that will store separating sets
  S = new SepSetList(neighborhood);
  if (verbose){
    Rcout << "\nOur initial separating sets:\n";
    S -> printSepSetList();
  }
}

void ConstrainedAlgo::print_elements(){
  Rcout << "p: " << p << std::endl;
  Rcout << "n: " << n << std::endl;
  Rcout << "N: " << N << std::endl;
  Rcout << "Number of Targets: " << num_targets << std::endl;
  Rcout << "Node names: ";
  std::for_each(names.begin(),names.end(),[](String n) {
    Rcout << n.get_cstring() << " ";}
  );
  Rcout << std::endl;
  Rcout << "lmax: " << lmax << std::endl;
  Rcout << "verbose: " << verbose << std::endl;
  Rcout << "Nodes under consideration: ";
  printVecElementsNoNames(neighborhood);
  Rcout << std::endl << "Ctilde:\n";
  Rcout << "Our Ctilde matrix is ";
  Rcout << C_tilde->getNRow() << "x" << C_tilde->getNCol() << std::endl;
  C_tilde->printAmat();
  Rcout << "Our DAG matrix is " << std::endl;
  true_DAG->printAmat();
  Rcout << "Our Markov Blankets are" << std::endl;
  mb_list->printMBs();
  Rcout << "Separating Set Values:\n";
  S->printSepSetList();
  Rcout << "Number of tests so far: " << num_tests << std::endl;
  if (n>0){
    Rcout << "First and last elements of the dataset: ";
    Rcout << df(0,0) << " " << df(df.n_rows-1,df.n_cols-1) << std::endl;
  }
}

// i and j are efficient values and must be transformed to true values
// kvals are true values and do not need to be transformed
void ConstrainedAlgo::checkSeparation(int l,size_t i,size_t j,
                                      NumericMatrix kvals){
  size_t k;
  size_t kp = kvals.cols();
  bool keep_checking_k;
  List test_result;
  
  // Initially assumes we are considering an empty set
  arma::uvec sep_arma;
  
  if (l == 0){
    sep = NA_REAL;
    if (pop){
      NumericMatrix g = true_DAG -> getAmat();
      test_result = condIndTestPop(g,neighborhood(i),neighborhood(j),sep_arma);
    } else {
      
      if (test=="testIndFisher"){
        test_result = condIndTest(R,neighborhood(i),neighborhood(j),
                                  sep_arma,n,signif_level); 
      }
      else if (test == "gSquare"){
        test_result = condInttestdis(df,neighborhood(i),neighborhood(j),
                                  sep_arma,signif_level);
      }
    }
    ++num_tests;
    p_vals.push_back(test_result["pval"]);
    if (verbose){
      Rcout << "The p-value is " << p_vals[p_vals.size()-1] << std::endl;
    }
    if (test_result["result"]){ 
      // We have statistically significant evidence of conditional independence
      if (verbose){
        Rcout << names(neighborhood(i)) << " is separated from ";
        Rcout << names(neighborhood(j));
        Rcout << " (p-value>" << signif_level << ")"<< std::endl;
      }
      S->changeList(i,j);
      S->changeList(j,i);
      
      C_tilde->setAmatVal(i,j,0);
      C_tilde->setAmatVal(j,i,0);
    }
  } else {
    k = 0;
    keep_checking_k = true;
    while (keep_checking_k && (k<kp)){
      // sep is the vector of the true values of the potential separating nodes
      // (i.e. not efficient numbering)
      sep = kvals( _ , k );
      std::sort(sep.begin(),sep.end());
      sep_arma = as<arma::uvec>(sep);
      if (verbose){
        Rcout << "Efficient Setup: " << i << " -> " << neighborhood(i);
        Rcout << " | " << j << " -> " << neighborhood(j);
        Rcout << " | k (True Vals): ";
        printVecElementsNoNames(sep,""," ("," ");
        printVecElements(sep,names,"",")\n");
      }
      if (pop){
        test_result = condIndTestPop(true_DAG->getAmat(),
                                     neighborhood(i),neighborhood(j),
                                     sep_arma);
      } else {
        
        if (test=="testIndFisher"){
          test_result = condIndTest(R,neighborhood(i),neighborhood(j),
                                    sep_arma,n,signif_level); 
        }
        else if (test == "gSquare"){
          test_result = condInttestdis(df,neighborhood(i),neighborhood(j),
                                       sep_arma,signif_level);
        }
      }
      ++num_tests;
      p_vals.push_back(test_result["pval"]);
      //pval = as<double>(get_pval(i,j,true_dag,names,sep));
      if (verbose){
        Rcout << "The p-value is " << p_vals[p_vals.size()-1] << std::endl;
      }
      if (test_result["result"]){
        if (verbose){
          Rcout << names(neighborhood(i)) << " is separated from ";
          Rcout << names(neighborhood(j)) << " by node(s): ";
          printVecElements(sep,names,""," ");
          Rcout << " (p-value>" << signif_level << ")"<< std::endl;
        }
        S->changeList(i,j,sep);
        S->changeList(j,i,sep);
        C_tilde->setAmatVal(i,j,0);
        C_tilde->setAmatVal(j,i,0);
        keep_checking_k = false;
      } else {
        if (verbose){
          Rcout << names(neighborhood(i)) << " is NOT separated from ";
          Rcout << names(neighborhood(j)) << " by node(s): ";
          printVecElements(sep,names,""," ");
          Rcout << " (p-value<" << signif_level << ")"<< std::endl;
        }
      }
      ++k; // Increment to obtain the next potential separating set
    }
  }
}

// We are trying to identify structures i -> k <- j
// Where i and j are not adjacent, and k is not in the separating set of i and j
int ConstrainedAlgo::getVStructures() {
  int times_used = 0;
  bool no_neighbors;
  bool j_invalid;
  NumericVector placeholder;
  
  NumericVector i_adj;
  NumericVector j_adj;
  NumericVector j_vals;
  NumericVector k_vals;
  
  if (verbose){
    Rcout << "Beginning loops to find v-structures.\n";
  }
  // We are searching for i-k-j where i and j are not adjacent and k is
  // not in the separating set for i and j
  for (size_t i=0;i<N;++i){
    // We will search this vector for nodes connected to node i
    placeholder = C_tilde->getAmatRow(i); 
    no_neighbors = (all(placeholder==0)).is_true();
    if (!no_neighbors){ // If there are neighbors to consider
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
        // or if i and j are from separate neighborhoods
        placeholder = C_tilde->getAmatRow(j);
        j_invalid = (all(placeholder==0)).is_true();
        j_invalid = j_invalid || (C_tilde->getAmatVal(j,i)!= 0) ||  j <= i;
        // j must be in the neighborhood of i for it to make a v-structure with it (Local PC)
        j_invalid = j_invalid || !(mb_list->inMB(neighborhood(i),
                                                 neighborhood(j)));
        if (!j_invalid){
          if (verbose){
            Rcout << "j: " << j << " (" << names(neighborhood(j)) << ")"<< std::endl;
          }
          j_adj = C_tilde->getAdjacent(j);
          k_vals = intersect(i_adj,j_adj); // k must be a neighbor of i and j
          // If there are no common neighbors, move to next j
          if (k_vals.length()!=0){
            if (verbose){
              Rcout << "Potential k values: ";
              printVecElements(k_vals,names[neighborhood],"","\n");
            }
            std::sort(k_vals.begin(),k_vals.end());
            // We loop through all of the common neighbors
            for (auto k : k_vals){
              if (verbose){
                Rcout << "k: " << k << " (" << names(neighborhood(k)) << ")\n";
              }
              // Verify if k is in separating set for i and j
              size_t k_eff = k; // efficient numbering of k
              k = neighborhood(k); // Switch k to true numbering
              if (S->isPotentialVStruct(i,j,k)){
                if (verbose){
                  NumericVector sepset_ij = S->getSepSet(i,j);
                  Rcout << "Separation Set: ";
                  printVecElementsNoNames(sepset_ij);
                  Rcout << " | V-Structure (True Numbering): ";
                  Rcout << neighborhood(i) << "*->" << k << "<-*";
                  Rcout << neighborhood(j) << std::endl;
                }
                C_tilde->setAmatVal(i,k_eff,1);
                C_tilde->setAmatVal(j,k_eff,1);
                C_tilde->setAmatVal(k_eff,i,0);
                C_tilde->setAmatVal(k_eff,j,0);
                ++times_used;
              }
            }
          }
        }
      }
    }
  }
  return times_used;
}

