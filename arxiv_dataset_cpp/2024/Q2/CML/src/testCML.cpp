#include "CML.h"
using namespace Rcpp;


// [[Rcpp::export]]
void initializeCML(NumericMatrix td,arma::mat df,NumericVector t,
                   NumericVector nodes_interest,StringVector names){
  CML cml(td,df,t,nodes_interest,names,3,0.01,true);
  Rcout << "\n\n";
  cml.print_elements();
}

// [[Rcpp::export]]
void initializeCMLPop(NumericMatrix td,NumericVector t,
                      NumericVector nodes_interest,StringVector names){
  CML cml(td,t,nodes_interest,names,3,true);
  Rcout << "\n\n";
  cml.print_elements();
}

// [[Rcpp::export]]
int getSizeCML(NumericMatrix td,arma::mat df,NumericVector t,
               NumericVector nodes_interest,StringVector names){
  CML cml(td,df,t,nodes_interest,names,3,0.01,true);
  return cml.getSize();
}

// [[Rcpp::export]]
List setSCML(NumericMatrix td,arma::mat df,NumericVector t,
             NumericVector nodes_interest,StringVector names,
             size_t i,size_t j,NumericVector k){
  CML cml(td,df,t,nodes_interest,names,3,0.01,true);
  cml.setS(i,j,k);
  return cml.getSepSetList();
}

// [[Rcpp::export]]
void setVerboseCML(NumericMatrix td,arma::mat df,NumericVector t,
                   NumericVector nodes_interest,StringVector names){
  CML cml(td,df,t,nodes_interest,names,3,0.01,false);
  Rcout << "CML Verbose: " << cml.getVerbose() << std::endl;
  cml.setVerboseTrue();
  Rcout << "CML Verbose: " << cml.getVerbose() << std::endl;
}

// [[Rcpp::export]]
NumericMatrix checkSkeletonTotal(NumericMatrix td,arma::mat df,NumericVector t,
                                 NumericVector nodes_interest,StringVector names){
  CML cml(td,df,t,nodes_interest,names,3,0.01,true);
  Rcout << "\n\n";
  cml.getSkeletonTotal();
  return cml.getAmat();
}

// [[Rcpp::export]]
NumericMatrix checkSkeletonTotalPop(NumericMatrix td,NumericVector t,
                                    NumericVector nodes_interest,StringVector names){
  CML cml(td,t,nodes_interest,names,3,true);
  Rcout << "\n\n";
  cml.getSkeletonTotal();
  return cml.getAmat();
}

// [[Rcpp::export]]
NumericMatrix checkVStruct(NumericMatrix td,arma::mat df,NumericVector t,
                           NumericVector nodes_interest,StringVector names){
  CML cml(td,df,t,nodes_interest,names,3,0.01,true);
  Rcout << "\n\n";
  cml.getSkeletonTotal();
  // Get the skeleton for each target node and its neighborhood
  std::for_each(t.begin(),t.end(),[&cml](int t){ cml.getSkeletonTarget(t); });
  cml.getVStructures();
  return cml.getAmat();
}

// [[Rcpp::export]]
NumericMatrix checkVStructPop(NumericMatrix td,NumericVector t,
                              NumericVector nodes_interest,StringVector names){
  CML cml(td,t,nodes_interest,names,3,true);
  Rcout << "\n\n";
  cml.getSkeletonTotal();
  // Get the skeleton for each target node and its neighborhood
  std::for_each(t.begin(),t.end(),[&cml](int t){ cml.getSkeletonTarget(t); });
  cml.getVStructures();
  cml.convertMixedGraph();
  cml.convertFinalGraph();
  return cml.getAmat();
}

/*
 * TODO: This needs to be checked more rigorously with an appropriate test graph
 */
// [[Rcpp::export]]
NumericMatrix checkAdjMatConversion(NumericMatrix td,arma::mat df,NumericVector t,
                                    NumericVector nodes_interest,StringVector names,
                                    NumericMatrix m,NumericVector neighbors){
  CML cml(td,df,t,nodes_interest,names,3,0.01,true);
  cml.setAmat(m);
  cml.setNeighbors(neighbors);
  cml.convertMixedGraph();
  cml.convertFinalGraph();
  
  Rcout << "Final\n";
  cml.print_elements();
  return cml.getAmat();
}

// [[Rcpp::export]]
NumericMatrix checkNotationWarnings(NumericMatrix td,arma::mat df,NumericVector t,
                                    NumericVector nodes_interest,StringVector names,
                                    NumericMatrix m){
  CML cml(td,df,t,nodes_interest,names,3,0.01,true);
  cml.setAmat(m);
  cml.checkNotation();
  
  Rcout << "Final\n";
  cml.print_elements();
  return cml.getAmat();
}

// [[Rcpp::export]]
double checkSeparationTest(NumericMatrix td,arma::mat df,NumericVector t,
                           NumericVector nodes_interest,StringVector names,
                           int i,int j,int l,NumericVector nodes_to_skip){ 
  CML cml(td,df,t,nodes_interest,names,3,0.01,true);
  NumericVector edges_i = cml.getAdjacent(i);
  // Find neighbors of i and j from the current graph C
  NumericVector neighbors = setdiff(union_(edges_i,cml.getAdjacent(j)),NumericVector::create(i,j));
  neighbors = setdiff(neighbors,nodes_to_skip);
  if (neighbors.length()>= l){
    if (l>0){
      if (neighbors.length() > 1){
        Rcout << "There are " << neighbors.length() << " neighbors.\n";
      } else {
        Rcout << "There is " << neighbors.length() << " neighbor.\n";
      }
    }
    NumericMatrix kvals = combn_cpp(cml.getNeighborhood()[neighbors],l);
    // check whether nodes i and j are separated by any of the potential separating sets in kvals
    cml.checkSeparation(l,i,j,kvals);
  }
  
  return cml.getMostRecentPVal();
}

// [[Rcpp::export]]
NumericMatrix checkCMLSummary(NumericMatrix td,arma::mat df,
                              NumericVector targets,
                              NumericVector nodes_interest,StringVector names){
  // Instantiate the Local FCI object
  CML cml(td,df,targets,nodes_interest,names,3,0.05,false);
  cml.getSkeletonTotal();
  std::for_each(targets.begin(),
                targets.end(),
                [&cml](int t){ cml.getSkeletonTarget(t); });
  
  // Rule 0: Obtain V Structures
  cml.getVStructures();
  
  // Remaining FCI Rules
  cml.allRules();
  
  cml.convertMixedGraph();
  
  cml.convertFinalGraph();
  cml.print_elements();
  
  return cml.getAmat();
}

// [[Rcpp::export]]
NumericMatrix checkCMLSummaryPop(NumericMatrix td,
                                 NumericVector targets,
                                 NumericVector nodes_interest,StringVector names){
  // Instantiate the Local FCI object
  CML cml(td,targets,nodes_interest,names,3,false);
  cml.getSkeletonTotal();
  std::for_each(targets.begin(),
                targets.end(),
                [&cml](int t){ cml.getSkeletonTarget(t); });
  
  // Rule 0: Obtain V Structures
  cml.getVStructures();
  
  // Remaining FCI Rules
  cml.allRules();
  
  cml.convertMixedGraph();
  
  cml.convertFinalGraph();
  cml.print_elements();
  
  return cml.getAmat();
}
