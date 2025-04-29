#include "MBList.h"

// [[Rcpp::export]]
void testInitializeMBList(NumericVector nodes,NumericMatrix mb_mat){
  MBList my_mb_list = MBList(nodes,mb_mat,true);
}

// [[Rcpp::export]]
void testInitializeMBListPop(NumericVector nodes,NumericMatrix amat){
  StringVector node_names = StringVector::create();
  size_t p = amat.nrow();
  makeNodeNames(p,node_names);
  DAG* my_dag = new DAG(p,node_names,amat,false);
  MBList my_mb_list = MBList(nodes,my_dag,true);
  delete my_dag;
  my_dag = nullptr;
}

// [[Rcpp::export]]
NumericVector testAccessMB(NumericVector nodes,NumericMatrix mb_mat,size_t i){
  MBList mb_list = MBList(nodes,mb_mat,false);
  return mb_list.getMB(i);
}

// [[Rcpp::export]]
NumericVector testAccessMultipleMB(NumericVector nodes,NumericMatrix mb_mat,
                                   NumericVector v,bool include_targets=false,
                                   bool exclude_targets=false,
                                   bool verbose=false){
  MBList mb_list = MBList(nodes,mb_mat,verbose);
  return mb_list.getMBMultipleTargets(v,include_targets,exclude_targets);
}

// [[Rcpp::export]]
bool testIsMBMember(NumericVector nodes,NumericMatrix mb_mat,
                    size_t target,size_t i){
  MBList mb_list = MBList(nodes,mb_mat,false);
  return mb_list.inMB(target,i);
}

// [[Rcpp::export]]
void testSilencer(NumericVector nodes,NumericMatrix mb_mat,
                  double target,double i){
  NumericVector a {target,i};
  NumericVector b {i,target};
  MBList mb_list = MBList(nodes,mb_mat,true);
  // Should print output
  Rcout << "Round 1:\n";
  NumericVector tmp = mb_list.getMBMultipleTargets(a);
  bool tmp1 = mb_list.silencer();
  // Should not print output
  Rcout << "Round 2:\n";
  tmp = mb_list.getMBMultipleTargets(b);
  mb_list.removeSilencer(tmp1);
  // Should print output
  Rcout << "Round 3:\n";
  tmp = mb_list.getMBMultipleTargets(b);
  
  return;
}


