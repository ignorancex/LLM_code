#include "SNL.h"

// [[Rcpp::export]]
int checkInitializeSNL(NumericMatrix td,arma::mat df,
                           NumericVector t,
                           NumericVector nodes_interest,
                           StringVector names){
  SNL snl(td,df,t,nodes_interest,names,3,0.01,true);
  Rcout << snl.getSize() << std::endl;
  return snl.getSize();
}


// [[Rcpp::export]]
NumericMatrix checkGetTargetSkel(NumericMatrix td,arma::mat df,
                                 NumericVector t,
                                 NumericVector nodes_interest,StringVector names){
  SNL snl(td,df,t,nodes_interest,names,3,0.01,true);
  snl.getSkeletonTarget(t(0));
  return snl.getAmat();
}

// [[Rcpp::export]]
NumericMatrix checkGetVStructures(NumericMatrix td,arma::mat df,
                                  NumericVector t,
                                  NumericVector nodes_interest,StringVector names){
  SNL snl(td,df,t,nodes_interest,names,3,0.02,true);
  snl.getSkeletonTarget(t(0));
  int rule_0_used = snl.getVStructures();
  Rcout << "Rules Used (0) " << rule_0_used << std::endl;
  return snl.getAmat();
}

// [[Rcpp::export]]
int checkInitializeSNLPop(NumericMatrix td,NumericVector t,
                              NumericVector nodes_interest,StringVector names){
  SNL snl(td,t,nodes_interest,names,3,true);
  Rcout << snl.getSize() << std::endl;
  return snl.getSize();
}

// [[Rcpp::export]]
NumericMatrix checkGetTargetSkelPop(NumericMatrix td,
                                    NumericVector t,
                                    NumericVector nodes_interest,
                                    StringVector names){
  SNL snl(td,t,nodes_interest,names,3,true);
  snl.getSkeletonTarget(t(0));
  return snl.getAmat();
}

// [[Rcpp::export]]
NumericMatrix checkGetVStructuresPop(NumericMatrix td,
                                     NumericVector t,
                                     NumericVector nodes_interest,
                                     StringVector names){
  SNL snl(td,t,nodes_interest,names,3,true);
  snl.getSkeletonTarget(t(0));
  int rule_0_used = snl.getVStructures();
  Rcout << "Rules Used (0) " << rule_0_used << std::endl;
  return snl.getAmat();
}

// [[Rcpp::export]]
NumericMatrix checkRule1(NumericMatrix td,
                         NumericMatrix test_mat,
                         NumericVector t,
                         NumericVector nodes_interest,
                         StringVector names){
  SNL snl(td,t,nodes_interest,names,3,true);
  snl.setAmat(test_mat);
  bool no_changes = true;
  snl.rule1(no_changes);
  if (no_changes){
    Rcout << "There was no change\n";
  } else {
    Rcout << "There was a change\n";
  }
  Rcout << "Rules Used (1) " << snl.getRulesUsed()(1) << std::endl;
  return snl.getAmat();
}

// [[Rcpp::export]]
NumericMatrix checkRule2(NumericMatrix td,
                         NumericMatrix test_mat,
                         NumericVector t,
                         NumericVector nodes_interest,
                         StringVector names){
  SNL snl(td,t,nodes_interest,names,3,true);
  snl.setAmat(test_mat);
  bool no_changes = true;
  snl.rule2(no_changes);
  if (no_changes){
    Rcout << "There was no change\n";
  } else {
    Rcout << "There was a change\n";
  }
  Rcout << "Rules Used (2) " << snl.getRulesUsed()(2) << std::endl;
  return snl.getAmat();
}

// [[Rcpp::export]]
NumericMatrix checkRule3(NumericMatrix td,
                         NumericMatrix test_mat,
                         NumericVector t,
                         NumericVector nodes_interest,
                         StringVector names){
  SNL snl(td,t,nodes_interest,names,3,true);
  snl.setAmat(test_mat);
  bool no_changes = true;
  snl.rule3(no_changes);
  if (no_changes){
    Rcout << "There was no change\n";
  } else {
    Rcout << "There was a change\n";
  }
  Rcout << "Rules Used (3) " << snl.getRulesUsed()(3) << std::endl;
  return snl.getAmat();
}

// [[Rcpp::export]]
NumericMatrix checkRule4(NumericMatrix td,
                         NumericMatrix test_mat,
                         NumericVector t,
                         NumericVector nodes_interest,
                         StringVector names){
  SNL snl(td,t,nodes_interest,names,3,true);
  snl.setAmat(test_mat);
  bool no_changes = true;
  snl.rule4(no_changes);
  if (no_changes){
    Rcout << "There was no change\n";
  } else {
    Rcout << "There was a change\n";
  }
  Rcout << "Rules Used (4) " << snl.getRulesUsed()(4) << std::endl;
  return snl.getAmat();
}

// [[Rcpp::export]]
NumericMatrix checkSNLRules(NumericMatrix td,
                            NumericMatrix test_mat,
                            NumericVector t,
                            NumericVector nodes_interest,
                            StringVector names){
  SNL snl(td,t,nodes_interest,names,3,true);
  snl.setAmat(test_mat);
  snl.meeksRules();
  NumericVector num_used = snl.getRulesUsed();
  int i=0;
  std::for_each(num_used.begin(),num_used.end(),[&i](int num){ Rcout << "Rule " << i++ << ": " << num << std::endl;});
  return snl.getAmat();
}

// [[Rcpp::export]]
NumericMatrix checkSNL(NumericMatrix td,arma::mat df,
                           NumericVector t,
                           NumericVector nodes_interest,StringVector names){
  SNL snl(td,df,t,nodes_interest,names,3,0.01,true);
  Rcout << "\n\n";
  // Get the skeleton for each target node and its neighborhood
  std::for_each(t.begin(),t.end(),[&snl](int t){ snl.getSkeletonTarget(t); });
  snl.getVStructures();
  snl.convertFinalGraph();
  return snl.getAmat();
}

// [[Rcpp::export]]
NumericMatrix checkSNLRun(NumericMatrix td,arma::mat df,
                              NumericVector t,
                              NumericVector nodes_interest,StringVector names){
  SNL snl(td,df,t,nodes_interest,names,3,0.01,true);
  snl.run();
  NumericVector num_used = snl.getRulesUsed();
  int i=0;
  std::for_each(num_used.begin(),num_used.end(),[&i](int num){ Rcout << "Rule " << i++ << ": " << num << std::endl;});
  return snl.getAmat();
}

// [[Rcpp::export]]
NumericMatrix checkSNLPop(NumericMatrix td,
                              NumericVector t,
                              NumericVector nodes_interest,
                              StringVector names){
  SNL snl(td,t,nodes_interest,names,3,true);
  std::for_each(t.begin(),t.end(),[&snl](int t){ snl.getSkeletonTarget(t); });
  snl.getVStructures();
  snl.convertFinalGraph();
  return snl.getAmat();
}
