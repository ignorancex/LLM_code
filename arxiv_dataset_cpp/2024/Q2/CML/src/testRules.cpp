#include "CML.h"
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix testRule1(NumericMatrix td,arma::mat dummy_df,NumericVector dummy_t,StringVector names,NumericMatrix m){
  NumericVector nodes_interest;
  for (int i=0;i<td.ncol();++i){
    nodes_interest.push_back(i);
  }
  CML cml(td,dummy_df,dummy_t,nodes_interest,names,3,0.01,false);
  cml.setAmat(m);
  bool track_changes=false;
  cml.setVerboseTrue();
  cml.rule1(track_changes);
  Rcout << "Track changes: " << track_changes << std::endl;
  return cml.getAmat();
}

// [[Rcpp::export]]
NumericMatrix testRule2(NumericMatrix td,arma::mat dummy_df,NumericVector dummy_t,StringVector names,NumericMatrix m){
  NumericVector nodes_interest;
  for (int i=0;i<td.ncol();++i){
    nodes_interest.push_back(i);
  }
  CML cml(td,dummy_df,dummy_t,nodes_interest,names,3,0.01,false);
  cml.setAmat(m);
  bool track_changes=false;
  cml.setVerboseTrue();
  cml.rule2(track_changes);
  Rcout << "Track changes: " << track_changes << std::endl;
  return cml.getAmat();
}

// [[Rcpp::export]]
NumericMatrix testRule3(NumericMatrix td,arma::mat dummy_df,NumericVector dummy_t,StringVector names,NumericMatrix m){
  NumericVector nodes_interest;
  for (int i=0;i<td.ncol();++i){
    nodes_interest.push_back(i);
  }
  CML cml(td,dummy_df,dummy_t,nodes_interest,names,3,0.01,false);
  cml.setAmat(m);
  bool track_changes=false;
  cml.setVerboseTrue();
  cml.rule3(track_changes);
  Rcout << "Track changes: " << track_changes << std::endl;
  return cml.getAmat();
}

// [[Rcpp::export]]
NumericMatrix testRule4(NumericMatrix td,arma::mat dummy_df,NumericVector dummy_t,StringVector names,NumericMatrix m,int i,int j,NumericVector k){
  NumericVector nodes_interest;
  for (int i=0;i<td.ncol();++i){
    nodes_interest.push_back(i);
  }
  CML cml(td,dummy_df,dummy_t,nodes_interest,names,3,0.01,false);
  cml.setAmat(m);
  cml.setS(i,j,k);
  cml.setS(j,i,k);
  bool track_changes=false;
  cml.setVerboseTrue();
  cml.rule4(track_changes);
  Rcout << "Track changes: " << track_changes << std::endl;
  return cml.getAmat();
}

// [[Rcpp::export]]
NumericMatrix testRule8(NumericMatrix td,arma::mat dummy_df,NumericVector dummy_t,StringVector names,NumericMatrix m){
  NumericVector nodes_interest;
  for (int i=0;i<td.ncol();++i){
    nodes_interest.push_back(i);
  }
  CML cml(td,dummy_df,dummy_t,nodes_interest,names,3,0.01,false);
  cml.setAmat(m);
  bool track_changes=false;
  cml.setVerboseTrue();
  cml.rule8(track_changes);
  Rcout << "Track changes: " << track_changes << std::endl;
  return cml.getAmat();
}

// [[Rcpp::export]]
NumericMatrix testRule9(NumericMatrix td,arma::mat dummy_df,NumericVector dummy_t,StringVector names,NumericMatrix m){
  NumericVector nodes_interest;
  for (int i=0;i<td.ncol();++i){
    nodes_interest.push_back(i);
  }
  CML cml(td,dummy_df,dummy_t,nodes_interest,names,3,0.01,false);
  cml.setAmat(m);
  bool track_changes=false;
  cml.setVerboseTrue();
  cml.rule9(track_changes);
  Rcout << "Track changes: " << track_changes << std::endl;
  return cml.getAmat();
}

// [[Rcpp::export]]
NumericMatrix testRule10(NumericMatrix td,arma::mat dummy_df,NumericVector dummy_t,StringVector names,NumericMatrix m){
  NumericVector nodes_interest;
  for (int i=0;i<td.ncol();++i){
    nodes_interest.push_back(i);
  }
  CML cml(td,dummy_df,dummy_t,nodes_interest,names,3,0.01,false);
  cml.setAmat(m);
  bool track_changes=false;
  cml.setVerboseTrue();
  cml.rule10(track_changes);
  Rcout << "Track changes: " << track_changes << std::endl;
  return cml.getAmat();
}

// [[Rcpp::export]]
NumericMatrix testAllRules(NumericMatrix td,arma::mat dummy_df,NumericVector dummy_t,StringVector names,NumericMatrix m){
  NumericVector nodes_interest;
  for (int i=0;i<td.ncol();++i){
    nodes_interest.push_back(i);
  }
  CML cml(td,dummy_df,dummy_t,nodes_interest,names,3,0.01,false);
  cml.setAmat(m);
  cml.setVerboseTrue();
  cml.allRules();
  return cml.getAmat();
}

// [[Rcpp::export]]
NumericMatrix testConvertMixed(NumericMatrix td,NumericVector t,StringVector names,NumericMatrix m,NumericVector v){
  NumericVector nodes_interest;
  for (int i=0;i<td.ncol();++i){
    nodes_interest.push_back(i);
  }
  CML cml(td,t,nodes_interest,names,3,true);
  cml.setAmat(m);
  cml.setNeighbors(v);
  cml.convertMixedGraph();
  return cml.getAmat();
}


