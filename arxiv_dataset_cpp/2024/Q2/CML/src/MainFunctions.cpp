#include <chrono>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include "CML.h"
#include "SNL.h"

using namespace std::chrono;
using namespace Rcpp;

// [[Rcpp::export]]
List sampleCML(NumericMatrix true_dag,arma::mat df,
               NumericVector targets,
               NumericVector nodes_interest,
               StringVector names,int lmax=3,
               double signif_level = 0.01,
               bool verbose=true,
               std::string test="testIndFisher",
               bool estDAG=false){
  // Variable to keep track of timing
  auto start = high_resolution_clock::now();
  
  // Instantiate the Local FCI object
  CML cml(true_dag,df,targets,nodes_interest,names,lmax,signif_level,verbose,test,estDAG);
  
  cml.run(); 
  
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(end-start);
  double total_time = duration.count() / 1e6;  // Get time in seconds
  
  // Ensure we have proper notation for every edge
  cml.checkNotation();
  
  return List::create(
    _["G"]=cml.getAmat(),
    _["S"]=cml.getSepSetList(),
    _["NumTests"]=cml.getNumTests(),
    _["RulesUsed"]=cml.getRulesCount(),
    _["allNodes"]=cml.getNeighborhood(),
    _["totalSkeletonTime"]=cml.getTotalSkeletonTime(),
    _["targetSkeletonTimes"]=cml.getTargetSkeletonTimes(),
    _["algorithmTotalTime"]=cml.getTotalTime(),
    _["totalTime"]=total_time
  );
}

// [[Rcpp::export]]
List popCML(NumericMatrix true_dag,
            NumericVector targets,
            NumericVector nodes_interest,
            StringVector names,int lmax=3,
            bool verbose=true){
  // Variable to keep track of timing
  auto start = high_resolution_clock::now();
  
  // Instantiate the Local FCI object
  CML cml(true_dag,targets,nodes_interest,names,lmax,verbose);
  
  if (verbose){
    Rcout << "Beginning algorithm over all neighborhoods.\n";
  }
  
  cml.run();
  
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(end-start);
  double total_time = duration.count() / 1e6;
  total_time /= 60; // Get time in minutes
  
  // Ensure we have proper notation for every edge
  cml.checkNotation();
  
  return List::create(
    _["G"]=cml.getAmat(),
    _["S"]=cml.getSepSetList(),
    _["NumTests"]=cml.getNumTests(),
    _["RulesUsed"]=cml.getRulesCount(),
    _["allNodes"]=cml.getNeighborhood(),
    _["totalSkeletonTime"]=cml.getTotalSkeletonTime(),
    _["targetSkeletonTimes"]=cml.getTargetSkeletonTimes(),
    _["totalTime"]=total_time
  );
}

// [[Rcpp::export]]
List sampleCML_mag(NumericMatrix true_dag,arma::mat df,
                   NumericVector targets,
                   NumericVector nodes_interest,
                   StringVector names,int lmax=3,
                   double signif_level = 0.01,
                   bool verbose=true,
                   std::string test="testIndFisher",
                   bool estDAG=false){
  // Variable to keep track of timing
  auto start = high_resolution_clock::now();
  
  // Instantiate the Local FCI object
  CML cml(true_dag,df,targets,nodes_interest,names,lmax,signif_level,verbose,test,estDAG);
  
  cml.run_mag(); 
  
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(end-start);
  double total_time = duration.count() / 1e6;  // Get time in seconds
  
  // Ensure we have proper notation for every edge
  cml.checkNotation();
  
  return List::create(
    _["G"]=cml.getAmat(),
    _["S"]=cml.getSepSetList(),
    _["NumTests"]=cml.getNumTests(),
    _["RulesUsed"]=cml.getRulesCount(),
    _["allNodes"]=cml.getNeighborhood(),
    _["totalSkeletonTime"]=cml.getTotalSkeletonTime(),
    _["targetSkeletonTimes"]=cml.getTargetSkeletonTimes(),
    _["algorithmTotalTime"]=cml.getTotalTime(),
    _["totalTime"]=total_time
  );
}

// [[Rcpp::export]]
List popCML_mag(NumericMatrix true_dag,
            NumericVector targets,
            NumericVector nodes_interest,
            StringVector names,int lmax=3,
            bool verbose=true){
  // Variable to keep track of timing
  auto start = high_resolution_clock::now();
  
  // Instantiate the Local FCI object
  CML cml(true_dag,targets,nodes_interest,names,lmax,verbose);
  
  if (verbose){
    Rcout << "Beginning algorithm over all neighborhoods.\n";
  }
  
  cml.run_mag();
  
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(end-start);
  double total_time = duration.count() / 1e6;
  total_time /= 60; // Get time in minutes
  
  // Ensure we have proper notation for every edge
  cml.checkNotation();
  
  return List::create(
    _["G"]=cml.getAmat(),
    _["S"]=cml.getSepSetList(),
    _["NumTests"]=cml.getNumTests(),
    _["RulesUsed"]=cml.getRulesCount(),
    _["allNodes"]=cml.getNeighborhood(),
    _["totalSkeletonTime"]=cml.getTotalSkeletonTime(),
    _["targetSkeletonTimes"]=cml.getTargetSkeletonTimes(),
    _["totalTime"]=total_time
  );
}

// [[Rcpp::export]]
List sampleSNL(NumericMatrix true_dag,arma::mat df,
               NumericVector targets,
               NumericVector nodes_interest,
               StringVector names,int lmax=3,
               double signif_level = 0.01,
               bool verbose=true,
               std::string test="testIndFisher",
               bool estDAG=false){
  // Variable to keep track of timing
  auto start = high_resolution_clock::now();
  
  // Instantiate the Local PC object
  SNL snl(true_dag,df,targets,nodes_interest,names,lmax,signif_level,verbose,test,estDAG);
  
  snl.run();
  
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(end-start);
  double total_time = duration.count() / 1e6;
  total_time /= 60; // Get time in minutes
  
  return List::create(
    _["G"]=snl.getAmat(),
    _["S"]=snl.getSepSetList(),
    _["NumTests"]=snl.getNumTests(),
    _["allNodes"]=snl.getNeighborhood(),
    _["rulesUsed"]=snl.getRulesUsed(),
    _["targetSkeletonTimes"]=snl.getTargetSkeletonTimes(),
    _["algorithmTotalTime"]=snl.getTotalTime(),
    _["totalTime"]=total_time
  );
}

// [[Rcpp::export]]
List popSNL(NumericMatrix true_dag,
            NumericVector targets,
            NumericVector nodes_interest,
            StringVector names,int lmax=3,
            bool verbose=true){
  // Variable to keep track of timing
  auto start = high_resolution_clock::now();
  
  //Instantiate the Local FCI object
  SNL snl(true_dag,targets,nodes_interest,names,lmax,verbose);
  
  if (verbose){
    Rcout << "Beginning algorithm over all neighborhoods.\n";
  }
  
  snl.run();
  
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(end-start);
  double total_time = duration.count() / 1e6;
  total_time /= 60; // Get time in minutes
  
  return List::create(
    _["G"]=snl.getAmat(),
    _["S"]=snl.getSepSetList(),
    _["NumTests"]=snl.getNumTests(),
    _["allNodes"]=snl.getNeighborhood(),
    _["rulesUsed"]=snl.getRulesUsed(),
    _["targetSkeletonTimes"]=snl.getTargetSkeletonTimes(),
    _["totalTime"]=total_time
  );
}

