// r_interface.cpp
// (c) 2020 David Merrell
//

#include "trial_mdp.h"
#include <string>
#include <iostream>
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::depends(BH)]]

// [[Rcpp::plugins("cpp11")]]


//' Use TrialMDP to compute an optimal trial design
//'
//' Given the number of patients, failure cost, and stage cost, compute an optimal trial design and save it to a SQLite database. 
//'
//' @param n_patients the number of patients in the trial
//' @param failure_cost parameter representing the cost of patient failures
//' @param block_cost parameter representing the cost of each additional trial stage
//' @param sqlite_fname output filepath for trial design SQLite database
//' @param min_size minimum size for a trial stage. Default=4
//' @param block_incr require trial stage sizes to be multiples of this number. Default=2
//' @param prior_a0 smoothing/pseudocount hyperparameter for computing transition probabilities. Default=1.0
//' @param prior_a1 smoothing/pseudocount hyperparameter for computing transition probabilities. Default=1.0
//' @param prior_b0 smoothing/pseudocount hyperparameter for computing transition probabilities. Default=1.0
//' @param prior_b1 smoothing/pseudocount hyperparameter for computing transition probabilities. Default=1.0
//' @param transition_dist name of transition probability distribution. Default="beta_binom". We do not recommend changing this.
//' @param test_statistic name of test statistic for which to optimize. Default="scaled_cmh". We do not recommend changing this.
//' @param act_l smallest allocation fraction to treatment A. i.e., Phi = {act_l, ..., act_u}. Default=0.2
//' @param act_u largest allocation fraction to treatment A. i.e., Phi = {act_l, ..., act_u}. Default=0.8
//' @param act_n number of possible allocation fractions, uniformly spaced. i.e., |Phi| = act_n. Default=7
//'
//' @return None. Trial design is written to disk.
// [[Rcpp::export]]
void trial_mdp(int n_patients,
               float failure_cost, float block_cost,
               std::string sqlite_fname,
               int min_size=4,
               int block_incr=2,
               float prior_a0 = 1.0,
               float prior_a1 = 1.0,
               float prior_b0 = 1.0,
               float prior_b1 = 1.0,
               std::string transition_dist="beta_binom",
               std::string test_statistic="scaled_cmh",
               float act_l=0.2, float act_u=0.8, int act_n=7) {


  TrialMDP solver = TrialMDP(n_patients,
                             failure_cost, block_cost,
                             min_size, block_incr,
                             prior_a0, prior_a1,
                             prior_b0, prior_b1,
                             transition_dist,
                             test_statistic,
                             act_l, act_u, act_n);
  
  std::cout << "Solver initialized." << std::endl;
  std::cout << "\tN patients: " << n_patients << std::endl; 
  std::cout << "\tMin block size: " << min_size << std::endl;
  std::cout << "\tBlock increment: " << block_incr << std::endl;
  std::cout << "\tAllocations: {" << act_l << ", ..., " << act_u << "} (" << act_n << ")" << std::endl; 
  std::cout << "\tFailure cost: " << failure_cost << std::endl; 
  std::cout << "\tBlock cost: " << block_cost << std::endl; 
  std::cout << "\tTest statistic: " << test_statistic << std::endl; 
  std::cout << "Solving." << std::endl;
  
  solver.solve();
  std::cout << "Solver completed." << std::endl;
  
  char* fname = new char[sqlite_fname.length() + 1];
  strcpy(fname, sqlite_fname.c_str());
  
  solver.to_sqlite(fname, 10000);
  std::cout << "Saved to file: " << sqlite_fname << std::endl;
  
  delete[] fname;
  
}


