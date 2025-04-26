// transition_dist.cpp
// (c) 2021-01 David Merrell
//
// Implementation of TransitionDist class
// and its subclasses.

#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1

#include "transition_dist.h"
#include "contingency_table.h"
#include <cmath>
#include <string>

//// If we're on __linux__, use the std::beta function
//#ifdef __linux__
//
//float my_beta(float a, float b){
//    return std::beta(a,b);
//}
//
//// If we're on __APPLE__ or _WIN32, use the boost::math::beta function
//// (which seems to be slower...)
//#else

#include <boost/math/special_functions/beta.hpp>
float my_beta(float a, float b){
    return boost::math::beta(a,b);
}

//#endif 


////////////////////////////////
// Beta distribution
////////////////////////////////

float binom_coeff(int n, int k){
    return 1.0/((n+1.0)*my_beta(n-k+1,k+1));
}

float binom_prob(int N, float p, int x){
    return binom_coeff(N,x) * pow(p,x) * pow(1.0 - p, N-x);
}

std::vector<float> initialize_binom_probs(int N, float p){
    std::vector<float> probs = std::vector<float>(N+1, 0);
    for(int i=0; i < N+1; i++){
        probs[i] = binom_prob(N, p, i);
    }
    return probs;
}


BinomTransitionDist::BinomTransitionDist(float pr_a0, float pr_a1,
                                         float pr_b0, float pr_b1){
    prior_a0 = pr_a0;
    prior_a1 = pr_a1;
    prior_b0 = pr_b0;
    prior_b1 = pr_b1;
}

void BinomTransitionDist::set_state_action(ContingencyTable ct, 
                                           short unsigned int a_size,
                                           short unsigned int b_size){
    // compute smoothed point estimates
    float a_smoothing = prior_a0 + prior_a1;
    float b_smoothing = prior_b0 + prior_b1;

    float p_a = (float(ct.a1) + prior_a1) / (float(ct.a0 + ct.a1) + a_smoothing);
    a_probs = initialize_binom_probs(a_size, p_a);

    float p_b = (float(ct.b1) + prior_b1) / (float(ct.b0 + ct.b1) + b_smoothing);
    b_probs = initialize_binom_probs(b_size, p_b);

}


////////////////////////////////
// Beta-Binomial distribution
////////////////////////////////

float beta_binom_prob(int N, float pr_0, float pr_1, int x){
    return binom_coeff(N,x) * my_beta(x+pr_1, N - x + pr_0) / my_beta(pr_1, pr_0);
}


std::vector<float> initialize_beta_binom_probs(int N, float prior_0, float prior_1){
    std::vector<float> probs = std::vector<float>(N+1, 0);
    for(int i=0; i < N+1; i++){
        probs[i] = beta_binom_prob(N, prior_0, prior_1, i);
    }
    return probs;
}


BetaBinomTransitionDist::BetaBinomTransitionDist(float pr_a0, float pr_a1,
                                                 float pr_b0, float pr_b1){
    prior_a0 = pr_a0;
    prior_a1 = pr_a1;
    prior_b0 = pr_b0;
    prior_b1 = pr_b1;
}


void BetaBinomTransitionDist::set_state_action(ContingencyTable ct, 
                                          short unsigned int size_a,
                                          short unsigned int size_b){
    a_probs = initialize_beta_binom_probs(size_a, 
                                          ct.a0 + prior_a0,
                                          ct.a1 + prior_a1);

    b_probs = initialize_beta_binom_probs(size_b, 
                                          ct.b0 + prior_b0,
                                          ct.b1 + prior_b1);
}

/////////////////////////////////
// Factory method
/////////////////////////////////
TransitionDist* TransitionDist::make_transition_dist(std::string tr_dist_type,
                                                     float pr_a0, float pr_a1,
                                                     float pr_b0, float pr_b1){
    if (tr_dist_type == "binom"){
      return new BinomTransitionDist(pr_a0, pr_a1, pr_b0, pr_b1);
    }
    else if(tr_dist_type == "beta_binom"){
      return new BetaBinomTransitionDist(pr_a0, pr_a1, pr_b0, pr_b1);
    }
    else{
      std::cerr << tr_dist_type << " not a valid value for transition distribution." << std::endl;
      throw(1);
    }

}
