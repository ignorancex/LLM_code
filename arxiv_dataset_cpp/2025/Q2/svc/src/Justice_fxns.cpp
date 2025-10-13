#include <RcppArmadillo.h>
#include "svc_fxns.h"

// Remove these duplicate implementations since they're in svc_fxns.cpp:
/*
 double logit(double x, double l, double u) {
 if (x <= l || x >= u) Rcpp::stop("x must be between bounds (l < x < u)");
 return log((x - l)/(u - x));
 }
 
 double logistic(double x, double l, double u) {
 return l + (u - l)/(1.0 + exp(-x));
 }
 
 double normal_proposal_logitscale(const double& x, double l, double u) {
 if (x <= l || x >= u) Rcpp::stop("x must be between bounds for Jacobian");
 return -log(u - x) - log(x - l);
 }
 */

// Keeping only the unique functions that aren't defined elsewhere:
double log_transform(double x) {
  if (x <= 0) Rcpp::stop("x must be positive for log transform");
  return log(x);
}

double exp_transform(double x) {
  return exp(x);
}

arma::vec transform_theta(const arma::vec& theta, 
                          const arma::vec& phi_bounds_lower,
                          const arma::vec& phi_bounds_upper, 
                          unsigned int p) {
  if (theta.n_elem != 2*p + 1) Rcpp::stop("theta has incorrect dimension");
  
  arma::vec theta_trans(2*p + 1);
  
  // Transform variance parameters
  for(unsigned int i = 0; i <= p; i++) {
    theta_trans(i) = log_transform(theta(i));
  }
  
  // Transform phi parameters using the logit from svc_fxns.cpp
  for(unsigned int i = 1; i <= p; i++) {
    theta_trans(p + i) = logit(theta(p + i), 
                phi_bounds_lower(i-1), 
                phi_bounds_upper(i-1));
  }
  
  return theta_trans;
}

arma::vec inverse_transform_theta(const arma::vec& theta_trans,
                                  const arma::vec& phi_bounds_lower,
                                  const arma::vec& phi_bounds_upper,
                                  unsigned int p) {
  if (theta_trans.n_elem != 2*p + 1) Rcpp::stop("theta_trans has incorrect dimension");
  
  arma::vec theta(2*p + 1);
  
  // Transform back variance parameters
  for(unsigned int i = 0; i <= p; i++) {
    theta(i) = exp_transform(theta_trans(i));
  }
  
  // Transform back phi parameters using the logistic from svc_fxns.cpp
  for(unsigned int i = 1; i <= p; i++) {
    theta(p + i) = logistic(theta_trans(p + i),
          phi_bounds_lower(i-1),
          phi_bounds_upper(i-1));
  }
  
  return theta;
}

double calc_jacobian_theta(const arma::vec& theta,
                           const arma::vec& phi_bounds_lower,
                           const arma::vec& phi_bounds_upper,
                           unsigned int p) {
  double jac = 0.0;
  
  // Jacobian for variance parameters
  for(unsigned int i = 0; i <= p; i++) {
    jac += log(theta(i));
  }
  
  // Jacobian for phi parameters using normal_proposal_logitscale from svc_fxns.cpp
  for(unsigned int i = 1; i <= p; i++) {
    const double& phi = theta(p + i);
    double l = phi_bounds_lower(i-1);
    double u = phi_bounds_upper(i-1);
    jac += normal_proposal_logitscale(phi, l, u);
  }
  
  return jac;
}

arma::vec update_variances(const arma::vec& variances_current,
                           const arma::vec& proposal_sd) {
  arma::vec variances_prop(variances_current.n_elem);
  for(unsigned int i = 0; i < variances_current.n_elem; i++) {
    double current_trans = log_transform(variances_current(i));
    double prop_trans = current_trans + proposal_sd(i) * arma::randn();
    variances_prop(i) = exp_transform(prop_trans);
  }
  return variances_prop;
}

arma::vec update_phis(const arma::vec& phis_current,
                      const arma::vec& proposal_sd,
                      const arma::vec& phi_bounds_lower,
                      const arma::vec& phi_bounds_upper) {
  arma::vec phis_prop(phis_current.n_elem);
  for(unsigned int i = 0; i < phis_current.n_elem; i++) {
    double current_trans = logit(phis_current(i), 
                                 phi_bounds_lower(i), 
                                 phi_bounds_upper(i));
    double prop_trans = current_trans + proposal_sd(i) * arma::randn();
    phis_prop(i) = logistic(prop_trans, 
              phi_bounds_lower(i),
              phi_bounds_upper(i));
  }
  return phis_prop;
}