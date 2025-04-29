#include <Rcpp.h>
#include <RcppEigen.h>
#include <cmath>
#include <algorithm>
using namespace Rcpp;

// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
double logSumExp_cpp(NumericVector x) {
  double max_val = max(x);
  return log(sum(exp(x - max_val))) + max_val;
}

// [[Rcpp::export]]
NumericVector state_dist_cpp(double G12, double G21) {
  Eigen::MatrixXd m(2, 2);
  m(0, 0) = 1 - G12;
  m(0, 1) = G12;
  m(1, 0) = G21;
  m(1, 1) = 1 - G21;

  Eigen::MatrixXd b = m.transpose();

  // Eigen decomposition
  Eigen::EigenSolver<Eigen::MatrixXd> es(b);
  Eigen::VectorXd eigenvalues = es.eigenvalues().real();
  Eigen::MatrixXd eigenvectors = es.eigenvectors().real();

  // Index of eigenvalue close to 1
  int index = 0;
  double min_diff = std::abs(eigenvalues(0) - 1.0);
  for (int i = 1; i < eigenvalues.size(); ++i) {
    double diff = std::abs(eigenvalues(i) - 1.0);
    if (diff < min_diff) {
      min_diff = diff;
      index = i;
    }
  }

  // corresponding eigenvector
  Eigen::VectorXd stationary_distribution = eigenvectors.col(index);

  // Normalize stationary distribution
  stationary_distribution = stationary_distribution / stationary_distribution.sum();

  // Convert result to NumericVector
  NumericVector result(stationary_distribution.data(),
                       stationary_distribution.data() + stationary_distribution.size());

  return result;
}


// Computing log likelihood
// [[Rcpp::export]]
double GeneralLoglikelihood_cpp(NumericMatrix y, NumericVector r, NumericVector s, NumericVector u,
                                NumericMatrix Gamma, NumericMatrix e_it, NumericVector B, int model,
                                NumericMatrix z_it, NumericMatrix z_it2) {
  int ndept = y.nrow();
  int nstate = Gamma.ncol();
  int time = y.ncol();
  double gamma_11 = log(Gamma(0, 0));
  double gamma_12 = log(Gamma(0, 1));
  double gamma_21 = log(Gamma(1, 0));
  double gamma_22 = log(Gamma(1, 1));

  NumericMatrix log_forward_probs(ndept, nstate);

  // Model 0
  if(model == 0) {
    NumericMatrix log_likelihood(ndept, time);

    for(int i = 0; i < ndept; ++i) {
      for(int t = 0; t < time; ++t) {
        if(y(i, t) == -1){
          log_likelihood(i, t) = 0;
        }else{
        log_likelihood(i, t) = R::dpois(y(i, t), e_it(i, t) * exp(r[t] + s[t] + u[i]), true);
        }
      }
    }

    double full_log_likelihood = sum(log_likelihood);
    return full_log_likelihood;
  }

  // Model 1, 2, 4, 5 or 7
  if(model == 1 || model == 2 || model == 4 || model == 5 || model == 7) {
    NumericVector init_density = state_dist_cpp(Gamma(0, 1), Gamma(1, 0));
    init_density = log(init_density);
    NumericMatrix Alphas(time, nstate);
    NumericMatrix alpha(time, nstate);
    NumericMatrix beta(time, nstate);
    NumericMatrix log_forward_probs(ndept, nstate);
    NumericVector rowlogsumexp(ndept);

    for(int i = 0; i < ndept; ++i) {
      if(y(i, 0) == -1){
        alpha(0, 0) = init_density[0];
        alpha(0, 1) = 0;
        beta(0, 0) = 0;
        beta(0, 1) = init_density[1];
      }else{
      alpha(0, 0) = init_density[0] + R::dpois(y(i, 0), e_it(i, 0) * exp(r[0] + s[0] + u[i]), true);
      alpha(0, 1) = 0;
      beta(0, 0) = 0;
      beta(0, 1) = init_density[1] + R::dpois(y(i, 0), e_it(i, 0) * exp(r[0] + s[0] + u[i] + B[0] * z_it(i, 0)), true);
      }
      Alphas(0, 0) = alpha(0, 0);
      Alphas(0, 1) = beta(0, 1);

      for(int t = 1; t < time; ++t) {
        if(y(i, t) == -1){
          alpha(t, 0) = Alphas(t-1, 0) + gamma_11;
          alpha(t, 1) = Alphas(t-1, 1) + gamma_21;
          beta(t, 0) = Alphas(t-1, 0) + gamma_12;
          beta(t, 1) = Alphas(t-1, 1) + gamma_22;
          Alphas(t, 0) = logSumExp_cpp(alpha(t, _));
          Alphas(t, 1) = logSumExp_cpp(beta(t, _));
        }else{
        alpha(t, 0) = Alphas(t-1, 0) + gamma_11 + R::dpois(y(i, t), e_it(i, t) * exp(r[t] + s[t] + u[i]), true);
        alpha(t, 1) = Alphas(t-1, 1) + gamma_21 + R::dpois(y(i, t), e_it(i, t) * exp(r[t] + s[t] + u[i]), true);
        beta(t, 0) = Alphas(t-1, 0) + gamma_12 + R::dpois(y(i, t), e_it(i, t) * exp(r[t] + s[t] + u[i] + B[0] * z_it(i, t)), true);
        beta(t, 1) = Alphas(t-1, 1) + gamma_22 + R::dpois(y(i, t), e_it(i, t) * exp(r[t] + s[t] + u[i] + B[0] * z_it(i, t)), true);
        Alphas(t, 0) = logSumExp_cpp(alpha(t, _));
        Alphas(t, 1) = logSumExp_cpp(beta(t, _));
        }
      }

      log_forward_probs(i, 0) = logSumExp_cpp(alpha(time-1, _));
      log_forward_probs(i, 1) = logSumExp_cpp(beta(time-1, _));
    }

    for(int i = 0; i < ndept; ++i) {
      rowlogsumexp[i] = logSumExp_cpp(log_forward_probs(i, _));
    }

    double full_log_likelihood = sum(rowlogsumexp);
    return full_log_likelihood;
  }

  // Model 3 or 6
  if(model == 3 || model == 6) {
    NumericVector init_density = state_dist_cpp(Gamma(0, 1), Gamma(1, 0));
    init_density = log(init_density);
    NumericMatrix Alphas(time, nstate);
    NumericMatrix alpha(time, nstate);
    NumericMatrix beta(time, nstate);
    NumericMatrix log_forward_probs(ndept, nstate);
    NumericVector rowlogsumexp(ndept);

    for(int i = 0; i < ndept; ++i) {
      if(y(i, 0) == -1){
        alpha(0, 0) = init_density[0];
        alpha(0, 1) = 0;
        beta(0, 0) = 0;
        beta(0, 1) = init_density[1];
      }else{
      alpha(0, 0) = init_density[0] + R::dpois(y(i, 0), e_it(i, 0) * exp(r[0] + s[0] + u[i]), true);
      alpha(0, 1) = 0;
      beta(0, 0) = 0;
      beta(0, 1) = init_density[1] + R::dpois(y(i, 0), e_it(i, 0) * exp(r[0] + s[0] + u[i] + B[0] * z_it(i, 0) + B[1] * z_it2(i, 0)), true);
      }
      Alphas(0, 0) = alpha(0, 0);
      Alphas(0, 1) = beta(0, 1);

      for(int t = 1; t < time; ++t) {
        if(y(i, t) == -1){
          alpha(t, 0) = Alphas(t-1, 0) + gamma_11;
          alpha(t, 1) = Alphas(t-1, 1) + gamma_21;
          beta(t, 0) = Alphas(t-1, 0) + gamma_12;
          beta(t, 1) = Alphas(t-1, 1) + gamma_22;
          Alphas(t, 0) = logSumExp_cpp(alpha(t, _));
          Alphas(t, 1) = logSumExp_cpp(beta(t, _));
        }else{
        alpha(t, 0) = Alphas(t-1, 0) + gamma_11 + R::dpois(y(i, t), e_it(i, t) * exp(r[t] + s[t] + u[i]), true);
        alpha(t, 1) = Alphas(t-1, 1) + gamma_21 + R::dpois(y(i, t), e_it(i, t) * exp(r[t] + s[t] + u[i]), true);
        beta(t, 0) = Alphas(t-1, 0) + gamma_12 + R::dpois(y(i, t), e_it(i, t) * exp(r[t] + s[t] + u[i] + B[0] * z_it(i, t) + B[1] * z_it2(i, t)), true);
        beta(t, 1) = Alphas(t-1, 1) + gamma_22 + R::dpois(y(i, t), e_it(i, t) * exp(r[t] + s[t] + u[i] + B[0] * z_it(i, t) + B[1] * z_it2(i, t)), true);
        Alphas(t, 0) = logSumExp_cpp(alpha(t, _));
        Alphas(t, 1) = logSumExp_cpp(beta(t, _));
        }
      }

      log_forward_probs(i, 0) = logSumExp_cpp(alpha(time-1, _));
      log_forward_probs(i, 1) = logSumExp_cpp(beta(time-1, _));
    }

    for(int i = 0; i < ndept; ++i) {
      rowlogsumexp[i] = logSumExp_cpp(log_forward_probs(i, _));
    }

    double full_log_likelihood = sum(rowlogsumexp);
    return full_log_likelihood;
  }

  return R_NegInf;
}


// Computing log likelihood using cyclic seasonal components
// [[Rcpp::export]]
double GeneralLoglikelihood_cpp2(NumericMatrix y, NumericVector r, NumericVector s, NumericVector u,
                                 NumericMatrix Gamma, NumericMatrix e_it, NumericVector B, int model,
                                 NumericMatrix z_it, NumericMatrix z_it2) {
  int ndept = y.nrow();
  int nstate = Gamma.ncol();
  int time = y.ncol();
  double gamma_11 = log(Gamma(0, 0));
  double gamma_12 = log(Gamma(0, 1));
  double gamma_21 = log(Gamma(1, 0));
  double gamma_22 = log(Gamma(1, 1));

  NumericMatrix log_forward_probs(ndept, nstate);

  // Model 0
  if(model == 0) {
    NumericMatrix log_likelihood(ndept, time);

    for(int i = 0; i < ndept; ++i) {
      for(int t = 0; t < time; ++t) {
        if(y(i, t) == -1){
          log_likelihood(i, t) = 0;
        }else{
        int month_index = t % 12;
        log_likelihood(i, t) = R::dpois(y(i, t), e_it(i, t) * exp(r[t] + s[month_index] + u[i]), true);
        }
      }
    }

    double full_log_likelihood = sum(log_likelihood);
    return full_log_likelihood;
  }

  // Model 1, 2, 4, 5 or 7
  if(model == 1 || model == 2 || model == 4 || model == 5 || model == 7) {
    NumericVector init_density = state_dist_cpp(Gamma(0, 1), Gamma(1, 0));
    init_density = log(init_density);
    NumericMatrix Alphas(time, nstate);
    NumericMatrix alpha(time, nstate);
    NumericMatrix beta(time, nstate);
    NumericMatrix log_forward_probs(ndept, nstate);
    NumericVector rowlogsumexp(ndept);

    for(int i = 0; i < ndept; ++i) {
      if(y(i, 0) == -1){
        alpha(0, 0) = init_density[0];
        alpha(0, 1) = 0;
        beta(0, 0) = 0;
        beta(0, 1) = init_density[1];
      }else{
      alpha(0, 0) = init_density[0] + R::dpois(y(i, 0), e_it(i, 0) * exp(r[0] + s[0] + u[i]), true);
      alpha(0, 1) = 0;
      beta(0, 0) = 0;
      beta(0, 1) = init_density[1] + R::dpois(y(i, 0), e_it(i, 0) * exp(r[0] + s[0] + u[i] + B[0] * z_it(i, 0)), true);
      }
      Alphas(0, 0) = alpha(0, 0);
      Alphas(0, 1) = beta(0, 1);

      for(int t = 1; t < time; ++t) {
        if(y(i, t) == -1){
          alpha(t, 0) = Alphas(t-1, 0) + gamma_11;
          alpha(t, 1) = Alphas(t-1, 1) + gamma_21;
          beta(t, 0) = Alphas(t-1, 0) + gamma_12;
          beta(t, 1) = Alphas(t-1, 1) + gamma_22;
          Alphas(t, 0) = logSumExp_cpp(alpha(t, _));
          Alphas(t, 1) = logSumExp_cpp(beta(t, _));
        }else{
        int month_index = t % 12;
        alpha(t, 0) = Alphas(t-1, 0) + gamma_11 + R::dpois(y(i, t), e_it(i, t) * exp(r[t] + s[month_index] + u[i]), true);
        alpha(t, 1) = Alphas(t-1, 1) + gamma_21 + R::dpois(y(i, t), e_it(i, t) * exp(r[t] + s[month_index] + u[i]), true);
        beta(t, 0) = Alphas(t-1, 0) + gamma_12 + R::dpois(y(i, t), e_it(i, t) * exp(r[t] + s[month_index] + u[i] + B[0] * z_it(i, t)), true);
        beta(t, 1) = Alphas(t-1, 1) + gamma_22 + R::dpois(y(i, t), e_it(i, t) * exp(r[t] + s[month_index] + u[i] + B[0] * z_it(i, t)), true);
        Alphas(t, 0) = logSumExp_cpp(alpha(t, _));
        Alphas(t, 1) = logSumExp_cpp(beta(t, _));
        }
      }

      log_forward_probs(i, 0) = logSumExp_cpp(alpha(time-1, _));
      log_forward_probs(i, 1) = logSumExp_cpp(beta(time-1, _));
    }

    for(int i = 0; i < ndept; ++i) {
      rowlogsumexp[i] = logSumExp_cpp(log_forward_probs(i, _));
    }

    double full_log_likelihood = sum(rowlogsumexp);
    return full_log_likelihood;
  }

  // Model 3 or 6
  if(model == 3 || model == 6) {
    NumericVector init_density = state_dist_cpp(Gamma(0, 1), Gamma(1, 0));
    init_density = log(init_density);
    NumericMatrix Alphas(time, nstate);
    NumericMatrix alpha(time, nstate);
    NumericMatrix beta(time, nstate);
    NumericMatrix log_forward_probs(ndept, nstate);
    NumericVector rowlogsumexp(ndept);

    for(int i = 0; i < ndept; ++i) {
      if(y(i, 0) == -1){
        alpha(0, 0) = init_density[0];
        alpha(0, 1) = 0;
        beta(0, 0) = 0;
        beta(0, 1) = init_density[1];
      }else{
      alpha(0, 0) = init_density[0] + R::dpois(y(i, 0), e_it(i, 0) * exp(r[0] + s[0] + u[i]), true);
      alpha(0, 1) = 0;
      beta(0, 0) = 0;
      beta(0, 1) = init_density[1] + R::dpois(y(i, 0), e_it(i, 0) * exp(r[0] + s[0] + u[i] + B[0] * z_it(i, 0) + B[1] * z_it2(i, 0)), true);
      }
      Alphas(0, 0) = alpha(0, 0);
      Alphas(0, 1) = beta(0, 1);

      for(int t = 1; t < time; ++t) {
        if(y(i, t) == -1){
          alpha(t, 0) = Alphas(t-1, 0) + gamma_11;
          alpha(t, 1) = Alphas(t-1, 1) + gamma_21;
          beta(t, 0) = Alphas(t-1, 0) + gamma_12;
          beta(t, 1) = Alphas(t-1, 1) + gamma_22;
          Alphas(t, 0) = logSumExp_cpp(alpha(t, _));
          Alphas(t, 1) = logSumExp_cpp(beta(t, _));
        }else{
        int month_index = t % 12;
        alpha(t, 0) = Alphas(t-1, 0) + gamma_11 + R::dpois(y(i, t), e_it(i, t) * exp(r[t] + s[month_index] + u[i]), true);
        alpha(t, 1) = Alphas(t-1, 1) + gamma_21 + R::dpois(y(i, t), e_it(i, t) * exp(r[t] + s[month_index] + u[i]), true);
        beta(t, 0) = Alphas(t-1, 0) + gamma_12 + R::dpois(y(i, t), e_it(i, t) * exp(r[t] + s[month_index] + u[i] + B[0] * z_it(i, t) + B[1] * z_it2(i, t)), true);
        beta(t, 1) = Alphas(t-1, 1) + gamma_22 + R::dpois(y(i, t), e_it(i, t) * exp(r[t] + s[month_index] + u[i] + B[0] * z_it(i, t) + B[1] * z_it2(i, t)), true);
        Alphas(t, 0) = logSumExp_cpp(alpha(t, _));
        Alphas(t, 1) = logSumExp_cpp(beta(t, _));
        }
      }

      log_forward_probs(i, 0) = logSumExp_cpp(alpha(time-1, _));
      log_forward_probs(i, 1) = logSumExp_cpp(beta(time-1, _));
    }

    for(int i = 0; i < ndept; ++i) {
      rowlogsumexp[i] = logSumExp_cpp(log_forward_probs(i, _));
    }

    double full_log_likelihood = sum(rowlogsumexp);
    return full_log_likelihood;
  }

  return R_NegInf;
}
