// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppDist)]]

#include <RcppArmadillo.h>
#include <RcppDist.h>

using namespace Rcpp;

double mh_ratio_fun(const arma::vec& y, const arma::vec& l_k, const arma::vec& l_star, const arma::vec& theta_k, const arma::vec& theta_star, const double& p_var){

  // numerator
  double mh_num = sum(y % l_star - exp(l_star)) - (dot(theta_star, theta_star) / 2 / p_var);

  // denominator
  double mh_denom = sum(y % l_k - exp(l_k)) - (dot(theta_k, theta_k) / 2 / p_var);

  // ratio
  double mh_r = mh_num - mh_denom;

  return mh_r;
}

// sample from uniform(0,1), take log
double log_unif() {
  double x = R::runif(0,1);
  double log_x = log(x);
  return (log_x);
}

// divide two integers and return a double
double int_div(const int a, const int b){
  double div_value = static_cast<double>(a) / static_cast<double>(b);
  return div_value;
}

// sample from multivariate normal
arma::vec mvrnormArma(arma::vec mu, arma::mat sigma) {
   int ncols = sigma.n_cols;
   arma::mat Y = arma::randn(1, ncols);
   return vectorise(arma::repmat(mu, 1, 1).t() + Y * arma::chol(sigma));
}

// modulus function (since % == element mult in armadillo)
int mod(int a, int n){
  return a - floor(a/n) * n;
}   

//' Sample parameters from a Poisson Bayesian regression, using MCMC (Metropolis Hastings).
//' 
//' @param y_vector The outcome vector.
//' @param X_matrix The covariate matrix.
//' @param offset A vector of offset parameters.
//' @param theta_0 The initial theta vector.
//' @param n_mcmc The number of MCMC samples to generate.
//' @param thin Specifies how often the MCMC samples should be saved (eg, 10 implies every 10th sample is saved).
//' @param burnin The number of burnin samples.
//' @param n_adapt Specifies how often the proposal covariance should be updated. Note that updates only occur during burnin.
//' @param Sigma_0 The initial covariance matrix for the theta proposal distribution.
//' @param keep_burnin A boolean specifying if burnin samples should be saved as output.
//' @param prior_var The prior variance.
//' @return A list containing a matrix of MCMC samples and the MCMC acceptance rate (post burn-in).
//' @export
//[[Rcpp::export]]
List pois_reg_cpp(
    const arma::vec& y_vector,
    const arma::mat& X_matrix,
    const arma::vec& offset,
    const arma::vec& theta_0,
    int n_mcmc, 
    int thin,
    int burnin,
    int n_adapt,
    const arma::mat& Sigma_0,
    bool keep_burnin,
    double prior_var
  ){

  // constants
  const int n_params = X_matrix.n_cols + 1;

  // matrix to store parameters
  arma::mat theta_mat;
  if (keep_burnin)
  {
    theta_mat = arma::zeros(n_mcmc / thin, n_params);
  }
  else
  {
    theta_mat = arma::zeros((n_mcmc - burnin) / thin, n_params);
  }

  // adaptive proposal parameters
  double s2_0 = 0.5;
  arma::mat S_0 = Sigma_0;
  arma::mat S_mat = s2_0 * S_0;

  double s2_old = 0.5;
  arma::mat S0_old = Sigma_0;
  arma::mat S_old = s2_0 * S_0;

  // check that cholesky decomposition is possible
  arma::mat S_chol;
  bool chol_success = chol(S_chol, S_mat);

  const int c0 = 1;
  const double c1 = 0.8;
  int adapt_time = 1;
  int adapt_acpt = 0;
  arma::mat theta_adapt;
  
  if (n_adapt > 0){
    theta_adapt = arma::zeros(n_adapt, n_params);
  }

  // initialize parameters
  arma::vec theta_k = theta_0;
  arma::vec log_lambda_k = theta_k(0) + offset + X_matrix * theta_k.subvec(1, n_params - 1);
  
  // acceptance rate
  int accept = 0;

  // MH sampler
  for (size_t k = 0; k < n_mcmc; k++)
  {
    // sample new theta parameters
    arma::vec theta_star = mvrnormArma(theta_k, S_mat);
    arma::vec log_lambda_star = theta_star(0) + offset + X_matrix * theta_star.subvec(1, n_params - 1);

    // MH ratio
    double mh_ratio = mh_ratio_fun(y_vector, log_lambda_k, log_lambda_star, theta_k, theta_star, prior_var);

    // S_mat = s2_0 * S_0;

    // accept or reject sample
    if (log_unif() < mh_ratio){
      // update theta and log_lambda
      theta_k = theta_star;
      log_lambda_k = log_lambda_star;

      if (keep_burnin){
        // increase number of accepted theta values by 1
        ++accept;
      } else {
        if (k >= burnin){
          // increase number of accepted theta values by 1
          accept++;
        }
      }
      
      if ((n_adapt > 0) & (k < burnin)){
        adapt_acpt++;
      }
    }

    // save theta
    if (mod(k + 1, thin) == 0){
      Rcpp::checkUserInterrupt();
      if (keep_burnin)
      {
        theta_mat.row((k + 1) / thin - 1) = theta_k.t();
      }
      else
      {
        if (k >= burnin){
          theta_mat.row((k + 1 - burnin) / thin - 1) = theta_k.t();  
        }
      }
    }

    // if adaptive proposals are used
    if ((n_adapt > 0) & (k < burnin)){

      if (mod(k + 1, n_adapt) == 0){
        // save theta
        theta_adapt.row(n_adapt - 1) = theta_k.t();

        // save old S_mat
        S_old = S_mat;
        S0_old = S_0;
        s2_old = s2_0;

        // update adaptive structures
        double gamma_1 = 1 / pow(adapt_time, c1);
        double gamma_2 = c0 * gamma_1;
        double prct_acpt = int_div(adapt_acpt, n_adapt);
        arma::mat S_hat = cov(theta_adapt);
        double log_s2_0 = log(s2_0) + gamma_2 * (prct_acpt - 0.234);
        s2_0 = exp(log_s2_0);
        S_0 = S_0 + gamma_1 * (S_hat - S_0);
        S_mat = s2_0 * S_0; 

        // check that cholesky decomposition is possible
        chol_success = chol(S_chol, S_mat);

        if (!chol_success){
          // revert to previous cholesky decomposition
          s2_0 = s2_old;
          S_0 = S0_old;
          S_mat = S_old;
          adapt_acpt = 0;
        }
        else
        {
          // update time and acceptance
          adapt_acpt = 0;
          adapt_time++;
        }
      }
      else
      {
        // save theta
        theta_adapt.row(mod(k + 1, n_adapt) - 1) = theta_k.t();
      }
    }
  }

  double accept_prop;
  if (keep_burnin){
    accept_prop = int_div(accept, n_mcmc);
  } else {
    accept_prop = int_div(accept, n_mcmc - burnin);
  }

  // return log_lambda_k and theta_mat
  List to_return(2);
  to_return[0] = theta_mat;
  to_return[1] = accept_prop;
  return to_return;
}

