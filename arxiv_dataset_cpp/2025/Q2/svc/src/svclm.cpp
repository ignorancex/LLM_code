#include <RcppArmadillo.h>
#include "svc_fxns.h"

// [[Rcpp::export]]
Rcpp::List svclm_cpp(
    const arma::vec& Y,               // Response vector (n x 1 vector)
    const arma::mat& X,               // Design matrix (n x p matrix)
    const arma::mat& coords,               // Spatial locations (n x 2 matrix)
    const arma::vec& Y_knots,         // Response vector at knots (m x 1 vector)
    const arma::mat& X_knots,         // Design matrix at knots (m x p matrix)
    const arma::mat& knots,           // Knot locations (m x 2 matrix)
    const arma::mat& w_knots_start,       // Initial values for coefficients (m x p matrix)
    const arma::vec& sigmasq_start,     // Initial values for sigma^2 (p x 1 vector)
    const arma::vec& a_sigmasq,            // Hyperparameter a for inverse gamma prior on sigma^2 (p x 1 vector)
    const arma::vec& b_sigmasq,            // Hyperparameter b for inverse gamma prior on sigma^2 (p x 1 vector)
    double tausq_start,               // Initial value for tau^2
    double a_tausq,               // Hyperparameter a for inverse gamma prior on tau^2
    double b_tausq,               // Hyperparameter b for inverse gamma prior on tau^2
    const arma::vec& phi_start,        // Initial values for phi (p x 1 vector)
    const arma::vec& phi_lower,      // Lower bounds for uniform prior on phi (p x 1 vector)
    const arma::vec& phi_upper,      // Upper bounds for uniform prior on phi (p x 1 vector)
    const arma::vec& phi_proposal_sd_start, // Starting proposal standard deviations for phi (p x 1 vector)
    double phi_target_acceptance, // Target acceptance rate for phi updates
    unsigned int mcmc                   // Number of MCMC iterations
){
  // Check input dimensions
  arma::uword n = Y.n_elem; // number of observations
  arma::uword p = X.n_cols; // number of predictors
  arma::uword m = knots.n_rows; // number of knots

  // input errors
  if (X_knots.n_rows != m || X_knots.n_cols != p) {
    Rcpp::stop("Design matrix at knots 'X_knots' must be an m x p matrix.");
  }
  if (Y_knots.n_elem != m) {
    Rcpp::stop("Response vector at knots 'Y_knots' must have length m.");
  }
  if (coords.n_rows != n || coords.n_cols != 2) {
    Rcpp::stop("Spatial locations 's' must be an n x 2 matrix.");
  }
  if (knots.n_cols != 2) {
    Rcpp::stop("Knot locations 'knots' must be an m x 2 matrix.");
  }
  if (w_knots_start.n_cols != p) {
    Rcpp::stop("Initial w coefficients 'w_knots_start' must have length p.");
  }
  if (phi_start.n_elem != p) {
    Rcpp::stop("Initial phi vector 'phi_start' must have length p.");
  }
  if (sigmasq_start.n_elem != p) {
    Rcpp::stop("Initial sigma^2 vector 'sigmasq_start' must have length p.");
  }
  if (X_knots.has_nan()) {
    Rcpp::stop("'X_knots' must not contain NaN values.");
  }
  if (Y_knots.has_nan()) {
    Rcpp::stop("'Y_knots' must not contain NaN values.");
  }

  // Rcpp::Rcout << "Check 10. " << std::endl;

  // initialize outputs
  arma::cube w_samples = arma::zeros(mcmc, n, p);  // posterior w at observed locations
  arma::mat phi_samples = arma::zeros(mcmc, p); // mcmc x p matrix of samples for phi's
  arma::mat phi_acceptance = arma::zeros(mcmc, p); // mcmc x p matrix to track acceptance for phi
  arma::mat sigmasq_samples = arma::zeros(mcmc, p); // mcmc x p matrix of samples for sigmasq
  arma::vec tausq_samples = arma::zeros(mcmc); // mcmc vector of samples for tau^2

  // initialize parameters
  arma::mat w_knots_cur = w_knots_start; // current w coefficients
  arma::vec sigmasq_cur = sigmasq_start;    // Current sigma^2
  double tausq_cur = tausq_start; // Current tau^2

  // Create lower and upper bounds matrices
  arma::mat phi_bounds = arma::join_horiz(phi_lower, phi_upper);

  // calculate constant parts of various calculations
  // constant part of bigC
  arma::mat const_bigC(m, m);
  for (unsigned int i = 0; i < m; i++) {
    for (unsigned int j = 0; j < m; j++) {
      // calculate the distance^2
      const_bigC(i, j) = (knots(i, 0) - knots(j, 0)) * (knots(i, 0) - knots(j, 0)) +
        (knots(i, 1) - knots(j, 1)) * (knots(i, 1) - knots(j, 1));
    }
  }
  const_bigC *= -0.5;

  // constant part of lilc
  arma::mat const_lilc(m, n);
  for (unsigned int i = 0; i < m; i++) {
    for (unsigned int j = 0; j < n; j++) {
      // calculate the distance between each s and each knot
      const_lilc(i, j) = (coords(j, 0) - knots(i, 0)) * (coords(j, 0) - knots(i, 0)) +
        (coords(j, 1) - knots(i, 1)) * (coords(j, 1) - knots(i, 1));
    }
  }
  const_lilc *= -0.5;

  // constant X_knots_squared
  arma::mat X_knots_squared = arma::pow(X_knots, 2);

  // // Initialize phi objects
  std::vector<phi> phi_vec = initialize_phi(
    p,
    mcmc,
    phi_start,
    phi_proposal_sd_start,
    phi_bounds,
    const_bigC,
    const_lilc,
    phi_target_acceptance
  );

  // MCMC loop
  for (unsigned int i = 0; i < mcmc; i++) {

    Rcpp::Rcout << "Iteration: " << i + 1 << std::endl;

    for (unsigned int j = 0; j < p; j++) {
      // Update phi_j via Random Walk Metropolis
      phi_vec.at(j).RWupdate(w_knots_cur, sigmasq_cur);

      // Update sigma^2
      sigmasq_cur(j) = update_sigma2_r(w_knots_cur.col(j), a_sigmasq(j), a_sigmasq(j), phi_vec.at(j).C_phi_cur_inv);

      // update w_knots_j
      w_knots_cur.col(j) = update_w_r_knots(
        Y_knots,
        X_knots,
        X_knots_squared,
        w_knots_cur,
        phi_vec.at(j).C_phi_cur_inv,
        sigmasq_cur(j),
        tausq_cur,
        j,
        m
      );

      // predict w_j at all locations
      w_samples.subcube(i, 0, j, i, n - 1, j) = calc_x_tilde(
        phi_vec.at(j).c_phi_cur,
        phi_vec.at(j).C_phi_cur_inv,
        w_knots_cur.col(j)
      );
      
    }

    sigmasq_samples.row(i) = sigmasq_cur.t(); // store the current sigma^2 samples

    // update tau^2
    // Currently using only knots
    // Should use all locations if we can get w predictions
    tausq_cur = update_tau2_r(Y_knots, X_knots, w_knots_cur, a_tausq, b_tausq);
    tausq_samples(i) = tausq_cur;
  }

  for (unsigned int j = 0; j < p; j++) {
    // Store the final phi samples and acceptance rates
    phi_samples.col(j) = phi_vec.at(j).samples;
    phi_acceptance.col(j) = phi_vec.at(j).acceptance;
  }

  Rcpp::List output;
  output["w_samples"] = w_samples; // posterior w samples at observed locations
  output["phi_samples"] = phi_samples;
  output["phi_acceptance"] = phi_acceptance;
  output["sigmasq_samples"] = sigmasq_samples; // posterior sigma^2 samples
  output["tausq_samples"] = tausq_samples; // posterior tau^2 samples

  return output;
}
