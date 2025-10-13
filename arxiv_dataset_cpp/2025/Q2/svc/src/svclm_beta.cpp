// #include <RcppArmadillo.h>
// #include "svc_fxns.h"
// 
// // [[Rcpp::export]]
// Rcpp::List svclm_beta(
//     const arma::vec& Y,               // Response vector
//     const arma::mat& X,               // Design matrix
//     const arma::mat& s,               // Spatial locations (n x 2 matrix)
//     const arma::vec& Y_knots,         // Response vector at knots
//     const arma::mat& X_knots,         // Design matrix at knots
//     const arma::mat& knots,           // Knot locations (m x 2 matrix)
//     const arma::mat& beta_knots_start,       // Initial value for coefficients: beta 1, ..., beta p
//     const arma::vec& phi_beta_start,        // Initial value for phi: beta 1, ..., beta p
//     const arma::vec& sigmasq_beta_start,     // Initial value for sigma^2: beta 1, ..., beta p
//     double tausq_start,               // Initial value for tau^2
//     const arma::vec& phi_beta_proposal_sd, // Proposal standard deviations for phi: beta 1, ..., beta p
//     const arma::vec& phi_beta_lower,      // Lower bounds for phi: beta 1, ..., beta p
//     const arma::vec& phi_beta_upper,      // Upper bounds for phi: beta 1, ..., beta p
//     unsigned int mcmc = 1000                   // Number of MCMC iterations
// ){
//   // Check input dimensions
//   arma::uword n = Y.n_elem; // number of observations
//   arma::uword p = X.n_cols; // number of predictors
//   arma::uword m = knots.n_rows; // number of knots
//   
//   // input errors
//   if (X_knots.n_rows != m || X_knots.n_cols != p) {
//     Rcpp::stop("Design matrix at knots 'X_knots' must be an m x p matrix.");
//   }
//   if (Y_knots.n_elem != m) {
//     Rcpp::stop("Response vector at knots 'Y_knots' must have length m.");
//   }
//   if (s.n_rows != n || s.n_cols != 2) {
//     Rcpp::stop("Spatial locations 's' must be an n x 2 matrix.");
//   }
//   if (knots.n_cols != 2) {
//     Rcpp::stop("Knot locations 'knots' must be an m x 2 matrix.");
//   }
//   if (beta_knots_start.n_cols != p) {
//     Rcpp::stop("Initial beta coefficients 'beta_knots_start' must have length p.");
//   }
//   if (phi_beta_start.n_elem != p) {
//     Rcpp::stop("Initial phi beta vector 'phi_beta_start' must have length p.");
//   }
//   if (sigmasq_beta_start.n_elem != p) {
//     Rcpp::stop("Initial sigmasq beta coefficients 'sigmasq_beta_start' must have length p.");
//   }
//   
//   // Rcpp::Rcout << "Check 10. " << std::endl;
//   
//   // initialize outputs
//   arma::cube beta_samples = arma::zeros(mcmc, n, p);  // posterior beta at observed locations
//   arma::mat phi_beta_samples = arma::zeros(mcmc, p); // mcmc x p matrix of samples for phi_beta's
//   arma::mat phi_beta_acceptance = arma::zeros(mcmc, p); // mcmc x p matrix to track acceptance for phi_beta
//   arma::mat sigmasq_beta_samples = arma::zeros(mcmc, p); // mcmc x p matrix of samples for sigmasq_beta
//   arma::vec tausq_samples = arma::zeros(mcmc); // mcmc vector of samples for tau^2
//   
//   // initialize parameters
//   arma::mat beta_knots_cur = beta_knots_start; // current beta coefficients
//   
//   // Rcpp::Rcout << "beta_knots_cur: " << beta_knots_cur << std::endl;
//   
//   arma::vec sigmasq_beta_cur = sigmasq_beta_start;    // Current sigma^2_beta
//   
//   // Rcpp::Rcout << "sigmasq_beta_cur: " << sigmasq_beta_cur << std::endl;
//   
//   double tausq_cur = tausq_start; // Current tau^2
//   
//   // Rcpp::Rcout << "tausq_cur: " << tausq_cur << std::endl;
//   
//   // Create lower and upper bounds matrices
//   arma::mat phi_beta_bounds = arma::join_horiz(phi_beta_lower, phi_beta_upper);
//   
//   // Rcpp::Rcout << "phi_beta_bounds: " << phi_beta_bounds << std::endl;
//   
//   // Rcpp::Rcout << "Check 20. " << std::endl;
//   
//   // calculate constant parts of various calculations
//   // constant part of bigC
//   arma::mat const_bigC(m, m);
//   for (unsigned int i = 0; i < m; i++) {
//     for (unsigned int j = 0; j < m; j++) {
//       // calculate the distance^2
//       const_bigC(i, j) = (knots(i, 0) - knots(j, 0)) * (knots(i, 0) - knots(j, 0)) +
//         (knots(i, 1) - knots(j, 1)) * (knots(i, 1) - knots(j, 1));
//     }
//   }
//   const_bigC *= -0.5;
//   
//   // Rcpp::Rcout << "const_bigC" << const_bigC << std::endl;
//   
//   // Rcpp::Rcout << "Check 30. " << std::endl;
//   
//   // constant part of lilc
//   arma::mat const_lilc(m, n);
//   for (unsigned int i = 0; i < m; i++) {
//     for (unsigned int j = 0; j < n; j++) {
//       // calculate the distance between each s and each knot
//       const_lilc(i, j) = (s(j, 0) - knots(i, 0)) * (s(j, 0) - knots(i, 0)) +
//         (s(j, 1) - knots(i, 1)) * (s(j, 1) - knots(i, 1));
//     }
//   }
//   const_lilc *= -0.5;
//   
//   // Rcpp::Rcout << "const_lilc: " << const_lilc << std::endl;
//   
//   // constant X_knots_squared
//   arma::mat X_knots_squared = arma::pow(X_knots, 2);
//   
//   // Rcpp::Rcout << "X_knots_squared: " << X_knots_squared << std::endl;
//   
//   // Rcpp::Rcout << "Check 40. " << std::endl;
//   
//   // // Initialize phi objects
//   
//   // Rcpp::Rcout << "Check 50. " << std::endl;
//   
//   std::vector<phi_beta> phi_beta_vec = initialize_phi_beta(
//     p,
//     mcmc,
//     phi_beta_start,
//     phi_beta_proposal_sd,
//     phi_beta_bounds,
//     const_bigC,
//     const_lilc
//   );
//   
//   // Rcpp::Rcout << "C_phi_cur" << phi_beta_vec.at(0).C_phi_cur << std::endl;
//   // Rcpp::Rcout << "R_phi_cur" << phi_beta_vec.at(0).R_phi_cur << std::endl;
//   // Rcpp::Rcout << "C_phi_cur_inv" << phi_beta_vec.at(0).C_phi_cur_inv << std::endl;
//   // Rcpp::Rcout << "C_phi_cur_logdet" << phi_beta_vec.at(0).C_phi_cur_logdet << std::endl;
//   // Rcpp::Rcout << "c_phi_cur" << phi_beta_vec.at(0).c_phi_cur << std::endl;
//   // Rcpp::Rcout << "phi bounds: " << phi_beta_vec.at(0).phi_bounds << std::endl;
//   // Rcpp::Rcout << "proposal_sd: " << phi_beta_vec.at(0).proposal_sd << std::endl;
//   // Rcpp::Rcout << "r: " << phi_beta_vec.at(0).r << std::endl;
//   // Rcpp::Rcout << "iter: " << phi_beta_vec.at(0).iter << std::endl;
//   // Rcpp::Rcout << "acceptance: " << phi_beta_vec.at(0).acceptance << std::endl;
//   // Rcpp::Rcout << "phi_cur: " << phi_beta_vec.at(0).phi_cur << std::endl;
//   // Rcpp::Rcout << "samples: " << phi_beta_vec.at(0).samples << std::endl;
//   // Rcpp::Rcout << "vector size: " << phi_beta_vec.size() << std::endl;
//   
//   // Rcpp::Rcout << "Check 60. " << std::endl;
//   
//   // MCMC loop
//   for (unsigned int i = 0; i < mcmc; i++) {
//     
//     // Rcpp::Rcout << "Iteration: " << i + 1 << std::endl;
//     
//     for (unsigned int j = 0; j < p; j++) {
//       
//       // update beta_knots_j
//       beta_knots_cur.col(j) = update_beta_r_knots(
//         Y_knots,
//         X_knots,
//         X_knots_squared,
//         beta_knots_cur,
//         phi_beta_vec.at(j).C_phi_cur_inv,
//         sigmasq_beta_cur(j),
//         tausq_cur,
//         j
//       );
//       
//       // Rcpp::Rcout << "Check 80. " << std::endl;
//       
//       // predict beta_j at all locations
//       beta_samples.subcube(i, 0, j, i, n - 1, j) = calc_x_tilde(
//         phi_beta_vec.at(j).c_phi_cur,
//         phi_beta_vec.at(j).C_phi_cur_inv,
//         beta_knots_cur.col(j)
//       );
//     }
//   }
//   
//   // Rcpp::Rcout << "Check 110. " << std::endl;
//   
//   Rcpp::List output;
//   output["beta_samples"] = beta_samples; // posterior beta samples at observed locations
//   
//   return output;
// }
