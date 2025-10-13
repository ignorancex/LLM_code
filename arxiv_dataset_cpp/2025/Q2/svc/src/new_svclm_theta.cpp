// #include <RcppArmadillo.h>
// #include "svc_fxns.h"
// 
// // [[Rcpp::export]]
// Rcpp::List new_svclm_theta(
//     const arma::vec& Y,               // Response vector
//     const arma::mat& X,               // Design matrix
//     const arma::mat& coords,               // Spatial locations (n x 2 matrix)
//     const arma::mat& beta_start,       // Initial value for coefficients: beta 1, ..., beta p
//     double tausq_start,               // Initial value for tau^2
//     const arma::vec& sigmasq_start,     // Initial value for sigma^2: beta 1, ..., beta p
//     const arma::vec& phi_start,        // Initial value for phi: beta 1, ..., beta p
//     double tausq_proposal_sd,        // Proposal standard deviation for tau^2
//     const arma::vec& sigmasq_proposal_sd, // Proposal standard deviations for sigma^2: beta 1, ..., beta p
//     const arma::vec& phi_proposal_sd, // Proposal standard deviations for phi: beta 1, ..., beta p
//     const arma::vec& phi_lower,      // Lower bounds for phi: beta 1, ..., beta p
//     const arma::vec& phi_upper,      // Upper bounds for phi: beta 1, ..., beta p
//     double a_tausq,               // Hyperparameter a for tau^2
//     double b_tausq,               // Hyperparameter b for tau^2
//     const arma::vec& a_sigmasq,            // Hyperparameter a for sigma^2_beta
//     const arma::vec& b_sigmasq,            // Hyperparameter b for sigma^2_beta
//     unsigned int mcmc = 1000                   // Number of MCMC iterations
// ){
//   // Check input dimensions
//   arma::uword n = Y.n_elem; // number of observations
//   arma::uword p = X.n_cols; // number of predictors
//   arma::uword np = n * p;
//   
//   // input errors
//   if (coords.n_rows != n || coords.n_cols != 2) {
//     Rcpp::stop("Spatial locations 'coords' must be an n x 2 matrix.");
//   }
//   if (X.n_rows != n) {
//     Rcpp::stop("Design matrix 'X' must have the same number of rows as the response vector 'Y'.");
//   }
//   if (phi_start.n_elem != p) {
//     Rcpp::stop("Initial phi beta vector 'phi_beta_start' must have length p.");
//   }
//   if (sigmasq_start.n_elem != p) {
//     Rcpp::stop("Initial sigmasq beta coefficients 'sigmasq_beta_start' must have length p.");
//   }
//   if (beta_start.n_rows != n || beta_start.n_cols != p) {
//     Rcpp::stop("Initial beta coefficients 'beta_start' must be an n x p matrix.");
//   }
//   if (phi_lower.n_elem != p || phi_upper.n_elem != p) {
//     Rcpp::stop("Lower and upper bounds for phi must have length p.");
//   }
//   
//   // Rcpp::Rcout << "Check 10. " << std::endl;
//   
//   // initialize outputs
//   arma::mat phi_samples = arma::zeros(mcmc, p); // mcmc x p matrix of samples for phi_beta's
//   arma::mat sigmasq_samples = arma::zeros(mcmc, p); // mcmc x p matrix of samples for sigmasq_beta
//   arma::vec tausq_samples = arma::zeros(mcmc); // mcmc vector of samples for tau^2
//   arma::vec acceptance = arma::zeros(mcmc); // mcmc vector to track random walk acceptance
//   
//   // initialize parameters
//   arma::vec tausig_cur = arma::vec(p + 1);
//   tausig_cur(0) = tausq_start; // tau^2
//   tausig_cur.subvec(1, p) = sigmasq_start; // sigma^2: beta 1, ..., beta p
//   arma::vec phi_cur = phi_start; // Current phi values
//   
//   // Create lower and upper bounds matrices
//   arma::mat phi_bounds = arma::join_horiz(phi_lower, phi_upper);
//   
//   // calculate constant parts of various calculations
//   // construct Z
//   arma::mat Z(n, np);
//   
//   // construct y:X
//   arma::mat YX = join_horiz(Y, X);
//   
//   // constant part of K
//   arma::mat const_K = arma::zeros(np, np);
//   for (unsigned int i = 0; i < n; i++) {
//     for (unsigned int j = 0; j < n; j++) {
//       // calculate the distance^2
//       const_K.submat(p * i, p * j, p * i + p - 1, p * j + p - 1).diag() = 
//         arma::ones(p) *
//         (coords(i, 0) - coords(j, 0)) * (coords(i, 0) - coords(j, 0)) +
//         (coords(i, 1) - coords(j, 1)) * (coords(i, 1) - coords(j, 1));
//     }
//   }
//   const_K *= -0.5; // scale by -0.5 for the Gaussian kernel
//   
//   // calculate cur log density
//   // calculate K_cur
//   arma::mat K_cur = calc_K(tausig_cur.subvec(1, p), phi_cur, const_K, n, p);
//   
//   // calculate cur log p(theta)
//   double logprior_cur; // log prior for current parameters
//   
//   // calculate cur log p(Y | theta)
//   double logdens_cur = calc_logdens(K_cur, tausig_cur(0), Z, YX, n, p);
//   
//   // calculate cur log posterior density
//   double logpost_cur = logprior_cur + logdens_cur;
//   
//   // MCMC loop
//   for (unsigned int iter = 0; iter < mcmc; iter++) {
//     
//     // Rcpp::Rcout << "Iteration: " << iter + 1 << std::endl;
//     
//     // propose alt tausq, sigmasq, phi in a 1-step random walk Metropolis
//     arma::vec tausig_alt; // copy current values
//     arma::vec phi_alt;
//       
//     // calculate K_alt
//     arma::mat K_alt = calc_K(tausig_alt.subvec(1, p), phi_alt, const_K, n, p);
//     
//     // calculate alt log p(theta)
//     double logprior_alt; // log prior for current parameters
//     
//     // calculate alt log p(Y | theta)
//     double logdens_alt = calc_logdens(K_alt, tausig_alt(0), Z, YX, n, p);
//     
//     // calculate alt log posterior density
//     double logpost_alt = logprior_alt + logdens_alt;
//       
//     // calculate Jacobian
//     double jacobian;
//     
//     // Calculate the log acceptance probability
//     double logaccept = logpost_alt - logpost_cur + jacobian;
//     
//     bool accepted = do_I_accept(logaccept);
//     
//     if(accepted){
//       tausig_cur = tausig_alt;
//       phi_cur = phi_alt; // accept the new phi values
//       K_cur = K_alt; // accept the new covariance matrix
//       logdens_cur = logdens_alt; // update the current log density
//       acceptance(iter) = 1; // update acceptance
//     }
//     
//     // Store the current samples
//     tausq_samples(iter) = tausig_cur(0); // store the current tau^2 sample
//     sigmasq_samples.row(iter) = tausig_cur.subvec(1, p).t(); // store the current sigma^2 samples
//     phi_samples.row(iter) = phi_cur.t(); // store the current phi samples
//   }
//   
//   Rcpp::List output;
//   output["tausq_samples"] = tausq_samples; // posterior tau^2 samples
//   output["sigmasq_samples"] = sigmasq_samples; // posterior sigma^2_beta samples
//   output["phi_samples"] = phi_samples;
//   output["acceptance"] = acceptance;
//   
//   return output;
// }
