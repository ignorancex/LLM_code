#include <RcppArmadillo.h>
#include "svc_fxns.h"

arma::mat calc_bigC(
    const arma::mat& const_bigC,
    double phi
) {
  return exp(const_bigC / phi);
}

arma::mat calc_lilc(
    const arma::mat& const_lilc,
    double phi
) {
  return exp(const_lilc / phi);
}

arma::mat stable_Chol(
    const arma::mat& A
){
  arma::mat symA = 0.5 * (A + A.t()); // Ensure symmetry
  symA += arma::eye(A.n_rows, A.n_cols) * 1e-10; // Add a small value to the diagonal for numerical stability
  arma::mat R = arma::chol(symA);
  return R; // Return the Cholesky factor
}

// get inverse of A given A
arma::mat inv_Chol(
    const arma::mat& A
) {
  arma::mat Rinv = arma::inv(arma::trimatu(stable_Chol(A)));
  arma::mat Ainv = Rinv * Rinv.t();
  return Ainv;
}

// get inverse of A given R = Chol(A)
arma::mat inv_Chol_R(
    const arma::mat& R
) {
  arma::mat Rinv = arma::inv(arma::trimatu(R));
  arma::mat Ainv = Rinv * Rinv.t();
  return Ainv;
}

// get logdet of A by giving R from stable_Chol(A)
double logdet_R(
    const arma::mat& R
) {
  double logdet = 2 * arma::accu(arma::log(arma::diagvec(R)));
  return logdet;
}

arma::mat calc_K(
    const arma::vec& sigmasq,
    const arma::vec& phi,
    const arma::mat& const_K,
    arma::uword n,
    arma::uword p
) {
  arma::mat K = const_K;
  for (unsigned int i = 0; i < n; i++) {
    arma::uword pi = p * i;
    arma::uword piplus = pi + p - 1;
    for (unsigned int j = 0; j < n; j++) {
      arma::uword pj = p * j;
      arma::uword pjplus = pj + p - 1;
      // calculate the distance^2
      K.submat(pi, pj, piplus, pjplus).diag() = 
        exp(K.submat(pi, pj, piplus, pjplus).diag() / phi);
    }
  }
  return K;
}

double calc_logdens(
    const arma::mat& K,
    double tausq,
    const arma::mat& Z,
    const arma::mat& YX,
    arma::uword n,
    arma::uword p
) {
  arma::mat Sigma = Z * K * Z.t() + arma::eye(n, n) * tausq; // n x n
  arma::mat L = stable_Chol(Sigma); // n x n, upper triangular matrix
  arma::mat vU = arma::solve(arma::trimatu(L), YX); // n x p+1
  arma::vec v = vU.col(0); // n x 1, first column of vU
  arma::mat U = vU.submat(arma::span::all, arma::span(1, p)); // n x p
  arma::mat W = stable_Chol(U.t() * U); // p x p
  arma::vec b = U.t() * v; // p x 1
  arma::vec btilde = arma::solve(arma::trimatu(W), b); // p x 1
  double logdens = -1 * arma::accu(arma::log(arma::diagvec(W))) -
    arma::accu(arma::log(arma::diagvec(L))) -
    0.5 * arma::as_scalar((v.t() * v - btilde.t() * btilde));
  return logdens;
}

double logit(double x, double l, double u){
  return -log( (u-l)/(x-l) -1.0 );
}

double logistic(double x, double l, double u){
  return l + (u-l)/(1.0+exp(-x));
}

double par_huvtransf_fwd(double par, const arma::mat& set_unif_bounds){
  par = logit(par, set_unif_bounds(0, 0), set_unif_bounds(0, 1));
  return par;
}

double par_huvtransf_back(double par, const arma::mat& set_unif_bounds){
  par = logistic(par, set_unif_bounds(0, 0), set_unif_bounds(0, 1));
  return par;
}

double normal_proposal_logitscale(const double& x, double l, double u){
  //return x - 2 * log(1 + exp(x));
  return -log(u-x) - log(x-l);
}

double calc_jacobian(double new_param, double param, const arma::mat& set_unif_bounds){
  // logit normal proposal
  double jac = normal_proposal_logitscale(param, set_unif_bounds(0, 0), set_unif_bounds(0, 1)) -
    normal_proposal_logitscale(new_param, set_unif_bounds(0, 0), set_unif_bounds(0, 1));
  
  return jac;
}

bool do_I_accept(double logaccept){
  double u = arma::randu();
  bool answer = exp(logaccept) > u;
  return answer;
}

arma::vec calc_x_tilde(
    const arma::mat& lilc_cur,
    const arma::mat& bigC_inv_cur,
    const arma::vec& x_knots
) {
  
  return lilc_cur.t() * bigC_inv_cur * x_knots;
}







// I think everything below here is no longer needed for new method
// but we need to keep for now until we get rid of the original svclm.
// If there is something you end up needing, move it up above this
// so we don't delete it later

arma::mat get_R(
    const arma::mat& A
){
  arma::mat symA = 0.5 * (A + A.t()); // Ensure symmetry
  symA += arma::eye(A.n_rows, A.n_cols) * 1e-10; // Add a small value to the diagonal for numerical stability
  arma::mat R = arma::chol(symA);
  return R; // Return the Cholesky factor
}

double GP_log_density(
    const arma::vec& x,
    double sigmasq,
    const arma::mat& C_phi_inv,
    double C_phi_logdet
) {
  arma::mat Sigma_inv = C_phi_inv / sigmasq;
  double Sigma_logdet = C_phi_inv.n_cols * log(sigmasq) + C_phi_logdet;
  
  double log_likelihood =  -0.5 * Sigma_logdet -
    0.5 * arma::as_scalar(x.t() * Sigma_inv * x);
  
  return log_likelihood;
}

phi::phi(
  unsigned int mcmc,
  unsigned int r_in,
  const arma::vec& phi_start,
  const arma::vec& proposal_sd_in,
  const arma::mat& phi_bounds_in,
  const arma::mat& const_bigC_in,
  const arma::mat& const_lilc_in,
  double target_accept
)
  : const_bigC(const_bigC_in),
    const_lilc(const_lilc_in)
{
  samples = arma::zeros(mcmc);
  acceptance = arma::zeros(mcmc);
  phi_cur = phi_start(r_in);
  C_phi_cur = calc_bigC(const_bigC_in, phi_cur);
  R_phi_cur = get_R(C_phi_cur);
  C_phi_cur_inv = inv_Chol_R(R_phi_cur);
  C_phi_cur_logdet = logdet_R(R_phi_cur);
  c_phi_cur = calc_lilc(const_lilc_in, phi_cur);
  paramsd = proposal_sd_in(r_in);
  phi_bounds = phi_bounds_in.row(r_in);
  accept_count = 0;
  iter = 0;
  alpha_star = target_accept;
  gamma = 0.5 + 1e-16;
  g0 = 100;
  S = paramsd * paramsd;
  prodparam = paramsd / (g0 + 1.0);
  started = false;
  r = r_in;
}

std::vector<phi> initialize_phi(
    unsigned int p,
    unsigned int mcmc,
    const arma::vec& phi_start,
    const arma::vec& phi_proposal_sd,
    const arma::mat& phi_bounds,
    const arma::mat& const_bigC,
    const arma::mat& const_lilc,
    double target_accept
){
  std::vector<phi> phi_vec;
  
  for (unsigned int k = 0; k < p; k++) {
    phi_vec.emplace_back(
      mcmc,
      k,
      phi_start,
      phi_proposal_sd,
      phi_bounds,
      const_bigC,
      const_lilc,
      target_accept
    );
  }
  return phi_vec;
}

void phi::RWupdate(
    const arma::mat& x_knots_mat,
    const arma::vec& sigmasq_cur_vec
) {
  arma::vec x_knots = x_knots_mat.col(r);
  double sigmasq_cur = sigmasq_cur_vec(r);
  
  double u_update = arma::randn();
  
  double phi_alt = par_huvtransf_back(par_huvtransf_fwd(
    phi_cur, phi_bounds) + paramsd * u_update, phi_bounds);

  arma::mat C_phi_alt = calc_bigC(const_bigC, phi_alt);
  arma::mat R_phi_alt = get_R(C_phi_alt);
  arma::mat C_phi_alt_inv = inv_Chol_R(R_phi_alt);
  double C_phi_alt_logdet = logdet_R(R_phi_alt);

  // Calculate the log density of (w | sigma^2, phi)
  double logdens_cur = GP_log_density(x_knots, sigmasq_cur, C_phi_cur_inv, C_phi_cur_logdet);
  double logdens_alt = GP_log_density(x_knots, sigmasq_cur, C_phi_alt_inv, C_phi_alt_logdet);

  // Calculate the Jacobian of the proposal from transformation
  double jacobian  = calc_jacobian(phi_alt, phi_cur, phi_bounds);

  // Calculate the log acceptance probability
  double logaccept = logdens_alt - logdens_cur + jacobian;
  
  bool accepted = do_I_accept(logaccept);

  if(accepted){
    phi_cur = phi_alt;
    C_phi_cur = C_phi_alt;
    R_phi_cur = R_phi_alt;
    C_phi_cur_inv = C_phi_alt_inv;
    C_phi_cur_logdet = C_phi_alt_logdet;
    c_phi_cur = calc_lilc(const_lilc, phi_cur);
    acceptance(iter) = 1;
    accept_count++;
    accept_ratio = static_cast<double>(accept_count) / (iter + 1);
  }
  
  phi::adapt(u_update, exp(logaccept));
  
  samples(iter) = phi_cur;
  iter++;
}

void phi::adapt(
    double u,
    double alpha
){
  if(!started & (iter < 2 * g0)) {
    started = true;
  }
  if(started){
    i = iter - g0;
    eta = std::min(1.0, pow(i + 1, -gamma));
    alpha = std::min(1.0, alpha);
    Sigma = 1 + eta * (alpha - alpha_star);
    S = paramsd * paramsd * Sigma;
    paramsd = sqrt(S);
  }
}

arma::vec update_w_r_knots(
    const arma::vec& Y_knots,
    const arma::mat& X_knots,
    const arma::mat& X_knots_squared,
    const arma::mat& w_knots,
    const arma::mat& C_phi_cur_inv,
    double sigma_r_square,
    double tau_square,
    int j,
    arma::uword m
) {
  // Calculate the covariance matrix K based on the locations and phi_r
  arma::mat Sigma0inv = C_phi_cur_inv / sigma_r_square; // Inverse of the prior covariance matrix
  arma::mat Sigmainv = arma::diagmat(X_knots_squared.col(j)) / tau_square;
  arma::mat Sigma1_Chol = arma::inv(arma::trimatu(stable_Chol(Sigma0inv + Sigmainv)));
  // arma::mat Sigma1 = inv_Chol(Sigma0inv + Sigmainv);
  
  arma::mat X_new = X_knots;
  X_new.shed_col(j); // Removes column j in-place
  
  arma::mat w_new = w_knots;
  w_new.shed_col(j); // Removes column j in-place
  
  // Compute sum without column j
  arma::vec sumXw = arma::sum(X_new % w_new, 1);
  arma::vec Y_tilde = Y_knots - sumXw;
  arma::vec result = Y_tilde / X_knots.col(j);
  arma::vec mu1 = Sigma1_Chol * Sigma1_Chol.t() * Sigmainv * result;
  
  arma::vec w_r_knots = mu1 + Sigma1_Chol * arma::randn(m);
  
  return w_r_knots;
}

double update_sigma2_r(
    const arma::vec& w_r,
    double a_r,
    double b_r,
    const arma::mat& C_inv
) {
  // Calculate (w_r' C^{-1} w_r)
  double w_r_sum_square = arma::as_scalar(w_r.t() * C_inv * w_r);
  
  // Calculate the shape and scale parameters for the inverse gamma distribution
  double a_r_post = a_r + w_r.n_elem / 2.0;
  double b_r_post = b_r + w_r_sum_square / 2.0;
  
  //Sample sigma_r^2 from the inverse gamma distribution
  return 1.0 / arma::randg(arma::distr_param(a_r_post, 1.0 / b_r_post));
}

double update_tau2_r(const arma::vec& Y, const arma::mat& X, const arma::mat& w, double a_t, double b_t) {
  arma::vec residual = Y - arma::sum(X % w, 1);
  double residual_sum_square = arma::dot(residual, residual);
  double a_t_post = a_t + Y.n_elem / 2.0;
  double b_t_post = b_t + residual_sum_square / 2.0;
  
  // Sample tau^2 from the inverse gamma distribution
  return 1.0 / arma::randg(arma::distr_param(a_t_post, 1.0 / b_t_post));
}
