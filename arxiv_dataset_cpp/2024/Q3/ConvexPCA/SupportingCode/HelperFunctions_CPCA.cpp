#include <RcppArmadillo.h>
#include <omp.h>  // Include OpenMP library
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;

// [[Rcpp::export]]
arma::vec undo_spherical_coords(const arma::vec& theta){
  // Undo spherical coordinates
  int len = theta.n_elem;
  arma::vec P(len + 1, arma::fill::zeros);  // create a vector of size len+1, and fill it with zeros
  if (len == 1){
    P[0] = cos(theta[0]);
    P[1] = sin(theta[0]);
  } else {
    arma::vec stheta = sin(theta);
    arma::vec sthetacp = cumprod(stheta);
    arma::vec ctheta = cos(theta);
    arma::vec fsin(len + 1, arma::fill::zeros);
    arma::vec fcos(len + 1, arma::fill::zeros);
    
    fsin[0] = 1;
    fcos[len] = 1;
    
    for (int i = 0; i < len + 1; ++i){
      if (i < len) {
        fcos[i] = ctheta[i];
        fsin[i + 1] = sthetacp[i];
      }
      P[i] = fcos[i] * fsin[i];
    }
  }
  return P;
}


// [[Rcpp::export]]
arma::vec spherical_coords(const arma::vec& P) {
  // Get spherical coordinate representation of a vector
  int d = P.n_elem;
  arma::vec theta(d - 1);
  
  if (d == 2) {
    theta[0] = acos(P[0]);
  } else {
    arma::vec P2 = square(P);
    arma::vec P2r = flipud(P2);
    arma::vec CS = cumsum(P2r);
    arma::vec CSr = flipud(CS);
    arma::vec CSr_ = CSr.subvec(0, d - 3);
    arma::vec P_ = P.subvec(0, d - 3);
    
    for (int i = 0; i < d - 2; i++) {
      theta[i] = acos(P_[i] / sqrt(CSr_[i]));
    }
    
    if (P[d - 1] < 0) {
      theta[d - 2] = 2 * arma::datum::pi - acos(P[d - 2] / sqrt(CSr[d - 2]));
    } else {
      theta[d - 2] = acos(P[d - 2] / sqrt(CSr[d - 2]));
    }
  }
  
  return theta;
}


// [[Rcpp::export]]
arma::mat project_orthogonal_complement(const arma::mat& X, const arma::mat& B) {
  // Calculate the projection of X onto B
  arma::mat projection_B = B * B.t() * X;
  
  // Subtract the projection from X to get the projection onto the orthogonal complement
  arma::mat projection_B_orthogonal = X - projection_B;
  
  return projection_B_orthogonal;
}

// [[Rcpp::export]]
arma::vec ls_fit(const arma::mat& B, const arma::vec& p_) {
  arma::vec p = solve(B, p_);
  return p;
}

// [[Rcpp::export]]
arma::vec matrix_vector_multiply(const arma::mat& M, const arma::vec& v) {
  // Matrix vector multiplication (for use in R)
  return M * v;
}

// [[Rcpp::export]]
arma::vec i_points_CPCA(const arma::vec& x0, const arma::vec& p, const arma::mat& A, const arma::vec& b) {
  // Function to obtain intersection points
  arma::vec num = b - matrix_vector_multiply(A, x0);
  arma::vec den = matrix_vector_multiply(A, p);
  arma::vec a = num / den;
  
  // Initialize tvals with Inf and -Inf respectively
  arma::vec tvals(2);
  tvals[0] = arma::datum::inf;
  tvals[1] = -arma::datum::inf;
  
  // Find the minimum positive value in 'a'
  arma::vec positive_a = a.elem(arma::find(a > 0));
  if (!positive_a.is_empty()) {
    tvals[0] = positive_a.min();
  }
  
  // Find the maximum negative value in 'a'
  arma::vec negative_a = a.elem(arma::find(a < 0));
  if (!negative_a.is_empty()) {
    tvals[1] = negative_a.max();
  }
  
  return tvals;
}

// [[Rcpp::export]]
double restricted_projection_coord(const arma::vec& x, const arma::vec& x0, const arma::vec& p, const arma::vec& tvals) {
  // Function to obtain restricted projection
  double num = dot(x - x0, p);  // Scalar projection of x-x0 onto p
  double den = dot(p, p);       // Norm squared of p
  double a = num / den;         // Normalize the scalar projection
  double t_sol;
  double min_tval = tvals.min();
  double max_tval = tvals.max();
  
  if(a >= min_tval && a <= max_tval) {
    t_sol = a;
  } else {
    double t1 = tvals[0];
    double t2 = tvals[1];
    if(std::abs(a - t1) < std::abs(a - t2)) {
      t_sol = t1;
    } else {
      t_sol = t2;
    }
  }
  return t_sol;
}

// [[Rcpp::export]]
double eval_CPCA_obj(const arma::vec& p, const arma::vec& x0, const arma::mat& A, const arma::vec& b, const arma::mat& X) {
  // Evaluate CPCA objective
  arma::vec tvals = i_points_CPCA(x0, p, A, b);
  int nobs = X.n_cols;
  double val = 0.0;
  
  arma::vec sol, diff;
  
  for(int i = 0; i < nobs; ++i){
    double t = restricted_projection_coord(X.col(i), x0, p, tvals);
    sol = x0 + t * p;
    diff = X.col(i) - sol;
    val += arma::dot(diff, diff);
  }
  
  return val / nobs;
}

// [[Rcpp::export]]
arma::vec eval_grad_CPCA_obj(const arma::vec& p, double h, const arma::vec& x0, const arma::mat& A, const arma::vec& b, const arma::mat& X) {
  // Evaluate CPCA gradient
  int d = p.n_elem;
  arma::vec grad(d, arma::fill::zeros);
  
  for(int i = 0; i < d; ++i){
    arma::vec p_p = p;
    arma::vec p_m = p;
    p_p[i] += h;
    p_m[i] -= h;
    grad[i] = (eval_CPCA_obj(p_p, x0, A, b, X) - eval_CPCA_obj(p_m, x0, A, b, X)) / (2.0 * h);
  }
  
  return grad;
}

// [[Rcpp::export]]
arma::vec eval_grad_CPCA_obj_parallel(const arma::vec& p, double h, const arma::vec& x0, const arma::mat& A, const arma::vec& b, const arma::mat& X, int num_threads) {
  // Evaluate CPCA gradient with parallelization
  
  // Set the number of threads
  omp_set_num_threads(num_threads);
  
  int d = p.n_elem;
  arma::vec grad(d, arma::fill::zeros);
  
  #pragma omp parallel for
  for(int i = 0; i < d; ++i){
    arma::vec p_p = p;
    arma::vec p_m = p;
    p_p[i] += h;
    p_m[i] -= h;
    grad[i] = (eval_CPCA_obj(p_p, x0, A, b, X) - eval_CPCA_obj(p_m, x0, A, b, X)) / (2.0 * h);
  }
  
  return grad;
}

// [[Rcpp::export]]
double eval_CPCA_obj_sph(const arma::vec& theta, const arma::mat& B, const arma::vec& x0, const arma::mat& A, const arma::vec& b, const arma::mat& X) {
  // Evaluate CPCA objective in spherical coordinates
  arma::vec omega = undo_spherical_coords(theta);
  arma::vec p = B * omega;
  arma::vec tvals = i_points_CPCA(x0, p, A, b);
  int nobs = X.n_cols;
  double val = 0.0;
  
  arma::vec sol, diff;
  
  for(int i = 0; i < nobs; ++i){
    double t = restricted_projection_coord(X.col(i), x0, p, tvals);
    sol = x0 + t * p;
    diff = X.col(i) - sol;
    val += arma::dot(diff, diff);
  }
  
  return val / nobs;
}

// [[Rcpp::export]]
arma::vec eval_grad_CPCA_obj_sph(const arma::vec& theta, const arma::mat& B, double h, const arma::vec& x0, const arma::mat& A, const arma::vec& b, const arma::mat& X) {
  // Evaluate CPCA gradient in spherical coordinates
  int d = theta.n_elem;
  arma::vec grad(d, arma::fill::zeros);
  
  for(int i = 0; i < d; ++i){
    arma::vec theta_p = theta;
    arma::vec theta_m = theta;
    theta_p[i] += h;
    theta_m[i] -= h;
    grad[i] = (eval_CPCA_obj_sph(theta_p, B, x0, A, b, X) - eval_CPCA_obj_sph(theta_m, B, x0, A, b, X)) / (2.0 * h);
  }
  
  return grad;
}

// [[Rcpp::export]]
arma::vec eval_grad_GPCA_obj_sph_parallel(const arma::vec& theta, const arma::mat& B, double h, const arma::vec& x0, const arma::mat& A, const arma::vec& b, const arma::mat& X, int num_threads) {
  // Evaluate CPCA gradient with parallelization in spherical coordinates
  
  // Set the number of threads
  omp_set_num_threads(num_threads);
  
  int d = theta.n_elem;
  arma::vec grad(d, arma::fill::zeros);
  
  #pragma omp parallel for
  for(int i = 0; i < d; ++i){
    arma::vec theta_p = theta;
    arma::vec theta_m = theta;
    theta_p[i] += h;
    theta_m[i] -= h;
    grad[i] = (eval_CPCA_obj_sph(theta_p, B, x0, A, b, X) - eval_CPCA_obj_sph(theta_m, B, x0, A, b, X)) / (2.0 * h);
  }
  
  return grad;
}

// [[Rcpp::export]]
arma::mat null_space_via_svd(const arma::mat& A) {
  // Get nullspace via SVD
  arma::mat U, V;
  arma::vec s;
  svd(U, s, V, A);
  
  arma::uword m = A.n_rows;
  
  // Select columns of V that are greater than the number of rows of A
  if (V.n_cols > m) {
    return V.cols(m, V.n_cols - 1);
  } else {
    return arma::mat();  // Return an empty matrix if there are no such columns
  }
  return V;
}