// [[Rcpp::depends("RcppArmadillo")]]
#include <RcppArmadillo.h>
#include <Rcpp.h>


using namespace Rcpp;
using namespace arma;


  
static double const log2pi = std::log(2.0 * M_PI);


/* C++ version of the dtrmv BLAS function */
void inplace_tri_mat_mult(arma::rowvec &x, arma::mat const &trimat){
  arma::uword const n = trimat.n_cols;
  
  for(unsigned j = n; j-- > 0;){
    double tmp(0.);
    for(unsigned i = 0; i <= j; ++i)
      tmp += trimat.at(i, j) * x[i];
    x[j] = tmp;
  }
}



// [[Rcpp::export]]
arma::vec dmvnrm_arma_fast(arma::mat const &x,  
                           arma::rowvec const &mean,  
                           arma::mat const &sigma, 
                           bool const logd = false) { 
  using arma::uword;
  uword const n = x.n_rows, 
    xdim = x.n_cols;
  arma::vec out(n);
  arma::mat const rooti = arma::inv(trimatu(arma::chol(sigma)));
  double const rootisum = arma::sum(log(rooti.diag())), 
    constants = -(double)xdim/2.0 * log2pi, 
    other_terms = rootisum + constants;
  
  arma::rowvec z;
  for (uword i = 0; i < n; i++) {
    z = (x.row(i) - mean);
    inplace_tri_mat_mult(z, rooti);
    out(i) = other_terms - 0.5 * arma::dot(z, z);     
  }  
  
  if (logd)
    return out;
  return exp(out);
}




/*
 the same as cs_unfold() in rTensor
 
 a: n1 x n2 x n3 tensor 
 k: {1, 2, 3}
 output:
 k=1: n2*n3 x n1 matrix
 k=2: n1*n3 x n2 matrix
 k=3: n1*n2 x n3 matrix
*/
Mat<double> k_unfold_cpp(Cube<double>& a, const uword k){
  const uword n1 = a.n_rows;
  const uword n2 = a.n_cols;
  const uword n3 = a.n_slices;
  
  Mat<double> res;
  Mat<double> sub;
  
  switch (k){
  case 1:{
    uword nm = n2 * n3;
    res.zeros(nm, n1);
    
    for (uword i = 0; i < n1; i++){
      sub = a.row(i);
      res.col(i) = vectorise(sub, 0);
    }
    // res = reshape(res, n1, nm);
  }
    break;
  case 2:{
    uword nm = n1 * n3;
    res.zeros(nm, n2);
    
    for (uword i = 0; i < n2; i++){
      sub = a.col(i);
      res.col(i) = vectorise(sub, 0);
    }
    // res = reshape(res, n2, nm);
  }
    break;
  case 3:{
    uword nm = n1 * n2;
    res.zeros(nm, n3);
    
    for (uword i = 0; i < n3; i++){
      sub = a.slice(i);
      res.col(i) = vectorise(sub, 0);
    }
    // res = reshape(res, n3, nm);
  }
    break;
  }
  return res;
}



double get_D_K_t_cpp(cube P_left,
                      cube P_right,
                      mat Z,
                      mat Sigma){
  int p, i, k,
  n1 = P_left.n_rows,
  n2 = P_left.n_cols,
  L = P_left.n_slices,
  M_t = Z.n_rows,
  n1_n2 = n1 + n2 - 1;
  
  double S_tilde_p;
  
  
  rowvec cur_P_sp_left(L), cur_P_sp_right(L);
  vec  D_K_t_left(M_t),
      D_K_t_right(M_t),
      D_K_t_diff(M_t);
  
  for (p = 0; p < n1_n2; p++){
    cur_P_sp_left.zeros();
    cur_P_sp_right.zeros();
    
    if (p < n2){
      S_tilde_p = 0.0 + n2 + 1.0 - p;
      for (i = 0; i < std::min(n2 - p, n1); i++){
        
        for (k = 0; k < L; k++){
          cur_P_sp_left[k] += P_left.at(i, i + p, k) / S_tilde_p;
          cur_P_sp_right[k] += P_right.at(i, i + p, k) / S_tilde_p;
        }
        
      }
    }
    else{
      S_tilde_p = 0.0 + n2 + n1 - p;
      for (i = 0; i < std::min(n1 + n2 - p - 1, n2); i++){
        
        for (k = 0; k < L; k++){
          cur_P_sp_left[k] += P_left.at(i + p - n2 + 1, i, k) / S_tilde_p;
          cur_P_sp_right[k] += P_right.at(i + p - n2 + 1, i, k) / S_tilde_p;
        }
        
      }
    }
    
    vec const kernel_left = dmvnrm_arma_fast(Z, cur_P_sp_left, Sigma, false);
    vec const kernel_right = dmvnrm_arma_fast(Z, cur_P_sp_right, Sigma, false);
    
    double factor = S_tilde_p / (1.0 * n1 * n2);
    
    for (k = 0; k < M_t; k++){
      D_K_t_left[k] += kernel_left[k] * factor;
      D_K_t_right[k] += kernel_right[k] * factor;
    }
    
  }
  
  for (k = 0; k < M_t; k++){
    D_K_t_diff[k] = std::abs(D_K_t_left[k] - D_K_t_right[k]);
  }
  
  return max(D_K_t_diff);
}





// [[Rcpp::export]]
double get_D_K_t_ud_cpp(cube P_left,
                      cube P_right,
                      mat Z,
                      mat Sigma){

  /*
  in the undirected case, X = Y so n1 = n2
  */

  int p, i, j, k, q, shift,
    n1 = P_left.n_rows,
    // n2 = P_left.n_cols,
    L = P_left.n_slices,
    M_t = Z.n_rows,
    n = n1;
    // n1_n2 = n1 + n2 - 1;
  
  double S_tilde_p, 
    S_tilde_sum = 0.0;
  
  
  rowvec cur_P_sp_left(L), cur_P_sp_right(L);
  vec  D_K_t_left(M_t),
      D_K_t_right(M_t),
      D_K_t_diff(M_t);
  
  for (p = 1; p < n; p++){
    cur_P_sp_left.zeros();
    cur_P_sp_right.zeros();
    

    q = floor((n - p) / (2.0 * p));

    S_tilde_p = 0.0 + q * p + std::min(p, n - p - 2 * p * q);
    S_tilde_sum += S_tilde_p;

    for (i = 0; i < q; i++){

      shift = i * 2 * p;

      for (j = 0; j < p; j++){

        for (k = 0; k < L; k++){
          cur_P_sp_left[k] += P_left.at(shift + j, shift + j + p, k) / S_tilde_p;
          cur_P_sp_right[k] += P_right.at(shift + j, shift + j + p, k) / S_tilde_p;
        }

      }
      
    }

    shift = q * 2 * p;

    for (j = 0; j + shift + p < n; j++){

      for (k = 0; k < L; k++){
        cur_P_sp_left[k] += P_left.at(shift + j, shift + j + p, k) / S_tilde_p;
        cur_P_sp_right[k] += P_right.at(shift + j, shift + j + p, k) / S_tilde_p;
      }

    }
    
    
    vec const kernel_left = dmvnrm_arma_fast(Z, cur_P_sp_left, Sigma, false);
    vec const kernel_right = dmvnrm_arma_fast(Z, cur_P_sp_right, Sigma, false);
    
    double factor = S_tilde_p;
    
    for (k = 0; k < M_t; k++){
      D_K_t_left[k] += kernel_left[k] * factor;
      D_K_t_right[k] += kernel_right[k] * factor;
    }
    
  }
  
  for (k = 0; k < M_t; k++){
    D_K_t_diff[k] = std::abs(D_K_t_left[k] - D_K_t_right[k]) / S_tilde_sum;
  }
  
  return max(D_K_t_diff);
}






cube fold_cpp(mat& x, const uword n1, const uword n2, const uword n3){
  cube res = cube(n1, n2, n3);
  for(uword k = 0; k < n1; k++){
    for(uword i = 0; i < n2; i++){
      for(uword j = 0; j < n3; j++){
        res(k, i, j) = x(k, i + j * n2);
      }
    }
  }
  return res;
}


// [[Rcpp::export]]
cube uase_cpp(cube& a, uword r){
  
  mat Y = k_unfold_cpp(a, 1);
  
  mat U;
  vec s;
  mat V;
  
  svd(U, s, V, Y);
  
  r = std::min(r, s.size());
  
  // mat U_t = mat(Y.n_rows, r);
  // mat V_t = mat(Y.n_cols, r);
  // mat S_t = mat(r, r);
  
  mat U_t = U.cols(0, r - 1);
  mat V_t = V.cols(0, r - 1);
  mat S_t = diagmat(s.subvec(0, r - 1));
  
  mat P_uase =  V_t * S_t * U_t.t();
  
  P_uase.clamp(0.0, 1.0);
  
  cube Pt_uase = cube(a.n_rows, a.n_cols, a.n_slices);
  Pt_uase = fold_cpp(P_uase, a.n_rows, a.n_cols, a.n_slices);
  
  return Pt_uase;
  
}



// [[Rcpp::export]]
double frobenius_cube_cpp(cube X){
  const uword L = X.n_slices;
  double f = 0.0;
  uword i;

  for (i = 0; i < L; i ++){
    f += pow(norm(X.slice(i), "fro"), 2);
  }

  return pow(f, 0.5);
}







// [[Rcpp::export]]
vec max_D_s_t_rescale_uase_cpp(field<cube> A_list, double h_kernel, uword r_hat = 1, const bool directed = true,
                               const uword buffer_left = 0, const uword buffer_right = 0, const bool verbose = false, const double alpha = 0.01){
  cube A = A_list(0);
  
  const uword n1 = A.n_rows;
  const uword n2 = A.n_cols;
  const uword L = A.n_slices;
  const uword TT = A_list.n_elem;
  
  const uword n_min = std::min(n1, n2);
  const uword n_max = std::max(n1, n2);
  
  const double C_M = 20.0;
  const int M_up = 1000;
  double M_td;
  int M_t;
  
  double n_left;
  double n_right;
  
  cube A_sum_left;
  cube A_sum_right;
  cube A_bar_left;
  cube A_bar_right;
  cube A_sum_left_0;
  cube A_sum_right_0;
  
  cube P_left;
  cube P_right;
  
  mat Z;
  
  mat Sigma = diagmat(vec(L, fill::value(1.0 / h_kernel)));
  
  vec D_K_t_max_rescale = vec(TT, fill::zeros);
  double rescale;
  
  A_sum_left_0.zeros(n1, n2, L);
  for (uword i = 0; i < buffer_left; i++){
    A_sum_left_0 += A_list(i);
  }
  A_sum_right_0.zeros(n1, n2, L);
  A_sum_right_0 += A_list(buffer_left);
  
  for (uword t = 1; t < TT; t++){
    vec D_K_t = vec(t, fill::zeros);
    
    if (t > buffer_left){
      A_sum_right_0 += A_list(t);
    }
    
    if (t - buffer_right < buffer_left){
      continue;
    }
    else{
      A_sum_left = cube(A_sum_left_0);
      A_sum_right = cube(A_sum_right_0);
      
      for (uword s = buffer_left; s < t - buffer_right - 1; s++){
        
        n_left = 0.0 + s + 1.0;
        n_right = 0.0 + t - s;
        
        A_sum_left += A_list(s);
        A_sum_right -= A_list(s);
        
        A_bar_left = A_sum_left / n_left;
        A_bar_right = A_sum_right / n_right;
        
        P_left = uase_cpp(A_bar_left, r_hat);
        P_right = uase_cpp(A_bar_right, r_hat);
        
        M_td = C_M * pow((static_cast<double>(t) * n_min) / ( 2.0 * L * L * log(1.0 / alpha * std::max(n_max, t))), 0.5 * L);
        M_t = std::min(static_cast<int>(M_td), M_up);
        
        // each column is a sample
        Z = mat(M_t, L, fill::randu);
        
        rescale = (1.0/pow(n_left, 0.5) + 1.0/pow(n_right, 0.5)) * pow(log(1.0 / alpha * std::max(n_max, t + 1)), 0.5);
        
        if (directed){
          D_K_t(s) = get_D_K_t_cpp(P_left, P_right, Z, Sigma) / rescale;
        }
        else{
          D_K_t(s) = get_D_K_t_ud_cpp(P_left, P_right, Z, Sigma) / rescale;
        }

        // double D_K_t_raw = get_D_K_t_cpp(P_left, P_right, Z, Sigma, 1);
        // D_K_t(s) = D_K_t_raw / rescale;
      }
    }
    
    D_K_t_max_rescale[t] = max(D_K_t);
    if (verbose){
      Rprintf("%i \n", t + 1);
    }
  }
  return D_K_t_max_rescale;
}





// [[Rcpp::export]]
List online_cpd_uase_cpp(field<cube> A_list, double tau_factor, double h_kernel, uword r_hat = 1, const bool directed = true, 
                         const uword buffer_left = 0, const uword buffer_right = 0, const bool verbose = false, const double alpha = 0.01){
  cube A = A_list(0);
  
  const uword n1 = A.n_rows;
  const uword n2 = A.n_cols;
  const uword L = A.n_slices;
  const uword TT = A_list.n_elem;
  
  const uword n_min = std::min(n1, n2);
  const uword n_max = std::max(n1, n2);
  
  const double C_M = 20.0;
  const int M_up = 1000;
  double M_td;
  int M_t;
  
  double n_left;
  double n_right;
  
  cube A_sum_left;
  cube A_sum_right;
  cube A_bar_left;
  cube A_bar_right;
  cube A_sum_left_0;
  cube A_sum_right_0;
  
  cube P_left;
  cube P_right;
  
  mat Z;
  
  mat Sigma = diagmat(vec(L, fill::value(1.0 / h_kernel)));
  
  vec D_K_t_max_rescale = vec(TT, fill::zeros);
  double rescale;
  
  A_sum_left_0.zeros(n1, n2, L);
  for (uword i = 0; i < buffer_left; i++){
    A_sum_left_0 += A_list(i);
  }
  A_sum_right_0.zeros(n1, n2, L);
  A_sum_right_0 += A_list(buffer_left);
  
  for (uword t = 1; t < TT; t++){
    vec D_K_t = vec(t, fill::zeros);
    
    if (t > buffer_left){
      A_sum_right_0 += A_list(t);
    }
    
    if (t - buffer_right < buffer_left){
      continue;
    }
    else{
      A_sum_left = cube(A_sum_left_0);
      A_sum_right = cube(A_sum_right_0);
      
      for (uword s = buffer_left; s < t - buffer_right - 1; s++){
        
        n_left = 0.0 + s + 1.0;
        n_right = 0.0 + t - s;
        
        A_sum_left += A_list(s);
        A_sum_right -= A_list(s);
        
        A_bar_left = A_sum_left / n_left;
        A_bar_right = A_sum_right / n_right;
        
        P_left = uase_cpp(A_bar_left, r_hat);
        P_right = uase_cpp(A_bar_right, r_hat);
        
        M_td = C_M * pow((static_cast<double>(t) * n_min) / ( 2.0 * L * L * log(1.0 / alpha * std::max(n_max, t))), 0.5 * L);
        M_t = std::min(static_cast<int>(M_td), M_up);
        
        // each column is a sample
        Z = mat(M_t, L, fill::randu);
        
        rescale = (1.0/pow(n_left, 0.5) + 1.0/pow(n_right, 0.5)) * pow(log(1.0 / alpha * std::max(n_max, t + 1)), 0.5);
        
        if (directed){
          D_K_t(s) = get_D_K_t_cpp(P_left, P_right, Z, Sigma) / rescale;
        }
        else{
          D_K_t(s) = get_D_K_t_ud_cpp(P_left, P_right, Z, Sigma) / rescale;
        }
        
      }
    }
    
    D_K_t_max_rescale[t] = max(D_K_t);
    
    if (verbose){
      Rprintf("%i \n", t + 1);
    }
    
    if (D_K_t_max_rescale[t] > tau_factor){
      return List::create( 
        _["t"] = t + 1,
        _["D_K_t_max_rescale"] = D_K_t_max_rescale,
        _["find"] = true
      );
    }
    
  }
  return List::create( 
    _["t"] = TT + 1, 
    _["D_K_t_max_rescale"] = D_K_t_max_rescale,
    _["find"] = false
  );
}




/*
Fixed latent positions
*/



// [[Rcpp::export]]
vec max_D_s_t_rescale_fixed_uase_cpp(field<cube> A_list, uword r_hat = 1,
                               const uword buffer_left = 0, const uword buffer_right = 0, const bool verbose = false, const double alpha = 0.01){
  cube A = A_list(0);
  
  const uword n1 = A.n_rows;
  const uword n2 = A.n_cols;
  const uword L = A.n_slices;
  const uword TT = A_list.n_elem;
  
  const uword t0 = 10;
  double n_left;
  double n_right;
  
  cube A_sum_left;
  cube A_sum_right;
  cube A_bar_left;
  cube A_bar_right;
  cube A_sum_left_0;
  cube A_sum_right_0;
  
  cube P_left;
  cube P_right;
  
  vec D_K_t_max_rescale = vec(TT, fill::zeros);
  double rescale;
  
  A_sum_left_0.zeros(n1, n2, L);
  for (uword i = 0; i < buffer_left; i++){
    A_sum_left_0 += A_list(i);
  }
  A_sum_right_0.zeros(n1, n2, L);
  A_sum_right_0 += A_list(buffer_left);
  
  for (uword t = 1; t < TT; t++){
    vec D_K_t = vec(t, fill::zeros);
    
    if (t > buffer_left){
      A_sum_right_0 += A_list(t);
    }
    
    if (t - buffer_right < buffer_left){
      continue;
    }
    else{
      A_sum_left = cube(A_sum_left_0);
      A_sum_right = cube(A_sum_right_0);
      
      for (uword s = buffer_left; s < t - buffer_right - 1; s++){
        
        n_left = 0.0 + s + 1.0;
        n_right = 0.0 + t - s;
        
        A_sum_left += A_list(s);
        A_sum_right -= A_list(s);
        
        A_bar_left = A_sum_left / n_left;
        A_bar_right = A_sum_right / n_right;
        
        P_left = uase_cpp(A_bar_left, r_hat);
        P_right = uase_cpp(A_bar_right, r_hat);
        
        rescale = (1.0/pow(n_left, 0.5) + 1.0/pow(n_right, 0.5)) * pow(log(1.0 / alpha * std::max(t0, t + 1)), 0.5);
        
        D_K_t(s) = frobenius_cube_cpp(P_left - P_right) / rescale;
      }
    }
    
    D_K_t_max_rescale[t] = max(D_K_t);
    if (verbose){
      Rprintf("%i \n", t + 1);
    }
  }
  return D_K_t_max_rescale;
}





// [[Rcpp::export]]
List online_cpd_fixed_uase_cpp(field<cube> A_list, double tau_factor, uword r_hat = 1,
                         const uword buffer_left = 0, const uword buffer_right = 0, const bool verbose = false, const double alpha = 0.01){
  cube A = A_list(0);
  
  const uword n1 = A.n_rows;
  const uword n2 = A.n_cols;
  const uword L = A.n_slices;
  const uword TT = A_list.n_elem;
  
  const uword t0 = 10;
  double n_left;
  double n_right;
  
  cube A_sum_left;
  cube A_sum_right;
  cube A_bar_left;
  cube A_bar_right;
  cube A_sum_left_0;
  cube A_sum_right_0;
  
  cube P_left;
  cube P_right;
  
  vec D_K_t_max_rescale = vec(TT, fill::zeros);
  double rescale;
  
  A_sum_left_0.zeros(n1, n2, L);
  for (uword i = 0; i < buffer_left; i++){
    A_sum_left_0 += A_list(i);
  }
  A_sum_right_0.zeros(n1, n2, L);
  A_sum_right_0 += A_list(buffer_left);
  
  for (uword t = 1; t < TT; t++){
    vec D_K_t = vec(t, fill::zeros);
    
    if (t > buffer_left){
      A_sum_right_0 += A_list(t);
    }
    
    if (t - buffer_right < buffer_left){
      continue;
    }
    else{
      A_sum_left = cube(A_sum_left_0);
      A_sum_right = cube(A_sum_right_0);
      
      for (uword s = buffer_left; s < t - buffer_right - 1; s++){
        
        n_left = 0.0 + s + 1.0;
        n_right = 0.0 + t - s;
        
        A_sum_left += A_list(s);
        A_sum_right -= A_list(s);
        
        A_bar_left = A_sum_left / n_left;
        A_bar_right = A_sum_right / n_right;
        
        P_left = uase_cpp(A_bar_left, r_hat);
        P_right = uase_cpp(A_bar_right, r_hat);
        
        rescale = (1.0/pow(n_left, 0.5) + 1.0/pow(n_right, 0.5)) * pow(log(1.0 / alpha * std::max(t0, t + 1)), 0.5);
        
        D_K_t(s) = frobenius_cube_cpp(P_left - P_right) / rescale;
      }
    }
    
    D_K_t_max_rescale[t] = max(D_K_t);
    
    if (verbose){
      Rprintf("%i \n", t + 1);
    }
    
    if (D_K_t_max_rescale[t] > tau_factor){
      return List::create( 
        _["t"] = t + 1,
        _["D_K_t_max_rescale"] = D_K_t_max_rescale,
        _["find"] = true
      );
    }
    
  }
  return List::create( 
    _["t"] = TT + 1, 
    _["D_K_t_max_rescale"] = D_K_t_max_rescale,
    _["find"] = false
  );
}





// [[Rcpp::export]]
double l2_cube_cpp(cube X){
  const uword L = X.n_slices;
  double f = 0.0;
  uword i;

  for (i = 0; i < L; i ++){
    f += pow(norm(X.slice(i), 2), 2);
  }

  return pow(f, 0.5);
}


// [[Rcpp::export]]
List online_cpd_fixed_uase2_cpp(field<cube> A_list, double tau_factor, uword r_hat,
                         const uword buffer_left = 0, const uword buffer_right = 0, const bool verbose = false, const double alpha = 0.01){
  cube A = A_list(0);
  
  const uword n1 = A.n_rows;
  const uword n2 = A.n_cols;
  const uword L = A.n_slices;
  const uword TT = A_list.n_elem;
  
  const uword t0 = 10;
  double n_left;
  double n_right;
  
  cube A_sum_left;
  cube A_sum_right;
  cube A_bar_left;
  cube A_bar_right;
  cube A_sum_left_0;
  cube A_sum_right_0;
  
  cube P_left;
  cube P_right;
  
  vec D_K_t_max_rescale = vec(TT, fill::zeros);
  double rescale;
  
  A_sum_left_0.zeros(n1, n2, L);
  for (uword i = 0; i < buffer_left; i++){
    A_sum_left_0 += A_list(i);
  }
  A_sum_right_0.zeros(n1, n2, L);
  A_sum_right_0 += A_list(buffer_left);
  
  for (uword t = 1; t < TT; t++){
    vec D_K_t = vec(t, fill::zeros);
    
    if (t > buffer_left){
      A_sum_right_0 += A_list(t);
    }
    
    if (t - buffer_right < buffer_left){
      continue;
    }
    else{
      A_sum_left = cube(A_sum_left_0);
      A_sum_right = cube(A_sum_right_0);
      
      for (uword s = buffer_left; s < t - buffer_right - 1; s++){
        
        n_left = 0.0 + s + 1.0;
        n_right = 0.0 + t - s;
        
        A_sum_left += A_list(s);
        A_sum_right -= A_list(s);
        
        A_bar_left = A_sum_left / n_left;
        A_bar_right = A_sum_right / n_right;
        
        P_left = uase_cpp(A_bar_left, r_hat);
        P_right = uase_cpp(A_bar_right, r_hat);
        
        rescale = (1.0/pow(n_left, 0.5) + 1.0/pow(n_right, 0.5)) * pow(log(1.0 / alpha * std::max(t0, t + 1)), 0.5);
        
        D_K_t(s) = l2_cube_cpp(P_left - P_right) / rescale;
      }
    }
    
    D_K_t_max_rescale[t] = max(D_K_t);
    
    if (verbose){
      Rprintf("%i \n", t + 1);
    }
    
    if (D_K_t_max_rescale[t] > tau_factor){
      return List::create( 
        _["t"] = t + 1,
        _["D_K_t_max_rescale"] = D_K_t_max_rescale,
        _["find"] = true
      );
    }
    
  }
  return List::create( 
    _["t"] = TT + 1, 
    _["D_K_t_max_rescale"] = D_K_t_max_rescale,
    _["find"] = false
  );
}


