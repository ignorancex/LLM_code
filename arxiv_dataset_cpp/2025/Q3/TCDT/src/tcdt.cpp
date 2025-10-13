#include <RcppArmadillo.h>
#include <cmath>
#include <limits> // Required for std::numeric_limits
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// Forward declaration
inline double kernel_1d(double u, double h, int kern_type, double nu, double tiny_num);

// [[Rcpp::export]]
double teststatg(int n1, int n2,
                 const arma::mat& kY11, const arma::mat& kY22, const arma::mat& kY12,
                 const arma::mat& G1_11, const arma::mat& G1_12,
                 const arma::mat& G2_22, const arma::mat& G2_21,
                 const arma::vec& S1_1, const arma::vec& S1_2,
                 const arma::vec& S2_1, const arma::vec& S2_2) {

  double T1 = 0.0;
  double T2 = 0.0;
  double T3 = 0.0;
  double T = 0.0;

  double tmp = 0.0;

  for (int i = 0; i < n1; i++) {
    tmp = 0.0;
    for (int j = 0; j < n1; j++) {
      if(i != j) {
        tmp += kY11(i,j) * G1_11(i,j);
      }
    }
    T1 += tmp * S2_1(i);
  }

  for (int l = 0; l < n2; l++) {
    tmp = 0.0;
    for (int m = 0; m < n2; m++) {
      if(l != m) {
        tmp += kY22(l,m) * G2_22(l,m);
      }
    }
    T2 += tmp * S2_2(l);
  }

  for (int i = 0; i < n1; i++) {
    for (int l = 0; l < n2; l++) {
      T3 += kY12(i,l) * (G1_12(i,l) + G2_21(l,i)) * (S1_2(l) - G1_12(i,l)) * (S1_1(i) - G2_21(l,i));
    }
  }

  T = (T1 + T2 - T3) / (n1 * (n1 - 1)) / (n2 * (n2 - 1));

  return T;

}

// [[Rcpp::export]]
double teststatl(int n1, int n2,
                 const arma::mat& kY11, const arma::mat& kY22, const arma::mat& kY12,
                 const arma::vec& G1X, const arma::vec& G2X,
                 double S1_1, double S1_2, double S2_1, double S2_2) {

  double T1 = 0.0;
  double T2 = 0.0;
  double T3 = 0.0;
  double T = 0.0;

  for (int i = 0; i < (n1 - 1); i++) {
    for (int j = (i + 1); j < n1; j++) {
      T1 += kY11(i,j) * G1X(i) * G1X(j);
    }
  }
  if (S2_1 != 0) T1 /= S2_1 / 2.0; else T1 = 0;


  for (int l = 0; l < (n2 - 1); l++) {
    for (int m = (l + 1); m < n2; m++) {
      T2 += kY22(l,m) * G2X(l) * G2X(m);
    }
  }
  if (S2_2 != 0) T2 /= S2_2 / 2.0; else T2 = 0;


  for (int i = 0; i < n1; i++) {
    for (int l = 0; l < n2; l++) {
      T3 += kY12(i,l) * G1X(i) * G2X(l) * (S1_1 - G1X(i)) * (S1_2 - G2X(l));
    }
  }
  if (S2_1 != 0 && S2_2 != 0) T3 /= S2_1 * S2_2; else T3 = 0;


  T = T1 + T2 - 2.0 * T3;

  return T;
}

inline double kernel_1d(double u, double h, int kern_type, double nu, double tiny_num) {
  if (h <= tiny_num) return 0.0; // Avoid division by zero if h is tiny
  double val = 0.0;
  double uh = u / h;
  double uh_sq = uh * uh; // Precompute square

  if (kern_type == 1) { // Gaussian
    double std_norm_pdf_uh = ::Rf_dnorm4(uh, 0.0, 1.0, 0);

    if (nu == 2) {
      val = std_norm_pdf_uh / h;
    } else if (nu == 4) {
      val = 0.5 * (3.0 - uh_sq) * std_norm_pdf_uh / h;
    } else if (nu == 6) {
      double uh_qu = uh_sq * uh_sq;
      val = (1.0 / 8.0) * (15.0 - 10.0 * uh_sq + uh_qu) * std_norm_pdf_uh / h;
    } else {
      Rcpp::warning("Gaussian kernel only implemented for nu=2, 4, 6 in Rcpp. Falling back to nu=2.");
      val = std_norm_pdf_uh / h;
    }
  } else if (kern_type == 2) { // Epanechnikov
    if (nu == 2) {
      if (std::abs(uh) <= 1.0) {
        val = 0.75 * (1.0 - uh_sq) / h;
      } else {
        val = 0.0;
      }
    } else {
      Rcpp::warning("Epanechnikov kernel only implemented for nu=2 in Rcpp.");
      if (std::abs(uh) <= 1.0) val = 0.75 * (1.0 - uh_sq) / h; else val = 0.0;
    }
  } else if (kern_type == 3) { // Uniform
    if (nu == 2) {
      if (std::abs(uh) <= 1.0) {
        val = 0.5 / h;
      } else {
        val = 0.0;
      }
    } else {
      Rcpp::warning("Uniform kernel only implemented for nu=2 in Rcpp.");
      if (std::abs(uh) <= 1.0) val = 0.5 / h; else val = 0.0;
    }
  } else {
    Rcpp::stop("Unsupported kernel type in Rcpp");
  }

  return (std::isfinite(val)) ? val : 0.0;
}

//' @title CV Objective Function (C++ Backend)
//' @description Calculates the (potentially locally weighted) leave-one-out
//' cross-validation loss.
//' @param C A numeric vector of scaling factors for the bandwidth.
//' @param X A numeric matrix of predictors.
//' @param kY The precomputed RKHS kernel matrix for the response.
//' @param local_weights A numeric vector of pre-computed weights for each
//'   observation, used for locally weighted CV. For standard CV, this should
//'   be a vector of ones.
//' @param kappa The exponent for the sample size in the bandwidth formula.
//' @param nu The order of the smoothing kernel.
//' @param kern_type An integer mapping to the kernel function.
//' @param lower_C The lower bound for C values.
//' @param tiny_num A small number to prevent division by zero.
//' @keywords internal
// [[Rcpp::export]]
double cv_objective(const arma::vec& C,
                    const arma::mat& X,
                    const arma::mat& kY,
                    const arma::vec& local_weights,
                    double kappa,
                    double nu,
                    int kern_type,
                    double lower_C,
                    double tiny_num = 1e-12) {

  const double LARGE_PENALTY = 1e10;
  int n = X.n_rows;
  int p = X.n_cols;

  if (arma::any(C <= lower_C)) {
    return LARGE_PENALTY;
  }

  arma::vec h_vec = C * std::pow(n, -kappa);

  if (arma::any(h_vec <= tiny_num)) {
    return LARGE_PENALTY;
  }

  double cv_sum = 0.0;

  // Outer loop for Leave-One-Out (for each observation i)
  for (int i = 0; i < n; ++i) {
    // If the weight for this observation is zero, we can skip the calculation
    if (local_weights(i) == 0) {
        continue;
    }
      
    arma::vec loo_weights(n, arma::fill::zeros);
    double Wi_sum = 0.0; // Sum of leave-one-out Nadaraya-Watson weights

    // Inner loop to calculate weights w_ij = G_h(X_j - X_i) for a fixed i
    for (int j = 0; j < n; ++j) {
      if (i == j) continue; // Leave-one-out

      arma::rowvec diff_ij = X.row(j) - X.row(i);
      double sign_G_prod = 1.0;
      double log_abs_G_prod = 0.0;

      // Product kernel over dimensions in log space
      for (int k = 0; k < p; ++k) {
        double G_val = kernel_1d(diff_ij(k), h_vec(k), kern_type, nu, tiny_num);
        if (G_val < 0) {
          sign_G_prod *= -1.0;
        }
        double abs_G_val = std::abs(G_val);
        if (abs_G_val > tiny_num) {
          log_abs_G_prod += std::log(abs_G_val);
        } else {
          log_abs_G_prod = -std::numeric_limits<double>::infinity();
          break; 
        }
      }

      double current_weight = sign_G_prod * std::exp(log_abs_G_prod);
      if (!std::isfinite(current_weight)) {
          current_weight = 0.0;
      }

      loo_weights(j) = current_weight;
      Wi_sum += current_weight;
    }

    if (std::abs(Wi_sum) < tiny_num) return LARGE_PENALTY;

    // Calculate the three terms of the squared norm expansion:
    // || \hat{Pi}^{-i} - k(Y_i,.) ||^2 = term1 - 2*term2 + term3
    double term1 = arma::as_scalar(loo_weights.t() * kY * loo_weights) / (Wi_sum * Wi_sum);
    double term2 = arma::as_scalar(loo_weights.t() * kY.col(i)) / Wi_sum;
    double term3 = kY(i, i);

    double loss_i = term1 - 2.0 * term2 + term3;

    if (!std::isfinite(loss_i)) {
      return LARGE_PENALTY;
    }

    // Apply the local weight and add to the total CV sum
    cv_sum += local_weights(i) * loss_i;
  }

  if (!std::isfinite(cv_sum)) {
    return LARGE_PENALTY;
  }

  return cv_sum;
}
