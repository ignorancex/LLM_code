/*
 * This code is largely based on the SSLB (Spike-and-Slab Lasso Biclustering) R package
 * developed by Gemma Moran et al., available at: https://github.com/gemoran/SSLB
 *
 * The following functions are new modifications to the original package:
 * - E_step_Gamma_tilde_multi
 * - MC_gradient_estimation  
 * - M_step_Weights_multi
 *
 * Additional modifications were made to the main cOGSSLB function to support these changes.
 */

#include <RcppArmadillo.h>
#include <math.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;
using namespace Rcpp;
using namespace arma;

// check convergence
int check_convergence(mat &B, mat &B_old, double EPSILON) {
  int G = B.n_rows;
  int K = B.n_cols;
  double diff = 0;
  int converged = 1;
  for (int j = 0; j < G; j++) {
    for (int k = 0; k < K; k++) {
      diff = fabs((B(j, k) - B_old(j, k))/B_old(j, k));
      if (diff > EPSILON) {
        converged = 0;
        break;
      }
    }
    if (converged == 0) {
      break;
    }
  }
  return converged;
}

// E-Step for factor matrix X
void E_step_X(mat &X, mat &V_traces, mat &ML, mat &B, vec &sigmas, mat &Tau, mat &Y) { 
  
  int K = Tau.n_cols;
  mat SinvB = B.each_col() / sigmas;
  mat tBSinvB = trans(B) * SinvB;
  mat M = zeros<mat>(K, K);
  mat V = zeros<mat>(K, K);
  mat Vinv = zeros<mat>(K, K);
  vec X_mean = zeros<vec>(K);
  rowvec e(K);
  e.fill(1e-6);
  mat keep_sympd = diagmat(e);

  for(int i = 0; i < Tau.n_rows; i++) {
    Vinv = tBSinvB + diagmat(1/(Tau.row(i) + e));
    V = inv_sympd(Vinv + keep_sympd);
    V_traces.row(i) = trans(diagvec(V));
    X_mean = V * trans(B) * (1/sigmas % trans(Y.row(i)));
    M += V;
    X.row(i) = trans(X_mean);
  }

  ML = chol(M, "lower");
}

// E-Step for factor indicator matrix Gamma_tilde
// First function to modify:
mat E_step_Gamma_tilde_multi(mat &disease_status, mat &weights, mat &Tau,
                             vec &theta_tildes, double lambda1_tilde,
                             double lambda0_tilde, int niter_exp_y = 1000,
                             bool flag_plot_conv = false, int n_lambda0_to_plot = 0,
                             string dir_save_conv = "conv_data") {

  int N = Tau.n_rows;
  int K = Tau.n_cols;
  int n_classes = disease_status.n_cols;

  // compute p(\gamma_tilde_ik = 1 | T, \theta_tilde)
  mat mult_mat = zeros<mat>(N, K);
  mult_mat.each_row() = trans((1 - theta_tildes)/theta_tildes);
  mult_mat = (pow(lambda0_tilde, 2)/pow(lambda1_tilde, 2)) * mult_mat;
  mat denom = 1 + mult_mat % exp(-Tau * (pow(lambda0_tilde, 2) - pow(lambda1_tilde, 2)) / 2);
  
  // here we have p(\gamma_tilde_ik = 1 | T, \theta_tilde)
  mat Gamma_tilde = 1/denom;

  // we now compute p(\gamma_tilde_ik = 1 | y, T, W, \theta_tilde)

  // we will save p(y_i | \gamma_tilde_ik = 1, W) in the following matrices
  mat prob_y_given_gammas_1 = zeros<mat>(N, K);
  mat prob_y_given_gammas_0 = zeros<mat>(N, K);

  // for each i, we generate MC samples from  p(y | \gamma_tilde_ik, T, \theta_tilde)
  int n_samples_mc = niter_exp_y;

  // we will declare a vector of ones that we will use later
  vec one_n_samples = ones<vec>(n_samples_mc);

  for (int i = 0; i < N; ++i) {

    // to save samples for each i
    umat gamma_tilde_samples = zeros<umat>(n_samples_mc, K);

    // create matrix of uniform distributed random values
    mat unif_vector = randu<mat>(n_samples_mc, K);

    // generating samples
    for (int j = 0; j < n_samples_mc; ++j) {
      gamma_tilde_samples.row(j) = unif_vector.row(j) < Gamma_tilde.row(i);
    }

    // add a column of ones (add intersection column)
    uvec uone(gamma_tilde_samples.n_rows, fill::ones);
    gamma_tilde_samples.insert_cols(0, uone);

    // Casting umat to mat
    mat gamma_tilde_samples_mat = conv_to<mat>::from(gamma_tilde_samples);

    // for multinomial logistic regression case - class index where y_i,c = 1
    int index_true_class;

    // we identify the y=1 class in the ith sample (in the mult. case)
    if (n_classes > 1) {
      for (int l = 0; l < n_classes; l++) {
        if (disease_status(i, l) == 1) {
          index_true_class = l;
          break;
        }
      }
    }

    // Declare p(y_i | \gamma_tilde_ik = 1, W)
    vec Py_given_samples_class;

    // computing p(y_i | \gamma_tilde_ik = 1, W)
    if (n_classes > 1) {
      mat Py_given_samples = exp(gamma_tilde_samples_mat * weights);
      Py_given_samples_class = Py_given_samples.col(index_true_class) / sum(Py_given_samples, 1);
    } else {
      Py_given_samples_class = 1 / (1 + exp(-gamma_tilde_samples_mat * weights));
    }

    if (flag_plot_conv && i == 5) {
      vec py_to_plot = cumsum(Py_given_samples_class) / regspace(1, 1, Py_given_samples_class.n_elem);
      py_to_plot.save(dir_save_conv + "/pr_y_" + to_string(n_lambda0_to_plot) + ".csv", csv_ascii);
    }

    // go for each k (column) of samples matrix and compute the probability
    for (int k = 0; k < K; k++) {
      // sometimes sum(gamma_tilde_samples_mat.col(k)) = 0.
      // If that happens, set prob_y_given_gammas_* to 1e-16
      double n_ones_col_k = sum(gamma_tilde_samples_mat.col(k + 1));

      if (n_ones_col_k == 0) {
        prob_y_given_gammas_1(i,k) = 1e-16;
        prob_y_given_gammas_0(i,k) = sum(Py_given_samples_class) / n_samples_mc;
      } else if (n_ones_col_k == n_samples_mc) {
        prob_y_given_gammas_1(i,k) = sum(Py_given_samples_class) / n_samples_mc;
        prob_y_given_gammas_0(i,k) = 1e-16;
      }
      else {
        prob_y_given_gammas_1(i,k) = sum(Py_given_samples_class % gamma_tilde_samples_mat.col(k + 1)) / n_ones_col_k;
        prob_y_given_gammas_0(i,k) = sum(Py_given_samples_class % (one_n_samples - gamma_tilde_samples_mat.col(k + 1))) / (n_samples_mc - n_ones_col_k);
      }

    }

  }

  mat prob_gammas_1 = prob_y_given_gammas_1 % Gamma_tilde;
  mat prob_gammas_0 = prob_y_given_gammas_0 % (1 - Gamma_tilde);
  mat Gamma_tilde_with_y = 1 / (1 + (prob_gammas_0 / prob_gammas_1)); 

  return Gamma_tilde_with_y;
  
}

// Monte Carlo estimation of gradient for logistic regression
// needed for M_step_Weights_multi
mat MC_gradient_estimation(mat &Gamma_tilde, mat &Gamma_tilde_iter, mat &sample_weights, mat &disease_output,
                         int n_samples, int n_features, int n_classes, int max_iter_mc,
                         bool plot_conv = FALSE, bool is_AGD = FALSE, bool iter_to_plot = FALSE,
                         string dir_save_conv = "conv_data", int n_lambda0_to_plot = 0) {
    // Initialize matrices
    mat probs_mc_est = zeros<mat>(n_samples, n_classes);
    mat grad_mc_sample;
    umat gamma_tilde_samples = zeros<umat>(n_samples, n_features + 1);
    
    // Add intersection column
    gamma_tilde_samples.col(0) = ones<uvec>(n_samples);
    
    // Initialize progress tracking vector
    vec mc_progress = zeros<vec>(max_iter_mc);
    
    for (int m = 0; m < max_iter_mc; ++m) {
        // Generate Gamma_tilde samples
        mat unif_temp(n_samples, n_features, fill::randu);
        gamma_tilde_samples(span::all, span(1, n_features)) = unif_temp < Gamma_tilde_iter;
        mat probs;
        
        // Compute probabilities based on number of classes
        if (n_classes > 1) {
            mat scores_all = zeros<mat>(n_samples, n_classes);
            scores_all(span::all, span(0, n_classes - 2)) = gamma_tilde_samples * sample_weights(span::all, span(0, n_classes - 2));
            probs = exp(scores_all);
            probs.each_col() /= sum(probs, 1);
        } else {
            mat scores_all = gamma_tilde_samples * sample_weights;
            probs = 1 / (1 + exp(-scores_all));
        }
        
        // Update MC estimate
        probs_mc_est = ((static_cast<double>(m) / (m + 1)) * probs_mc_est) + ((1.0 / (m + 1)) * probs);
        
        // Save progress if requested
        if (plot_conv && is_AGD && iter_to_plot) {
            mc_progress(m) = norm(probs_mc_est, "fro");
        }
    }
    
    // Save progress file if requested
    if (plot_conv && is_AGD && iter_to_plot) {
        mc_progress.save(dir_save_conv + "/grad_iter_" + to_string(n_lambda0_to_plot) + ".csv", csv_ascii);
    }
    
    // Compute gradient
    grad_mc_sample = Gamma_tilde.t() * (disease_output - probs_mc_est);
    
    // Set reference class to zero for multinomial case
    if (n_classes > 1) {
        grad_mc_sample.col(grad_mc_sample.n_cols - 1) = zeros<vec>(grad_mc_sample.n_rows);
    }
    
    return grad_mc_sample;
}

// M step for the logistic regression weights
// Second function added to the original SSLB:
// I'm using gradient descent together with Monte Carlo
// to infer the weights of the logistic regression 
List M_step_Weights_multi(mat &disease_output, mat &Gamma_tilde_iter, mat &Tau, vec &theta_tildes,
                   double lambda1_tilde, double lambda0_tilde, mat &initial_weights, mat &initial_sample_weights,
                   bool flag_plot_conv = false, int n_lambda0_to_plot = 0, double stepsize = 0.1,
                   int niter_burnIn_ULA_SOUL = 20, int niter_ULA_SOUL = 50,
                   int max_iters = 50, int max_iter_mc = 30, string dir_save_conv = "conv_data",
                   double init_l2_reg_log_reg = 0.5, double l2_reg_log_reg = 0.5, bool manual_set_stepsize_hyperparam_logreg = true,
                   double perc_max_stepsize_grad_desc = 0.5, int iter_stepsize_l2_reg_est = 1){

  int n_features = Tau.n_cols;
  int n_samples = Tau.n_rows;
  int n_classes = disease_output.n_cols;

  // I'm gonna save the Bernoulli samples here
  umat gamma_tilde_samples;

  // Redefine the covariates matrix to include intersection
  mat gamma_tilde = zeros<mat>(n_samples, n_features + 1);
  gamma_tilde.col(0) = ones<vec>(n_samples);
  gamma_tilde(span::all, span(1, n_features)) = Gamma_tilde_iter;

  // Initialize logistic regression weights
  mat weights = 0.1 * initial_weights;
  // mat sample_weights = initial_sample_weights;
  mat sample_weights = initial_weights;

  // Initialize AGD parameters
  mat y_agd = weights;
  mat weights_minus_1 = weights;
  double t_agd = 0;
  double t_agd_minus_1 = 0;

  // Initialize gradient matrix
  mat grad_mc_sample;

  // Initialize the L2 hyperparameter value to estimate
  double lambda_l2 = l2_reg_log_reg;

  // to save the final lambda estimate
  double final_lambda_reg = 0;

  // Set baseline class (for mult. logistic regression)
  if (n_classes > 1) {
    weights.col(n_classes - 1) = zeros<vec>(n_features + 1);
  }

  // set vector of values to save mc progress
  vec mc_progress = zeros<vec>(max_iter_mc);
  vec weight_progress;

  // save the estimate of the mean of the prob matrix
  mat probs_mc_est = zeros<mat>(n_samples, n_classes);

  // to save progress of the lambda estimate if plot = true
  vec list_lambda_l2 = zeros<vec>(niter_ULA_SOUL);
  // to save norm of weight_samples if plot = true
  vec samples_weight_progress = zeros<vec>(niter_ULA_SOUL);

  // for stepsize computation
  // Calculate the maximum bound of the Hessian
  double eig_max_value = 0;
  if (n_classes > 1) {
    vec eigval = eig_sym(0.5 * kron(eye(disease_output.n_cols, disease_output.n_cols) - ((1 / disease_output.n_cols) * ones<vec>(disease_output.n_cols) * ones<vec>(disease_output.n_cols).t()), gamma_tilde.t() * gamma_tilde));
    eig_max_value = eigval(eigval.n_elem - 1);
  } else {
    vec eigval = eig_sym(gamma_tilde.t() * gamma_tilde);
    eig_max_value = 0.25 * eigval(eigval.n_elem - 1);
  }

  // stepsize
  stepsize = perc_max_stepsize_grad_desc * 2 / (eig_max_value + lambda_l2);

  // if the lambda_reg parameter is estimated  
  if (!manual_set_stepsize_hyperparam_logreg) {
    // leave 25 iterations for the true estimation of lambda
    int thresh_iter_lambda;
    if (niter_ULA_SOUL < 20) {
      thresh_iter_lambda = round(niter_ULA_SOUL / 2.0);
    } else {
      thresh_iter_lambda = 10;
    }
    // dimension of W - for stepsize computation of the SAPG algorithm
    double d_sapg;
    if (sample_weights.n_cols > 1) {
      d_sapg = sample_weights.n_rows * (sample_weights.n_cols - 1);
    } else {
      d_sapg = sample_weights.n_rows;
    }
    // to save variable stepsize in the SAPG algorithm
    double step_size_lambda_est = 0;
    // to save the mean of the l2_lambda estimate
    double mean_lambda_est = 0;
    // to compute projections in projected gradient descent
    double min_lambda_l2 = 1e-3;
    double max_lambda_l2 = 50;
    double min_eta_lambda_l2 = log(min_lambda_l2);
    double max_eta_lambda_l2 = log(max_lambda_l2);
    // C_0 constant for the non-decreasing step-size in the SAPG algorithm
    double C_0 = 5 / (init_l2_reg_log_reg * d_sapg);
    // to save the progress of the convergence of the weights
    weight_progress = zeros<vec>(max_iters + 1);
    weight_progress(0) = norm(weights, "fro");
    // change of variable to log-scale, as recommended in paper
    double eta_lambda_l2 = log(lambda_l2);

    // burn-in stage
    for (int i = 0; i < niter_burnIn_ULA_SOUL; i++) {
      // MCMC sample (ULA algorithm)
      mat perturb_term = sqrt(2 * stepsize) * randn(sample_weights.n_rows, sample_weights.n_cols);
      if (sample_weights.n_cols > 1) {
        sample_weights.col(n_classes - 1) = zeros<vec>(sample_weights.n_rows);
      }

      // MC gradient estimation wihtout regularisation
      grad_mc_sample = MC_gradient_estimation(gamma_tilde, Gamma_tilde_iter, sample_weights,
                                              disease_output, n_samples, n_features,
                                              n_classes, max_iter_mc);

      // complete gradiente with regularisation
      mat complete_grad = grad_mc_sample - (lambda_l2 * sample_weights);

      // samples update
      sample_weights += (stepsize * complete_grad) + perturb_term;

    }

    // Sampling stage
    for (int i = 0; i < niter_ULA_SOUL; i++) {
      // MCMC sample (ULA algorithm)
      mat perturb_term = sqrt(2 * stepsize) * randn(sample_weights.n_rows, sample_weights.n_cols);
      if (sample_weights.n_cols > 1) {
        sample_weights.col(n_classes - 1) = zeros<vec>(sample_weights.n_rows);
      }

      // MC gradient estimation wihtout regularisation
      grad_mc_sample = MC_gradient_estimation(gamma_tilde, Gamma_tilde_iter, sample_weights,
                                              disease_output, n_samples, n_features,
                                              n_classes, max_iter_mc);

      // complete gradiente with regularisation
      mat complete_grad = grad_mc_sample - (lambda_l2 * sample_weights);

      // samples update
      sample_weights += (stepsize * complete_grad) + perturb_term;

      // save progress of norm of samples
      samples_weight_progress(i) = norm(sample_weights, "fro");

      // estimate lambda_reg
      step_size_lambda_est = C_0 * pow(iter_stepsize_l2_reg_est, -0.8);

      // lambda update - log scale
      eta_lambda_l2 += 0.5 * step_size_lambda_est * ((d_sapg / lambda_l2) - accu(pow(sample_weights, 2))) * exp(eta_lambda_l2);
      // projection step
      eta_lambda_l2 = min(max_eta_lambda_l2, max(eta_lambda_l2, min_eta_lambda_l2));
      // lambda update - normal scale
      lambda_l2 = exp(eta_lambda_l2);
      // to save progress of the lambda estimate
      list_lambda_l2(i) = lambda_l2;
      // to save the last thresh_iter_lambda to compute the final estimation of lambda
      if (i >= thresh_iter_lambda) {
        mean_lambda_est += lambda_l2;
      }
      // update step-size
      stepsize = perc_max_stepsize_grad_desc * 2 / (eig_max_value + lambda_l2);
      // update stepsize iteration index
      iter_stepsize_l2_reg_est++;
    }
    // save .csv file with lambda_l2 and sample_weight_norm progress, if flag_plot is true
    if (flag_plot_conv) {
      list_lambda_l2.save(dir_save_conv + "/lambda_l2_iter_" + to_string(n_lambda0_to_plot) + ".csv", csv_ascii);
      samples_weight_progress.save(dir_save_conv + "/sample_weight_iter_" + to_string(n_lambda0_to_plot) + ".csv", csv_ascii);
    }

    // Weights estimate step with lambda_reg estimated
    // compute mean_lambda_est
    mean_lambda_est = mean_lambda_est / (niter_ULA_SOUL - thresh_iter_lambda);

    // update learning rate
    stepsize = perc_max_stepsize_grad_desc * 2 / (eig_max_value + mean_lambda_est);

    for (int i = 0; i < max_iters; i++) {
      // Monte Carlo step to generate M bernoulli realisations
      // of Gamma_tilde and estimate P for gradient

      // parameters update - AGD
      t_agd_minus_1 = t_agd;
      t_agd = 0.5 * (1 + sqrt(1 + 4 * pow(t_agd, 2)));
      y_agd = weights + ((t_agd_minus_1 - 1) * (weights - weights_minus_1) / t_agd);

      // Gradient estimation
      // check if it's the last iteration to save convergence of MC gradient
      bool is_iter_to_plot = FALSE;
      if (flag_plot_conv && i == max_iters - 1) {
        is_iter_to_plot = TRUE;
      }

      // MC gradient estimate without regularisation
      grad_mc_sample = MC_gradient_estimation(gamma_tilde, Gamma_tilde_iter, y_agd,
                                              disease_output, n_samples, n_features,
                                              n_classes, max_iter_mc, flag_plot_conv,
                                              TRUE, is_iter_to_plot, dir_save_conv,
                                              n_lambda0_to_plot);

      // complete gradiente with regularisation
      mat complete_grad = grad_mc_sample - (mean_lambda_est * y_agd);

      // weights update - AGD
      weights_minus_1 = weights;
      weights = y_agd + (stepsize * complete_grad);

      // save norm of the weights update
      if (flag_plot_conv) {
        weight_progress(i + 1) = norm(weights, "fro");
      }
    }

    // to save the final lambda estimate
    final_lambda_reg = mean_lambda_est;

  } else {

    // to save the progress of the convergence of the weights
    weight_progress = zeros<vec>(max_iters + 1);
    weight_progress(0) = norm(weights, "fro");

    for (int i = 0; i < max_iters; i++) {
      // Monte Carlo step to generate M bernoulli realisations
      // of Gamma_tilde and estimate P for gradient

      // parameters update - AGD
      t_agd_minus_1 = t_agd;
      t_agd = 0.5 * (1 + sqrt(1 + 4 * pow(t_agd, 2)));
      y_agd = weights + ((t_agd_minus_1 - 1) * (weights - weights_minus_1) / t_agd);

      // check if it's the last iteration to save convergence of MC gradient
      bool is_iter_to_plot = FALSE;
      if (flag_plot_conv && i == max_iters - 1) {
        is_iter_to_plot = TRUE;
      }

      // MC gradient estimate with regularisation
      grad_mc_sample = MC_gradient_estimation(gamma_tilde, Gamma_tilde_iter, y_agd,
                                              disease_output, n_samples, n_features,
                                              n_classes, max_iter_mc, flag_plot_conv,
                                              TRUE, is_iter_to_plot, dir_save_conv,
                                              n_lambda0_to_plot);

      // complete gradiente without regularisation
      mat complete_grad = grad_mc_sample - (lambda_l2 * y_agd);

      // weights update - AGD
      weights_minus_1 = weights;
      weights = y_agd + (stepsize * complete_grad);

      // save norm of the weights update
      if (flag_plot_conv) {
        weight_progress(i + 1) = norm(weights, "fro");
      }
    }

    // to save the final lambda estimate
    final_lambda_reg = lambda_l2;
  }

  // save convergence of weights to plot
  if (flag_plot_conv) {
    weight_progress.save(dir_save_conv + "/weights_iter_" + to_string(n_lambda0_to_plot) + ".csv", csv_ascii);
  }

  List Output = List::create(
    Named("weights") = wrap(weights),
    Named("sample_weights") = wrap(sample_weights),
    Named("stepsize") = stepsize,
    Named("lambda_reg_est") = final_lambda_reg,
    Named("iter_stepsize_l2_reg_est") = iter_stepsize_l2_reg_est
  );

  return Output;
  
}

// M-step for error variances sigmas
vec M_step_sigmas(mat &Y, mat &B, mat &X, double eta, double xi, double sigma_min) {
  int G = Y.n_cols;
  int N = Y.n_rows;
  vec sigmas = zeros<vec>(G);
  double resid = 0;
  for (int j = 0; j < G; j++) {
    resid = sum(pow(Y.col(j) - X * trans(B.row(j)), 2));
    sigmas(j) = (resid + eta * xi) / (N + eta + 2);
    if (sigmas(j) < sigma_min) {
      sigmas(j) = eta * xi / (eta + 2);
    }
  }
  return sigmas;
}

// Functions for SSLASSO
double sumsq(vec y) {
  int n = y.n_elem;
  double sumsq = 0;
  for(int i = 0; i < n; i++) {
    sumsq += pow(y(i), 2);
  }
  return sumsq;
}

double pstar(double x, double theta, double lambda1, double lambda0){
  double value;
  if (lambda1 == lambda0){
    return 1;
  } else{
    value = (1 - theta)/theta * lambda0 / lambda1 * exp(-fabs(x) * (lambda0 - lambda1));
    value += 1;
    value = 1/value;
    return value;
  }
}

double lambdastar(double x, double theta, double lambda1, double lambda0){
  double aux;
  if (lambda1 == lambda0){
    return lambda1;
  } else{
    aux = pstar(x, theta, lambda1, lambda0);
    return aux * lambda1 + (1 - aux) * lambda0;
  }
}

double SSL(double z, double beta, double lambda1, double lambda0, double theta, double sumsq, double delta, double sigma2) {
  double s=0;
  double lambda;
  if (z > 0) s = 1;
  else if (z < 0) s = -1;
  if (fabs(z) <= delta) {
    return(0);
  } else { 
    lambda=lambdastar(beta, theta, lambda1, lambda0);
    double temp;
    temp = fabs(z) - sigma2 * lambda;
    if (temp > 0) {
      return(temp * s / sumsq);
    } else {
      return(0);  
    }
  }
}

double g(double x, double theta, double sigma2, double lambda1, double lambda0, double sumsq){
  double value=lambdastar(x,theta,lambda1,lambda0); 
  return pow((value-lambda1),2)+2*sumsq/sigma2*log(pstar(x,theta,lambda1,lambda0));
}


double threshold(double theta, double sigma2, double lambda1, double lambda0, double sumsq){ 
  if (lambda0 == lambda1){
    return sigma2 * lambda1;
  } else {
    if (g(0, theta, sigma2, lambda1 ,lambda0, sumsq) > 0){
      return sqrt(2 * sumsq * sigma2 * log(1/pstar(0, theta, lambda1, lambda0))) + sigma2 * lambda1;     
    }  else {    
      return sigma2 * lambdastar(0, theta, lambda1, lambda0);     
    }
  }
}

// SSLASSO
vec SSLASSO(vec &y, mat &X, vec &X_sumsq, vec &beta_old, double lambda1, double lambda0, vec &thetas, double sigma2) {
  int K = X.n_cols;
  vec beta = beta_old;
  double eps = 0.001;
  int max_iter = 500;
  int iter = 0;
  vec r = y - X * beta;
  vec z = zeros<vec>(K);
  double delta;

  while (iter < max_iter) {
    iter++;

    for(int k = 0; k < K; k++) {
      delta = threshold(thetas(k), sigma2, lambda1, lambda0, X_sumsq(k));
      z(k) = dot(X.col(k), r) + X_sumsq(k) * beta_old(k);
      beta(k) = SSL(z(k), beta_old(k), lambda1, lambda0, thetas(k), X_sumsq(k), delta, sigma2);
      double shift = beta(k) - beta_old(k);
      if (shift != 0) {
        r -= shift * X.col(k);
      }
    } 
    if (norm(beta - beta_old, 2) < eps) {
      break;
    }
    beta_old = beta;    
  }
  return beta;
}


mat M_step_B(mat &Y, mat &B_old, mat &X, mat &ML, vec &sigmas, vec &thetas, double lambda1, double lambda0) {
  int K = X.n_cols;
  int G = Y.n_cols;
  mat Y_star = join_cols(Y, zeros<mat>(K, G));
  mat X_star = join_cols(X, ML);
  mat B = zeros<mat>(G, K);
  vec beta_old;
  vec X_sumsq(K);
  for(int k = 0; k < K; k++) {
    X_sumsq(k) = sumsq(X_star.col(k));
  }
  vec Yj;
  vec out = zeros<vec>(K);
  for(int j = 0; j < G; j++) {
    Yj = Y_star.col(j);
    beta_old = trans(B_old.row(j));
    out = SSLASSO(Yj, X_star, X_sumsq, beta_old, lambda1, lambda0, thetas, sigmas(j));
    B.row(j) = trans(out);
  }
  return B;
}

vec M_step_thetas_approx(mat &B, double a, double b) {
  int K = B.n_cols;
  int G = B.n_rows;
  vec thetas = zeros<vec>(K);
  vec sum_B = zeros<vec>(K);
  for(int k = 0; k < K; k++){
    for(int j = 0; j < G; j++){
      if(B(j, k) != 0) sum_B(k)++;
    }
  }
  thetas = (a + sum_B)/(a + b + G);
  for (int k = 0; k < K; k++) {
    if (thetas(k) <= 0) {
      thetas(k) = 1e-10;
    } 
    if (thetas(k) > 1) {
      thetas(k) = 0.99999;
    }
  }
  return thetas;
}

vec M_step_thetas(mat &Gs, double a, double b) {
  int K = Gs.n_cols;
  vec thetas = zeros<vec>(K);
  int N = Gs.n_rows;
  double G_sum = 0;
  for (int k = 0; k < K; k++) {
    for (int i = 0; i < N; i++) {
      G_sum += Gs(i, k);
    }
    thetas(k) = (a + G_sum - 1) / (a + b + N - 2);
    if (thetas(k) < 0) {
      thetas(k) = 1e-10;
    } 
    if (thetas(k) > 1) {
      thetas(k) = 0.99999;
    }
    G_sum = 0;
  }
  return thetas;
}

mat M_step_Tau(mat &X, mat &Gamma_tilde, mat &V_traces, double lambda1_tilde, double lambda0_tilde) {
  int n = X.n_rows;
  int k = X.n_cols;
  mat lambda0s = zeros<mat>(n, k);
  lambda0s.fill(pow(lambda0_tilde, 2));
  mat lambda1s = zeros<mat>(n, k);
  lambda1s.fill(pow(lambda1_tilde, 2));
  mat Lambda_star = lambda1s % Gamma_tilde + (1 - Gamma_tilde) % lambda0s; 
  mat Tau = (-1 + sqrt(1 + 4 * Lambda_star % (pow(X, 2) + V_traces)))/(2 * Lambda_star);
  Tau.elem( find_nonfinite(Tau) ).fill(1e-8);
  return Tau;
}

// Rescale X and B 
void rescale_X_B(mat &X, mat &B) {
  int K = X.n_cols;
  int N = X.n_rows;
  int G = B.n_rows;
  double X_norm = 0; 
  double B_norm = 0;
  rowvec d = ones<rowvec>(K);

  for(int k = 0; k < K; k++) {
    for (int i = 0; i < N; i++) {
      X_norm += fabs(X(i, k));
    }
    for (int j = 0; j < G; j++) {
      B_norm += fabs(B(j, k));
    }
    if ((X_norm > 0) && (B_norm > 0)) {
      d(k) = pow(X_norm / B_norm, 0.5);
    }
    X_norm = 0;
    B_norm = 0;
  }
  X.each_row() /= d;
  B.each_row() %= d;

}


mat E_step_Q(vec &nus) {
  int K = nus.n_elem;
  mat PQ = zeros<mat>(K, K);
  PQ(0, 0) = 1;
  double mult = 1;
  for (int k = 1; k < K; k++) {
    PQ(k, 0) = 1 - nus(0);
    mult = 1;
    for (int l = 1; l <= k; l++) {
      mult *= nus(l - 1);
      PQ(k, l) = (1 - nus(l)) * mult;
    }
  }
  colvec PQ_sum = sum(PQ, 1);
  PQ_sum(find(PQ_sum == 0)).ones();
  PQ.each_col() /= PQ_sum;

  return PQ;
}

vec M_step_nus_IBP(vec &Gamma_tilde_sum, mat &PQ, double alpha, double d, int N) {
  int K = Gamma_tilde_sum.n_elem;
  vec nus = zeros<vec>(K);
  double a = 0;
  double b = 0;
  double PQ_sum = 0;

  for (int k = 0; k < K; k++) {
    for (int m = k; m < K; m++) {
      a += Gamma_tilde_sum(m);
      b += (N - Gamma_tilde_sum(m)) * PQ(m, k);
    }
    for (int m = k + 1; m < K; m++) {
      for (int i = k + 1; i <= m; i++) {
        PQ_sum += PQ(m, i);
      }
      a += (N - Gamma_tilde_sum(m)) * PQ_sum;
      PQ_sum = 0;
    }
    a += alpha + k * d - 1;
    b += - d;
    nus(k) = a/(a + b);
    if (nus(k) > 1) {
      nus(k) = (a + 1)/(a + b + 2);
    }
    a = 0;
    b = 0;
    if (nus(k) < 0) {
      nus(k) = 1e-8; 
    }
  }
  uvec replace_nus = find_nonfinite(nus);
  for (int i = 0; i < replace_nus.n_elem; i++) {
    nus(replace_nus(i)) = 1/(replace_nus(i) + 1);
  }

  return nus;
}

// [[Rcpp::export(.cOGSSLB)]]
SEXP cOGSSLB(
  SEXP dis_SEXP,
  SEXP Y_SEXP, 
  SEXP B_init,
  SEXP sigmas_init,
  SEXP Tau_init,
  SEXP thetas_init,
  SEXP theta_tildes_init,
  SEXP nus_init,
  SEXP lambda1_SEXP, 
  SEXP lambda0s_SEXP,
  SEXP lambda1_tilde_SEXP,
  SEXP lambda0_tildes_SEXP,
  SEXP weights_SEXP,
  SEXP a_SEXP,
  SEXP b_SEXP,
  SEXP a_tilde_SEXP,
  SEXP b_tilde_SEXP,
  SEXP alpha_SEXP,
  SEXP d_SEXP,
  SEXP eta_SEXP,
  SEXP xi_SEXP,
  SEXP sigma_min_SEXP,
  SEXP IBP_SEXP,
  SEXP EPSILON_SEXP,
  SEXP MAX_ITER_SEXP,
  SEXP plot_conv_SEXP,
  SEXP iter_em_to_plot_SEXP,
  SEXP dir_save_weight_grad_SEXP,
  SEXP l2_reg_log_reg_SEXP,
  SEXP stepsize_graddesc_logreg_SEXP,
  SEXP niter_burnIn_ULA_SOUL_SEXP,
  SEXP niter_ULA_SOUL_SEXP,
  SEXP niter_graddesc_logreg_SEXP,
  SEXP niter_expgrad_graddesc_logreg_SEXP,
  SEXP niter_exp_y_SEXP,
  SEXP manual_set_stepsize_hyperparam_logreg_SEXP,
  SEXP perc_max_stepsize_grad_desc_SEXP) {

  // Convert R objects to Armadillo objects
  // disease output vector of one/zeroes.
  // vec dis = as<vec>(dis_SEXP);
  mat dis = as<mat>(dis_SEXP);
  mat weights = as<mat>(weights_SEXP);
  // set baseline class on the weight matrix (for multinomial case)
  if (weights.n_cols > 1) {
    weights.col(weights.n_cols - 1) = zeros<vec>(weights.n_rows);
  }

  mat Y = as<mat>(Y_SEXP);
  int N = Y.n_rows;
  int G = Y.n_cols;

  mat B = as<mat>(B_init);
  int K_init = B.n_cols;
  int K = K_init;

  mat B_old = B;

  mat X = zeros<mat>(N, K);
  mat V_traces = zeros<mat>(N, K);
  mat ML = zeros<mat>(K, K);

  mat Tau = as<mat>(Tau_init);
  vec sigmas = as<vec>(sigmas_init);
  vec thetas = as<vec>(thetas_init);
  vec nus = as<vec>(nus_init);
  vec theta_tildes = as<vec>(theta_tildes_init); 
  vec lambda0s = as<vec>(lambda0s_SEXP);
  vec lambda0_tildes = as<vec>(lambda0_tildes_SEXP);
  
  mat Gamma_tilde(N, K);
  Gamma_tilde.fill(0.5);

  vec Gamma_tilde_sum(K);
  vec one_to_K = linspace<vec>(0, K - 1, K);

  double lambda1 = as<double>(lambda1_SEXP);
  double lambda1_tilde = as<double>(lambda1_tilde_SEXP);
  double lambda0;
  double lambda0_tilde;
  double l2_reg_log_reg = as<double>(l2_reg_log_reg_SEXP);
  double init_l2_reg_log_reg = l2_reg_log_reg;
  double a = as<double>(a_SEXP);
  double b = as<double>(b_SEXP);
  double a_tilde = as<double>(a_tilde_SEXP);
  double b_tilde = as<double>(b_tilde_SEXP);
  double alpha = as<double>(alpha_SEXP);
  double d = as<double>(d_SEXP);
  double eta = as<double>(eta_SEXP);
  double xi = as<double>(xi_SEXP);
  double sigma_min = as<double>(sigma_min_SEXP);
  int IBP = as<int>(IBP_SEXP);
  if (IBP == 1) {
    theta_tildes(0) = nus(0);
    for (int k = 1; k < K; k++) {
      theta_tildes(k) = theta_tildes(k - 1) * nus(k);
    }
  }
  mat PQ = zeros<mat>(K, K);

  const int L = lambda0s.n_elem;

  double EPSILON = as<double>(EPSILON_SEXP);
  double MAX_ITER = as<double>(MAX_ITER_SEXP);
  bool plot_conv = as<bool>(plot_conv_SEXP);
  double stepsize_graddesc_logreg = as<double>(stepsize_graddesc_logreg_SEXP);
  int niter_burnIn_ULA_SOUL = as<int>(niter_burnIn_ULA_SOUL_SEXP);
  int niter_ULA_SOUL = as<int>(niter_ULA_SOUL_SEXP);
  int niter_graddesc_logreg = as<int>(niter_graddesc_logreg_SEXP);
  int niter_expgrad_graddesc_logreg = as<int>(niter_expgrad_graddesc_logreg_SEXP);
  int niter_exp_y = as<int>(niter_exp_y_SEXP);
  int iter_em_to_plot = as<int>(iter_em_to_plot_SEXP);
  string dir_save_weight_grad = as<string>(dir_save_weight_grad_SEXP);
  bool manual_set_stepsize_hyperparam_logreg = as<bool>(manual_set_stepsize_hyperparam_logreg_SEXP);
  double perc_max_stepsize_grad_desc = as<double>(perc_max_stepsize_grad_desc_SEXP);
  int converged = 0;
  int iter_stepsize_l2_reg_est = 1;

  // To save results of M-step of Computation of Weights
  List M_weights_res;

  mat sample_weights;
  if (!manual_set_stepsize_hyperparam_logreg) {
    sample_weights = randn<mat>(weights.n_rows, weights.n_cols);
    if (weights.n_cols > 1) {
      sample_weights.col(sample_weights.n_cols - 1) = zeros<vec>(sample_weights.n_rows);
    }
  }

  vec NITERS = zeros<vec>(L);
  int ITER = 0;
  int update_lambda0_tilde = 1;

  vec Gamma_tilde_sum_all(L);

  for(int l = 0; l < L; l++) {

    lambda0 = lambda0s(l);
    if (update_lambda0_tilde == 1) {
      lambda0_tilde = lambda0_tildes(l);
    } 

    Rcout << "Lambda............................................" << l + 1 << endl;

    while(NITERS(l) < MAX_ITER) {
      Rcpp::checkUserInterrupt();

      NITERS(l)++;
      ITER = NITERS(l);

      // Update Gamma_tildes
      // now we add the disease outcome at the beginning
      if (lambda1 != lambda0) {
        if (ITER == iter_em_to_plot) {
          Gamma_tilde = E_step_Gamma_tilde_multi(dis, weights, Tau, theta_tildes, lambda1_tilde, lambda0_tilde,
                                                 niter_exp_y, plot_conv, l + 1, dir_save_weight_grad);
        } else {
          Gamma_tilde = E_step_Gamma_tilde_multi(dis, weights, Tau, theta_tildes, lambda1_tilde, lambda0_tilde,
                                                 niter_exp_y, false, l + 1, dir_save_weight_grad);
        }
      }      

      // Re-order in descending order of Gamma_tilde_sum (for IBP)
      for (int k = 0; k < K; k++) {
        Gamma_tilde_sum(k) = 0;
        for (int i = 0; i < N; i++) {
          Gamma_tilde_sum(k) += Gamma_tilde(i, k);
        }
      }

      if (IBP == 1) {

        uvec Gamma_tilde_order = stable_sort_index(Gamma_tilde_sum, "descend");

        if (any(Gamma_tilde_order != one_to_K)) {
          Gamma_tilde = Gamma_tilde.cols(Gamma_tilde_order);
          Gamma_tilde_sum = Gamma_tilde_sum.elem(Gamma_tilde_order);

          Tau = Tau.cols(Gamma_tilde_order);

          B = B.cols(Gamma_tilde_order);
          B_old = B_old.cols(Gamma_tilde_order);

          thetas = thetas.elem(Gamma_tilde_order);
          nus = nus.elem(Gamma_tilde_order);
        }
      }

      // Update X
      E_step_X(X, V_traces, ML, B, sigmas, Tau, Y);

      // Update B and sigmas
      if (ITER == 1 && l == 0) {
        mat B_temp = zeros<mat>(G, K);
        B = M_step_B(Y, B_temp, X, ML, sigmas, thetas, lambda1, lambda0);
        thetas = M_step_thetas_approx(B, a, b);
      } else {
        B = M_step_B(Y, B_old, X, ML, sigmas, thetas, lambda1, lambda0);
        thetas = M_step_thetas_approx(B, a, b);
      }

      // Update sigmas
      sigmas = M_step_sigmas(Y, B, X, eta, xi, sigma_min);

      // Update theta_tildes
      if (IBP == 1 && lambda1 != lambda0) {
        PQ = E_step_Q(nus);
        nus = M_step_nus_IBP(Gamma_tilde_sum, PQ, alpha, d, N);
        theta_tildes(0) = nus(0);
        for (int k = 1; k < K; k++) {
          theta_tildes(k) = theta_tildes(k - 1) * nus(k);
        }
      } else {
        theta_tildes = M_step_thetas(Gamma_tilde, a_tilde, b_tilde);
      }

      // Update Tau
      Tau = M_step_Tau(X, Gamma_tilde, V_traces, lambda1_tilde, lambda0_tilde);

      // Update the weights
      // we add this step to the EM algorithm to compute the weights for the E_Gamma_tilde
      if (ITER == iter_em_to_plot) {
        M_weights_res = M_step_Weights_multi(dis, Gamma_tilde, Tau, theta_tildes, lambda1_tilde,
                                       lambda0_tilde, weights, sample_weights, plot_conv, l + 1,
                                       stepsize_graddesc_logreg, niter_burnIn_ULA_SOUL, niter_ULA_SOUL,
                                       niter_graddesc_logreg, niter_expgrad_graddesc_logreg, dir_save_weight_grad,
                                       init_l2_reg_log_reg, l2_reg_log_reg, manual_set_stepsize_hyperparam_logreg,
                                       perc_max_stepsize_grad_desc, iter_stepsize_l2_reg_est);
      } else {
        M_weights_res = M_step_Weights_multi(dis, Gamma_tilde, Tau, theta_tildes, lambda1_tilde,
                                       lambda0_tilde, weights, sample_weights, false, l + 1,
                                       stepsize_graddesc_logreg, niter_burnIn_ULA_SOUL, niter_ULA_SOUL,
                                       niter_graddesc_logreg, niter_expgrad_graddesc_logreg, dir_save_weight_grad,
                                       init_l2_reg_log_reg, l2_reg_log_reg, manual_set_stepsize_hyperparam_logreg,
                                       perc_max_stepsize_grad_desc, iter_stepsize_l2_reg_est);
      }

      weights = as<mat>(M_weights_res["weights"]);
      if (!manual_set_stepsize_hyperparam_logreg) {
        sample_weights = as<mat>(M_weights_res["sample_weights"]);
        stepsize_graddesc_logreg = M_weights_res["stepsize"];
        l2_reg_log_reg = M_weights_res["lambda_reg_est"];
        iter_stepsize_l2_reg_est = M_weights_res["iter_stepsize_l2_reg_est"];
      }

       // remove zeros
      if ((l != 0 && lambda1 != lambda0) && ITER % 100 == 0) {
        vec B_zero = zeros<vec>(K);
        vec X_zero = zeros<vec>(K);
        for (int k = 0; k < K; k++) {
          for (int j = 0; j < G; j++) {
            if (B(j, k) == 0) {
              B_zero(k)++;
            }
          }
          for (int i = 0; i < N; i++) {
            if (Gamma_tilde(i, k) < 0.025) {
              X_zero(k)++;
            }
          }
        }

        uvec keep = find(B_zero < G-1 && X_zero < N-1);

        if (keep.n_elem < K && keep.n_elem > 0) {
          mat X_new = X.cols(keep);
          K = X_new.n_cols;
          mat Gamma_tilde_new = Gamma_tilde.cols(keep);
          mat Tau_new = Tau.cols(keep);
          mat B_new = B.cols(keep);
          mat B_old_new = B_old.cols(keep);

          Gamma_tilde_sum = Gamma_tilde_sum.elem(keep);
          thetas = thetas.elem(keep);
          theta_tildes = theta_tildes.elem(keep);
          nus = nus.elem(keep);
          uvec keep_weights = join_cols(uvec{0}, keep + 1);
          weights = weights.rows(keep_weights);
          sample_weights = sample_weights.rows(keep_weights);
          uvec new_K = linspace<uvec>(0, K - 1, K);
          one_to_K = one_to_K.elem(new_K);

          V_traces.set_size(N, K);
          ML.set_size(K, K);
          PQ.set_size(K, K);
          X.set_size(size(X_new));
          X = X_new;
          Tau.set_size(size(Tau_new));
          Tau = Tau_new;
          B.set_size(size(B_new));
          B = B_new;
          B_old.set_size(size(B_old_new));
          B_old = B_old_new;
          Gamma_tilde.set_size(size(Gamma_tilde_new));
          Gamma_tilde = Gamma_tilde_new;
        }

        if (keep.n_elem == 0) {
          K = 0;
          break;
        }

      }
      
      // remove zero components
      if (K == 0) {
        Rcout << "Number of biclusters is 0" << endl;
        break;
      }

      // Check convergence
      converged = check_convergence(B, B_old, EPSILON);

      if (converged == 1) {
        Rcout << "Iteration: " << ITER  << "......Converged." << endl;
        break;
      }

      B_old = B;

      if (ITER % 100 == 0) {
        Rcout << "Iteration: " << ITER << endl;

      }

      // Rescale X and B 
     if (lambda0 != lambda1) {
      rescale_X_B(X, B);
     }
      
    }

    // reset number of iterations for the estimation of l2 hyperparameter for log. reg. weights
    iter_stepsize_l2_reg_est = 1;

    Gamma_tilde_sum_all(l) = sum(Gamma_tilde_sum);
    if (l > 1 && update_lambda0_tilde == 1) {
      if (Gamma_tilde_sum_all(l) > Gamma_tilde_sum_all(l - 1)) {
        lambda0_tilde = lambda0_tildes(l - 1);
        update_lambda0_tilde = 0;
      }
    }

    if (K == 0) {
      break;
    }

  }

  List out(13);
  out["X"] = X;
  out["B"] = B;
  out["ML"] = ML;
  out["Tau"] = Tau;
  out["Gamma_tilde"] = Gamma_tilde;
  out["sigmas"] = sigmas;
  out["thetas"] = thetas;
  out["theta_tildes"] = theta_tildes;
  out["nus"] = nus;
  out["iter"] = NITERS;
  out["W"] = weights;
  out["lambda_l2_reg"] = l2_reg_log_reg;
  out["stepsize_logreg"] = stepsize_graddesc_logreg;

  return out;

}