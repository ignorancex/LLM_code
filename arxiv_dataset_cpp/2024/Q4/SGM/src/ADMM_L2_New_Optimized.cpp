#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;
using namespace Eigen;

typedef Eigen::Map<Eigen::MatrixXd> MapMat; 
typedef Eigen::Map<Eigen::VectorXd> MapVec;

// [[Rcpp::depends(RcppEigen)]]

/*##### L-minimization step 
 L.Min<-function(S, theta0,theta1 = 1, v0, Q, Dx.inv, Z, W, U, V,R,  rho, lambda, model= "LN"){
## S: p by p, sufficient statistic for covariance 
## theta0, theta1: theta parameters of the model; 2024/5/512: Notes: for all models theta1 should be fixed at 1
## v0: square-root of the degree vector of the graph; For Laplacian model: v0 should be the vector of 1 
## Q,D eigen decomposition of C = theta1^2 S + rho I + rho/2 v0 t(v0)
## Dx.inv= inverse of the diagonal of I outer D + D outer I: p^2 by 1 vector  
## Z, W: p by p matrices:  L- Z=0, L+W =0  
## U,V,R: dual variables, p by p (U, W) and p by 1 (R)  
## rho: ADMM scaled Lagrange parameter 
## lambda: l1 (trace) penalty parameter
## model: either "LN" -- normalized Laplacian or "LN.noloop" -- normalized Laplacian  with no self-loop or "L" -- Laplacian
## return: updated L, satisfy L=t(L)
 */



Rcpp::List L_Pre(const Eigen::MatrixXd& S, double theta1, Eigen::VectorXd& v0, double rho){
  int p = S.rows(); 
  Eigen::MatrixXd C = theta1*theta1*S  + rho*Eigen::MatrixXd::Identity(p,p) + (rho/2)*v0*v0.transpose(); 
  
  // Eigendecomposition: exploiting symmetric structure
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(C);
  Eigen::MatrixXd Q = eigensolver.eigenvectors(); // Eigenvectors
  Eigen::VectorXd d = eigensolver.eigenvalues(); // Eigenvalues 
  
  // Create Dx_inv
  Eigen::MatrixXd Dx_inv(p,p); 
  
  for (int i=0; i<p; ++i){
    for (int j=i; j<p; ++j){
      Dx_inv(i, j) = 1/(d(i) + d(j)); 
      Dx_inv(j, i) = Dx_inv(i,j); // Jed: Modified to incorporate symmetry
    }
  }
  
  return List::create(Named("Q") = Q, Named("Dx_inv") = Dx_inv);
}

// 5/18/24: Updated to a void function
void L_Min(const Eigen::MatrixXd& S, double theta0, double theta1, Eigen::VectorXd& v0, Eigen::MatrixXd& Q, Eigen::MatrixXd& Dx_inv, Eigen::MatrixXd& Z, Eigen::MatrixXd& W, Eigen::MatrixXd& U, Eigen::MatrixXd& V, Eigen::VectorXd& R, double rho, double lambda, std::string model, Eigen::MatrixXd& L_Mat){
  
  int p = S.rows(); // Extract Number of rows
  Eigen::MatrixXd Jp_t(p,p); // Initialize Jp.t Matrix 
  
  if (model == "LN" || model == "LN.noloop"){
    Jp_t = Eigen::MatrixXd::Ones(p, p) - Eigen::MatrixXd::Identity(p,p); // 2024/5/12: this penalty term matrix works for both general LN case and the special no self-loop LN case
  }
  else {
    Jp_t = -Eigen::MatrixXd::Identity(p,p); 
  }
  
  Eigen::MatrixXd Dk = 2*theta0*theta1*S-lambda*Jp_t-rho*(Z-U)+rho*(W+V)+rho*(R*v0.transpose() + v0*R.transpose())/2; 
  
  Eigen::MatrixXd Dk_tilde(p,p);  //
  Dk_tilde.triangularView<Eigen::Lower>() = Q.transpose()*Dk* Q; /// use symmetry to speed up
  Dk_tilde=Dk_tilde.selfadjointView<Eigen::Lower>();//
  
  Eigen::MatrixXd L_tilde = -Dx_inv.array()*Dk_tilde.array(); 
  L_Mat.triangularView<Eigen::Lower>() = Q * L_tilde * Q.transpose(); // Updated 5/18/24
  L_Mat = L_Mat.selfadjointView<Eigen::Lower>(); // Updated 5/18/24
}

// 5/18/24: Updated to a void function
void Z_Min(double theta0, double theta1, Eigen::MatrixXd& L, Eigen::MatrixXd& U, double rho, double eps_thre, Eigen::MatrixXd& Z_Mat){
  
  // Eigendecomposition via symmetric structure
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(L+U);
  Eigen::MatrixXd P = eigensolver.eigenvectors(); // vectors
  Eigen::VectorXd lambda = eigensolver.eigenvalues(); // values 
  
  // Define Lambda_tilde 
  int p = L.rows(); 
  Eigen::VectorXd lambda_tilde(p); 
  double a = rho*theta1; 
  
  for (int i=0; i<p; ++i){
    double b_i = rho*(theta0-theta1*lambda(i));
    double c_i = -rho*theta0*lambda(i)-2*theta1; 
    double temp = (-b_i+std::sqrt(b_i*b_i-4*a*c_i))/(2*a); 
    if (temp > eps_thre){
      lambda_tilde(i) = temp;
    }
    else {
      lambda_tilde(i) = 0; 
    }
    
  }
  
  Z_Mat.triangularView<Eigen::Lower>() = P * lambda_tilde.asDiagonal()* P.transpose(); 
  Z_Mat = Z_Mat.selfadjointView<Eigen::Lower>();
}

// 5/18/24: Updated to a void function
void W_Min(Eigen::MatrixXd& L, Eigen::MatrixXd& V, std::string model, double eps_thre, Eigen::MatrixXd& W_Mat){
  int p = L.rows(); 
  W_Mat = (-1)*(L+V); 
  Eigen::VectorXd temp = W_Mat.diagonal();
  W_Mat = (W_Mat.array() <= eps_thre).select(0, W_Mat); 
  if (model == "LN.noloop"){
    W_Mat.diagonal() = std::min(-eps_thre, temp.mean())*Eigen::VectorXd::Ones(p);
  }
  else {
    W_Mat.diagonal() = temp; 
  }
  W_Mat.triangularView<Eigen::Lower>() = (W_Mat+W_Mat.transpose())/2; 
  W_Mat = W_Mat.selfadjointView<Eigen::Lower>();
}

// 5/18/24: Updated to a void function
void W_Min_Zero(Eigen::MatrixXd& L, Eigen::MatrixXd& V, const Eigen::MatrixXd A, std::string model, double eps_thre, Eigen::MatrixXd& W_Mat){
  int p = L.rows(); 
  W_Mat = (-1)*(L+V); 
  Eigen::VectorXd temp = W_Mat.diagonal();
  W_Mat = (W_Mat.array() <= eps_thre).select(0, W_Mat); 
  W_Mat = W_Mat.array()*A.array(); 
  if (model == "LN.noloop"){
    W_Mat.diagonal() = std::min(-eps_thre, temp.mean())*Eigen::VectorXd::Ones(p);
  }
  else {
    W_Mat.diagonal() = temp; 
  }
  W_Mat = (W_Mat+W_Mat.transpose())/2; 
  W_Mat = W_Mat.selfadjointView<Eigen::Lower>();
}

// 5/18/24: Updated to a void function
void dual_update(Eigen::MatrixXd& U, Eigen::MatrixXd& V, Eigen::MatrixXd& L_u,  Eigen::MatrixXd& Z_u, Eigen::MatrixXd& W_u, Eigen::VectorXd& R, Eigen::VectorXd& v0, Eigen::MatrixXd& U_u, Eigen::MatrixXd& V_u, Eigen::VectorXd& R_u){
  U_u = U + (L_u - Z_u); 
  V_u = V + (L_u + W_u);
  R_u = R + L_u*v0; 
}

double Resid_Primal(Eigen::MatrixXd& L, Eigen::MatrixXd& Z, Eigen::MatrixXd& W, Eigen::VectorXd& v0){
  
  Eigen::VectorXd L_x_v0 = L*v0; 
  double resi = (L-Z).squaredNorm() + (L+W).squaredNorm() + L_x_v0.squaredNorm(); 
  resi = std::sqrt(resi); 
  
  return resi; 
}

double Resid_Dual(Eigen::MatrixXd& Z, Eigen::MatrixXd& Z_u, Eigen::MatrixXd& W, Eigen::MatrixXd& W_u, double rho){
  double resi = (Z-Z_u+W_u-W).squaredNorm();
  resi = rho*std::sqrt(resi); 
  return resi; 
}

double Eps_Primal(Eigen::MatrixXd& L, Eigen::MatrixXd& Z, Eigen::MatrixXd& W, Eigen::VectorXd& v0, double eps_abs, double eps_rel){
  
  int p = L.rows(); 
  Eigen::VectorXd L_x_v0 = L * v0; 
  double temp1 = std::sqrt(2*L.squaredNorm() + L_x_v0.squaredNorm());
  double temp2 = std::sqrt(Z.squaredNorm() + W.squaredNorm());
  double eps = std::sqrt(p*(2*p+1))*eps_abs + eps_rel*std::max(temp1, temp2); 
  return eps; 
}

double Eps_Dual(Eigen::MatrixXd& U, Eigen::MatrixXd& V, Eigen::VectorXd& R, Eigen::VectorXd& v0, double rho, double eps_abs, double eps_rel){
  int p = U.rows(); 
  Eigen::MatrixXd R_v0t = R * v0.transpose(); 
  double temp = std::sqrt((U + V + R_v0t).squaredNorm());
  double eps = p*eps_abs + eps_rel*rho*temp; 
  
  return eps; 
}

bool Primal_Cri(Eigen::MatrixXd& L_u, Eigen::MatrixXd& Z_u, Eigen::MatrixXd& W_u, Eigen::VectorXd& v0, double eps_abs, double eps_rel, bool verbose){
  
  double res_pri = Resid_Primal(L_u, Z_u, W_u, v0);
  double eps_pri = Eps_Primal(L_u, Z_u, W_u, v0, eps_abs, eps_rel); 
  
  if (verbose){
    std::cout << "primal: " + std::to_string(res_pri) + " " + std::to_string(eps_pri) << std::endl;
  }
  
  return res_pri <= eps_pri; 
}


bool Dual_Cri(Eigen::MatrixXd& Z, Eigen::MatrixXd& Z_u, Eigen::MatrixXd& W, Eigen::MatrixXd& W_u, Eigen::MatrixXd& U_u, Eigen::MatrixXd& V_u, Eigen::VectorXd& R_u, Eigen::VectorXd& v0, double rho, double eps_abs, double eps_rel, bool verbose){
  
  double res_dual = Resid_Dual(Z, Z_u, W, W_u, rho); 
  double eps_dual = Eps_Dual(U_u, V_u, R_u, v0, rho, eps_abs, eps_rel); 
  
  if (verbose){
    std::cout << "dual: " + std::to_string(res_dual) + " " + std::to_string(eps_dual) << std::endl;
  }
  
  return res_dual <= eps_dual; 
}


//' ADMM algorithm for estimating (normalized) graph Laplacian given v0 and theta0
//' 
//' @param s an estimated covariance matrix, such as the sample covariance matrix.
//' @param theta0 A given estimate for the graph filter parameter theta0.
//' @param v A given degree vector.
//' @param rho ADMM parameter. 
//' @param lambda Tuning parameter.
//' @param model A character specifying which type of Laplacian to use.
//' @param Z_ini An initial value for the ADMM algorithm
//' @param W_ini An initial value for the ADMM algorithm
//' @param eps_thre A small positive number
//' @param eps_abs ADMM stopping criterion
//' @param eps_rel ADMM stopping criterion
//' @param max_iter Maximum number of iterations to run
//' @param verbose Trace of ADMM algorithm
//' 
//' @returns
//' * `L`: A p by p matrix.
//' * `Z`: A p by p matrix.
//' * `W`: A p by p matrix,
//' * `theta0`: The estimate for `theta0` used.
//' * `conv`: A boolean indicating convergence (TRUE if converged).
//' @export 
// [[Rcpp::export]]
List ADMM_L2(const Rcpp::NumericMatrix& s, double theta0, Rcpp::NumericVector& v, double rho, double lambda, std::string model, const Rcpp::NumericMatrix& Z_ini, const Rcpp::NumericMatrix& W_ini, double eps_thre, double eps_abs, double eps_rel, int max_iter, bool verbose){
  
  //// Map to Eigen types: 
  const MapMat S = Rcpp::as<MapMat>(s);
  MapMat z_ini = Rcpp::as<MapMat>(Z_ini);
  MapMat w_ini = Rcpp::as<MapMat>(W_ini);  
  MapVec V0 = Rcpp::as<MapVec>(v); 
  Eigen::VectorXd v0 = V0;
  
  
  int p = S.rows();
  
  // Initialization function should be inserted here. For now, just set Z_ini, W_ini = 0 Matrix
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(p,p); 
  Eigen::MatrixXd Z = z_ini; 
  Eigen::MatrixXd W = w_ini; 
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(p,p); 
  Eigen::MatrixXd V = U;
  Eigen::VectorXd R = Eigen::VectorXd::Zero(p); 
  double theta1 = 1; 
  
  // For regular laplacian, must set v0 to vector of ones
  if (model == "L"){
    v0 = Eigen::VectorXd::Ones(p); 
  }
  
  // Prepare for L_Min
  List temp = L_Pre(S, theta1, v0, rho); 
  Eigen::MatrixXd Q = temp["Q"]; 
  Eigen::MatrixXd Dx_inv = temp["Dx_inv"];
  
  // INITIALIZE OBJECTS USED IN WHILE LOOP
  Eigen::MatrixXd L_u = L;
  Eigen::MatrixXd Z_u = Z;
  Eigen::MatrixXd W_u = W;
  Eigen::MatrixXd U_u = U; 
  Eigen::MatrixXd V_u = V;
  Eigen::VectorXd R_u = R; 
  
  // ADMM ALGORITHM
  bool conv = false; 
  int iter = 0; 
  while (!conv && iter <= max_iter){
    if (verbose){
      std::cout << "ADMM Step: " + std::to_string(iter) << std::endl; 
    }
    
    L_Min(S, theta0, theta1, v0, Q, Dx_inv, Z, W, U, V, R, rho, lambda, model, L_u); 
    Z_Min(theta0, theta1, L_u, U, rho, eps_thre, Z_u); 
    W_Min(L_u, V, model, eps_thre, W_u); 
    
    dual_update(U, V, L_u, Z_u, W_u, R, v0, U_u, V_u, R_u); 
    
    bool primal_conv = Primal_Cri(L_u, Z_u, W_u, v0, eps_abs, eps_rel, verbose); 
    
    if (primal_conv){
      bool dual_conv = Dual_Cri(Z, Z_u, W, W_u, U_u, V_u, R_u, v0, rho, eps_abs, eps_rel, verbose); 
      
      if (dual_conv){
        conv = true; 
      }
    }
    
    L = L_u; 
    Z = Z_u; 
    W = W_u; 
    U = U_u; 
    V = V_u; 
    R = R_u; 
    iter++; 
  }
  
  // Update theta1 and L
  theta1 = L.diagonal().maxCoeff();
  L = L.array()/theta1; 
  
  return List::create(Named("L") = L, Named("Z") = Z, Named("W") = W, Named("theta0") = theta0, Named("theta1") = theta1, Named("conv") = conv);
  
}

//' ADMM Algorithm for estimating (normalized) Graph Laplacian given v0, theta0, and a zero-pattern.
//' 
//' @param SS an estimated covariance matrix, such as the sample covariance matrix.
//' @param theta0 A given estimate for the graph filter parameter theta0.
//' @param v A given degree vector.
//' @param rho ADMM parameter. 
//' @param AA A p by p matrix encoding the zero pattern for the (normalized) graph Laplacian.
//' @param lambda Tuning parameter.
//' @param model A character specifying which type of Laplacian to use.
//' @param Z_ini An initial value for the ADMM algorithm
//' @param W_ini An initial value for the ADMM algorithm
//' @param eps_thre A small positive number
//' @param eps_abs ADMM stopping criterion
//' @param eps_rel ADMM stopping criterion
//' @param max_iter Maximum number of iterations to run
//' @param verbose Trace of ADMM algorithm
//' 
//' @returns
//' * `L`: A p by p matrix.
//' * `Z`: A p by p matrix.
//' * `W`: A p by p matrix,
//' * `theta0`: The estimate for `theta0` used.
//' * `conv`: A boolean indicating convergence (TRUE if converged).
//' @export
// [[Rcpp::export]]
List ADMM_L2_Zero(const Rcpp::NumericMatrix& SS, double theta0, Rcpp::NumericVector& v, double rho, const Rcpp::NumericMatrix AA, std::string model, const Rcpp::NumericMatrix& Z_ini, const Rcpp::NumericMatrix& W_ini, double eps_thre, double eps_abs, double eps_rel, int max_iter, bool verbose){
  
  //// Map to Eigen types: 
  const MapMat S = Rcpp::as<MapMat>(SS);
  const MapMat A = Rcpp::as<MapMat>(AA); 
  MapMat z_ini = Rcpp::as<MapMat>(Z_ini);
  MapMat w_ini = Rcpp::as<MapMat>(W_ini);  
  MapVec V0 = Rcpp::as<MapVec>(v); 
  Eigen::VectorXd v0 = V0;
  
  
  int p = S.rows();
  
  // Initialization function should be inserted here. For now, just set Z_ini, W_ini = 0 Matrix
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(p,p); 
  Eigen::MatrixXd Z = z_ini; 
  Eigen::MatrixXd W = w_ini.array()*A.array(); 
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(p,p); 
  Eigen::MatrixXd V = U;
  Eigen::VectorXd R = Eigen::VectorXd::Zero(p); 
  double theta1 = 1;
  
  // For regular laplacian, must set v0 to vector of ones
  if (model == "L"){
    v0 = Eigen::VectorXd::Ones(p); 
  }
  
  // Prepare for L_Min
  List temp = L_Pre(S, theta1, v0, rho); 
  Eigen::MatrixXd Q = temp["Q"]; 
  Eigen::MatrixXd Dx_inv = temp["Dx_inv"];
  
  // INITIALIZE OBJECTS USED IN WHILE LOOP
  Eigen::MatrixXd L_u = L;
  Eigen::MatrixXd Z_u = Z;
  Eigen::MatrixXd W_u = W;
  Eigen::MatrixXd U_u = U; 
  Eigen::MatrixXd V_u = V;
  Eigen::VectorXd R_u = R; 
  
  // ADMM ALGORITHM
  bool conv = false; 
  int iter = 0; 
  while (!conv && iter <= max_iter){
    if (verbose){
      std::cout << "ADMM Step: " + std::to_string(iter) << std::endl; 
    }
    
    L_Min(S, theta0, theta1, v0, Q, Dx_inv, Z, W, U, V, R, rho, 0, model, L_u); 
    Z_Min(theta0, theta1, L_u, U, rho, eps_thre, Z_u); 
    W_Min_Zero(L_u, V, A, model, eps_thre, W_u); 
    
    dual_update(U, V, L_u, Z_u, W_u, R, v0, U_u, V_u, R_u); 
    
    bool primal_conv = Primal_Cri(L_u, Z_u, W_u, v0, eps_abs, eps_rel, verbose); 
    
    if (primal_conv){
      bool dual_conv = Dual_Cri(Z, Z_u, W, W_u, U_u, V_u, R_u, v0, rho, eps_abs, eps_rel, verbose); 
      
      if (dual_conv){
        conv = true; 
      }
    }
    
    L = L_u; 
    Z = Z_u; 
    W = W_u; 
    U = U_u; 
    V = V_u; 
    R = R_u; 
    iter++; 
  }
  
  // Update theta1 and L
  theta1 = L.diagonal().maxCoeff();
  L = L.array()/theta1; 
  
  return List::create(Named("L") = L, Named("Z") = Z, Named("W") = W, Named("theta0") = theta0, Named("theta1") = theta1, Named("conv") = conv);
  
}




//' Extension of `ADMM_L2` to handle a sequence of lambda values.
//' 
//' @param S A p by p estimate of the covariance matrix, such as the sample covariance matrix.
//' @param theta0 Estimate for graph-filter parameter `theta0`.
//' @param v0 A given degree vector.
//' @param Rho A vector of ADMM parameters
//' @param Lambda A vector of tuning parameters
//' @param model To specify which model to use.
//' @param ini Whether to initialize ADMM with given values (TRUE) or not (FALSE). 
//' @param eps_thre A small positive number
//' @param eps_abs ADMM stopping criterion
//' @param eps_rel ADMM stopping criterion
//' @param max_iter Maximum number of iterations to run
//' @param verbose Trace of ADMM algorithm
//' 
//' @returns A list containing the corresponding output of `ADMM_L2` applied to each value of `Lambda`
//' @export
// [[Rcpp::export]]
List ADMM_L2_seq(const Rcpp::NumericMatrix& S, double theta0, Rcpp::NumericVector& v0, Rcpp::NumericVector& Rho, Rcpp::NumericVector& Lambda, std::string model, bool ini, double eps_thre, double eps_abs, double eps_rel, int max_iter, bool verbose){
  
  // Map to Eigen types
  //// Map to Eigen types: 
  MapVec rho = Rcpp::as<MapVec>(Rho);
  MapVec lambda = Rcpp::as<MapVec>(Lambda); 
  
  
  
  
  // Initialize variables
  int p = S.rows();
  int n = lambda.size();
  List resList(n);
  List temp(4);
  Eigen::MatrixXd Z_ini = Eigen::MatrixXd::Zero(p,p);
  Eigen::MatrixXd W_ini = Z_ini;
  
  Rcpp::NumericMatrix Z = Rcpp::wrap(Z_ini);
  Rcpp::NumericMatrix W = Rcpp::wrap(W_ini); 
  
  
  for (int j=0; j < n; ++j){
    
    // Extract current lambda and rho
    double lambda_j = lambda(j);
    double rho_j = rho(j);
    
    // Decreasing lambda method
    if (j >= 1 && ini && temp["conv"]){
      Z_ini = temp["Z"];
      W_ini = temp["W"];
      
      Z = Rcpp::wrap((Z_ini + Z_ini.transpose())/2);
      W_ini = (W_ini + W_ini.transpose())/2;
      // 5/12/24: W_ini.diagonal() = (-1)*Eigen::VectorXd::Ones(p);
      W = Rcpp::wrap(W_ini);
      
    }
    try {temp = ADMM_L2(S, theta0, v0, rho_j, lambda_j, model, Z, W, eps_thre, eps_abs, eps_rel, max_iter, verbose);}
    catch(std::runtime_error e){
      temp = List::create(Named("L") = NULL, Named("Z") = NULL, Named("W") = NULL, Named("conv") = false);
    }
    
    resList[j] = temp;
  }
  
  return resList;
}
