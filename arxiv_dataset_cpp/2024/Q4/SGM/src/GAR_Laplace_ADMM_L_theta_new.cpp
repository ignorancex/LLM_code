#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;
using namespace Eigen;
// [[Rcpp::depends(RcppEigen)]]


typedef Eigen::Map<Eigen::MatrixXd> MapMat; //Eigen templated class map: to have the contents of an R matrix mapped to the contents of the object from the Eigen class
typedef Eigen::Map<Eigen::VectorXd> MapVec;


// L AND THETA MINIMIZATION STEP:

////// L AND THETA PREPARATION FUNCTION:
////// S: pxp Covariance Estimate
////// v0: square-root of the degree vector of the graph; For Laplacian model: v0 should be the vector of 1 
////// Rho: ADMM Scaled Lagrangian Parameter (Scalar)
////// Returns: Q: p by p; QT: p by p;  Dx.inv, Dx.inv.Sx: p^2 by 1; dstar: scalar 

List L_theta_pre(const Eigen::MatrixXd S, Eigen::VectorXd v0, double rho){
  int p = S.rows(); // Extract Dimensions of S
  
  // Create C Matrix
  Eigen::MatrixXd v0_x_v0t = v0 * v0.transpose();
  Eigen::MatrixXd C = S + Eigen::MatrixXd::Identity(p, p)*rho + (rho/2)*(v0_x_v0t); // Added v0
  
  // Diagonalize C Matrix (exploiting symmetric structure)
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(C); 
  Eigen::MatrixXd Q = eigensolver.eigenvectors(); // Returns Q as in: QVQ' 
  Eigen::VectorXd d = eigensolver.eigenvalues(); // Returns a vector of Eigenvalues
  
  Eigen::MatrixXd QT =  Q.transpose(); 
  Eigen::MatrixXd Sx(p,p);
  Sx.triangularView<Eigen::Lower>() = 2.0 * QT * S * Q; // Calculate the Sx Matrix: only calculate the lower half 
  
  // Next, we calculate Dx and Dx.Inverse and Sx
  Eigen::VectorXd Dx_inv(p * p); 
  Eigen::VectorXd Dx_inv_Sx(p * p); 
  double temp = 0; 
  
  // The line below calculates Dx_inv in vector format
  for (int i = 0; i < p; ++i) {
    for (int j = i; j < p; ++j) {///use symmetry: only need to record half of the vectors  
      Dx_inv(i * p + j) = 1.0/(d[i] + d[j]); // Calculate Dx_inv (p2 x p2)
      Dx_inv_Sx(i * p + j) = Dx_inv(i * p + j)*Sx(j,i); // Calculate element-wise product of Dx_inv and (vectorized) Sx Matrix 
      temp = temp + ((i==j) ? 1.0 : 2.0)*Dx_inv_Sx(i * p + j)*Sx(j,i); // Calculates the dot product of (vectorized) Sx and Dx_inv_Sx
    }
  }
  
  // Delta_x
  double deltax = 2.0*S.trace() + rho;
  
  // Delta star = deltax - Sx_vec.dot(Dx_inv_Sx);
  double dstar = deltax - temp;
  
  // Return List 
  return List::create(
    Named("Q") = Q,
    Named("QT") = QT,
    Named("Dx.inv") = Dx_inv,
    Named("Dx.inv.Sx") = Dx_inv_Sx, 
    Named("dstar") = dstar
  );
}

/////// L AND THETA MINIMIZATION FUNCTION:
/////// Q: Eigendecomposition of C 
/////// Dx_inv= inverse of the diagonal of I outer D + D outer I: p^2 by 1 vector  
/////// Dx.inv.Sx: Dx.inv%*%Sx: p^2 by 1  
/////// dstar: deltax-sum(Sx^2*Dx): scalar 
/////// Z, W: p by p matrices for constraints:  L- Z=0, L+W =0; phi: scalar for constraint: theta-phi=0 
/////// U,V,R,t: dual variables; p by p (U, V); p by 1 vector (R); scalar (t) 
/////// rho: ADMM scaled Lagrange parameter; positive scalar
/////// v0: square-root of the degree vector of the graph; For Laplacian model: v0 should be the vector of 1 
/////// model: either "LN" -- normalized Laplacian or "LN.noloop" -- normalized Laplacian  with no self-loop or "L" -- Laplacian
/////// lambda: l1 (trace) penalty parameter: positive scalar 
/////// return/modified: updated L: p by p in L_mat,  satisfy L=t(L); updated theta: scalar (not necessarily positive);

double  L_theta_min(const Eigen::MatrixXd Q, const Eigen::MatrixXd QT, const Eigen::VectorXd Dx_inv, const Eigen::VectorXd Dx_inv_Sx,
                    double dstar, const Eigen::MatrixXd Z, const Eigen::MatrixXd W, const Eigen::MatrixXd U,
                    const Eigen::MatrixXd V, const Eigen::VectorXd R, double phi, double t, double rho, double lambda, Eigen::MatrixXd& L_mat,
                    Eigen::VectorXd v0, std::string model){
  
  // Extract number of rows
  int p = Q.rows(); 
  Eigen::MatrixXd Jp_t(p,p);
  // Determine Calculation by Model
  if (model == "LN" || model == "LN.noloop"){
    Jp_t = Eigen::MatrixXd::Ones(p,p) - Eigen::MatrixXd::Identity(p,p); 
  }
  else {
    Jp_t = -Eigen::MatrixXd::Identity(p,p); 
  }
  Eigen::MatrixXd R_m = R * v0.transpose(); // R ^ v0' -- pxp matrix
  
  // Calculate temp matrix
  Eigen::MatrixXd temp = Jp_t * (-lambda) - rho * (Z - U) + rho * (W + V) + (rho / 2.0) * (R_m + R_m.transpose());
  
  // Compute ep 
  double ep = rho * (phi - t);
  
  // Compute E (matrix): only calculate the lower half 
  Eigen::MatrixXd E(p,p);
  E.triangularView<Eigen::Lower>() = QT * temp * Q; // O(n^3) *************
  
  // Use E to calculate L_mat and terms required for theta_new
  
  double temp_dxinv_E = 0; // Initialize temp_dx_inv_E
  for (int i = 0; i < p; ++i){
    for (int j = i; j<p;++j){
      temp_dxinv_E = temp_dxinv_E + ((i==j) ? 1.0 : 2.0)*Dx_inv_Sx(i*p+j)*E(j,i); // Dot product
    }
  }
  
  // Calculate theta new
  double theta_new = temp_dxinv_E/dstar + ep/dstar;
  
  // Update L Mat
  for (int i=0; i<p; ++i){
    for(int j=i; j<p; ++j){
      L_mat(j,i) = (-1.0)*Dx_inv(i*p+j)*E(j, i)-Dx_inv_Sx(i*p+j)*theta_new;
      L_mat(i,j) = L_mat(j,i); 
    }
  }
  
  /*Eigen::VectorXd L_vec_term1(p*p); // Initialize
  Eigen::VectorXd L_vec_term3(p*p); // Initialize 
  
  for (int i=0; i<p;++i){
    for (int j=i; j<p; ++j){///use symmetry 
      temp_dxinv_E = temp_dxinv_E + ((i==j) ? 1.0 : 2.0)*Dx_inv_Sx[i * p + j]*E(j,i); // Dot product
      L_vec_term1(i*p+j) = (-1.0)*Dx_inv(i*p+j)*E(j,i); 
      L_vec_term3(i*p+j) = Dx_inv_Sx(i*p+j)*ep/dstar;
    }
  }
  
  // Calculate theta.new:
  double theta_new = temp_dxinv_E / dstar + ep / dstar;
  
  // Update L_mat
  double L_vec_term2_i = 0;
  for (int i = 0; i < p; ++i){
    for(int j = i; j < p; ++j){ /// use L_mat symmetry 
      L_vec_term2_i = Dx_inv_Sx(i*p+j)*temp_dxinv_E/dstar; 
      L_mat(j,i) = L_vec_term1(i*p+j)-L_vec_term2_i-L_vec_term3(i*p+j);
      L_mat(i,j) = L_mat(j,i); 
    }
  }*/
  
  // Calculate L_new: only calculate lower half, then make symnmetric 
  L_mat.triangularView<Eigen::Lower>() = Q * L_mat * QT;  // O(n^3) *************
  L_mat = L_mat.selfadjointView<Eigen::Lower>();
  
  return theta_new; 
}

/////// Phi Update
double phi_update(const Eigen::VectorXd Lambda_star, const Eigen::VectorXd Lambda, double theta, double t, double rho, double eps_thre){
  int p = Lambda.size();
  double phi_new = std::max(((Lambda_star - Lambda).sum() + theta + t) / (p + 1.0), eps_thre);
  return phi_new;
}


/////// Lambda.Star update step: inner iteration
////// Lambda: px1 vector of Eigenvalues of L+U
////// Phi: Scalar (current step)
////// rho: ADMM scaled lagrangian parameter
////// return/modified: Updated Lambda.star: px1
void Lambda_update(const Eigen::VectorXd Lambda, double phi, double rho, Eigen::VectorXd& Lambda_new) {
  
  int p = Lambda.size();
  
  for (int i = 0; i < p; ++i) {
    double phi_p_Lambda_i = phi + Lambda[i];
    double temp = (rho * phi_p_Lambda_i + std::sqrt(rho * rho * phi_p_Lambda_i * phi_p_Lambda_i + 8 * rho)) / (2 * rho);
    Lambda_new[i] = (temp <= phi) ? phi : temp;
  }
}

////// Function to Check convergence of the Z/phi updates
bool Z_phi_Conv(const Eigen::VectorXd Lambda_c, const Eigen::VectorXd Lambda_new, double phi_c, double phi_new, double conv_abs, double conv_rel, bool verbose){
  // Extract the size of Lambda
  int p = Lambda_c.size();
  
  //// BENCHMARKS SHOWED THAT THIS WAS FASTER THAN FOR LOOP
  double resi = std::sqrt( (phi_new-phi_c) * (phi_new-phi_c) + (Lambda_new-Lambda_c).squaredNorm());
  double temp1 = std::sqrt(phi_new * phi_new + Lambda_new.squaredNorm());
  double temp2 = std::sqrt(phi_c * phi_c + Lambda_c.squaredNorm());
  double eps = std::sqrt(p+1.0) * conv_abs + conv_rel*std::max(temp1, temp2);
  
  if (verbose){
    Rcpp::Rcout << "-- Z.phi conv: " + std::to_string(resi) + " " + std::to_string(eps) << std::endl;  
  }
  return resi <= eps;
}

////// Z/Phi Update
////// L, U: p x p matrices 
////// phi_ini: kth (outer) step. phi: scalar; used as initial value for (k+1)th (outer)step Z/phi update iteration
////// theta, t: scalars 
////// rho: ADMM scaled Lagrange Parameter; Positive Scalar
////// eps_thre: Lower bound for phi; positive small scalar
////// max.step; conv_abs, conv_rel: convergence criteria
////// return/modified: Updated Z: p x p matrix; updated phi (scalar); convergence status (boolean)

List Z_phi_Min(const Eigen::MatrixXd L, const Eigen::MatrixXd U, double phi_ini, double theta, double t, double rho,
               double eps_thre, int max_step, double conv_abs, double conv_rel, bool verbose, Eigen::MatrixXd& Z_new){
  
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(L + U); //// O(n^3)**********************
  Eigen::MatrixXd P = eigensolver.eigenvectors(); // Fixed
  Eigen::VectorXd Lambda = eigensolver.eigenvalues(); // Fixed
  
  int p = L.rows();
  
  // Initial Values
  double phi_c = phi_ini; // To be updated
  Eigen::VectorXd Lambda_c(p);
  Lambda_update(Lambda, phi_c, rho, Lambda_c); // Lambda.star: to be updated
  
  int step = 1;
  bool conv_stat = false; 
  
  // Initialize phi_new and Lambda_new
  double phi_new = phi_ini; 
  Eigen::VectorXd Lambda_new(p);
  
  // Inner Iteration
  while(step <= max_step && !conv_stat){
    phi_new = phi_update(Lambda_c, Lambda, theta, t, rho, eps_thre); 
    Lambda_update(Lambda, phi_new, rho, Lambda_new);  
    
    // Check for convergence
    conv_stat = Z_phi_Conv(Lambda_c, Lambda_new, phi_c, phi_new, conv_abs, conv_rel, verbose);
    
    // Increment Step
    step++;
    phi_c = phi_new; // Update phi_c
    Lambda_c = Lambda_new; // Update to Lambda_new 
  }
  
  Eigen::VectorXd Lambda_tilde = Lambda_c - phi_c*Eigen::VectorXd::Ones(p);
  Z_new = P * Lambda_tilde.asDiagonal() * P.transpose(); //here seems more efficient than using the triangularView
  ///Z_new.triangularView<Eigen::Lower>() = P * Lambda_tilde.asDiagonal() * P.transpose(); ////Caluclate the lower half: O(n*3)**************
  ///Z_new =Z_new.selfadjointView<Eigen::Lower>(); //make symmetry
  
  return List::create(Named("phi") = phi_c, Named("status")=conv_stat);
  
}

///////////////////////////////
///////// W MINIMIZATION STEP 
////// L, V: p x p matrices
////// eps.thre: lower bound for zero; positive small scalar
////// return/modified: Updatesd W with non-negative (positive) off-diagonals
////// BENCHMARKS SHOWED THAT THIS WAS FASTER THAN A FOR LOOP
////// Added model parameter
void W_Min_S(const Eigen::MatrixXd L, const Eigen::MatrixXd V, std::string model, double eps_thre, Eigen::MatrixXd& W){
  
  int p = L.rows(); 
  W = -(L+V); 
  Eigen::VectorXd temp=W.diagonal();  // define temp as a vector = diag(W)
  
  // Set off-diagonal elements less than eps_thre to be 0
  W = (W.array() <= eps_thre).select(0, W);
  
  if (model == "LN.noloop"){
    W.diagonal() = std::min(-eps_thre, temp.mean())*Eigen::VectorXd::Ones(p);
  }
  else {
  // Restore diagonal elements of W
  W.diagonal() = temp; /// directly assign temp  to W
  }
  
  // Ensure symmetry
  W = (W + W.transpose()) / 2.0; 
}

void W_Min_S_Zero(const Eigen::MatrixXd L, const Eigen::MatrixXd V, const Eigen::MatrixXd A, std::string model, double eps_thre, Eigen::MatrixXd& W){
  
  int p = L.rows(); 
  W = -(L+V); 
  Eigen::VectorXd temp=W.diagonal();  // define temp as a vector = diag(W)
  
  // Set off-diagonal elements less than eps_thre to be 0
  W = (W.array() <= eps_thre).select(0, W);
  W = W.array()*A.array(); 
  
  if (model == "LN.noloop"){
    W.diagonal() = std::min(-eps_thre, temp.mean())*Eigen::VectorXd::Ones(p);
  }
  else {
    // Restore diagonal elements of W
    W.diagonal() = temp; /// directly assign temp  to W
  }
  
  // Ensure symmetry
  W = (W + W.transpose()) / 2.0; 
}

///////////////////////
/////// DUAL UPDATES //
/////////////////////// 
//// U.u, V.u, R.u, t.u: current dual variables; p x p symmetric (U.u, V.u); p x 1 (R.u); Scalar (t.u)
//// L,Z,W: p x p symmetric
//// theta, pi: scalar
//// return/modified: U,V, R; t 
//// added: v0

double dual_update(const Eigen:: MatrixXd U_u, const Eigen::MatrixXd V_u, const Eigen::VectorXd R_u, 
                   double t_u, const Eigen::MatrixXd L, const Eigen::MatrixXd Z, const Eigen::MatrixXd W, double theta, 
                   double phi, Eigen::MatrixXd& U, Eigen::MatrixXd& V, Eigen::VectorXd& R, Eigen::VectorXd v0){
  
  int p = L.cols();
  U = U_u + L - Z;
  V = V_u + L + W; 
  R = R_u + L*v0; 
  
  double t = t_u + theta - phi; 
  return t;
}

///////////////////////////////////
////// Stopping CRITERIA /////////
///////////////////////////////////

///////// PRIMAL CRITERIA
///////// Inputs: L, Z, W, p x p symmetric; theta, phi, positive scalars; 
///////// eps_abs and eps_rel are convergence thresholds
///////// Return T if primal residual <= primal threshold

bool Primal_Cri(const Eigen::MatrixXd L, const Eigen::MatrixXd Z, const Eigen::MatrixXd W, double theta, double phi, Eigen::VectorXd v0, double eps_abs, double eps_rel, bool verbose){
  
  int p = L.cols();
  Eigen::VectorXd L1 = L*v0; // L %*% 1
  
  // Define L1_squared_norm
  double L1_norm2 = L1.squaredNorm();
  
  // Primal Residual
  double resi = std::sqrt((L-Z).squaredNorm() + (L+W).squaredNorm() + L1_norm2 + (theta-phi) * (theta - phi)); //// define double l1_norm2=L1.squaredNorm() and then reuse
  double temp1 = std::sqrt( 2 * L.squaredNorm() + L1_norm2 + theta * theta );
  double temp2 = std::sqrt( Z.squaredNorm() + W.squaredNorm() + phi * phi );
  
  // Calculate the Primal Threshold   
  double eps = std::sqrt(p*(2*p+1) + 1)*eps_abs + eps_rel*std::max(temp1, temp2);
  
  // Calculate the residual if verbose is true
  if (verbose){
    Rcpp::Rcout << "primal: " + std::to_string(resi) + " " + std::to_string(eps) << std::endl;  
  }
  
  return resi <= eps;
}

/////// DUAL RESIDUAL AND THRESHOLD
////// Corrected formula
double Resid_Dual_S(const Eigen::MatrixXd Z, const Eigen::MatrixXd Z_u, const Eigen::MatrixXd W, const Eigen::MatrixXd W_u, double phi, double phi_u, double rho){
  
  double resi = rho*std::sqrt((Z-Z_u+W-W_u).squaredNorm() + (phi-phi_u) * (phi - phi_u)); 
  return resi;
}

///// Corrected Formula
double Eps_Dual_S(const Eigen::MatrixXd U, const Eigen::MatrixXd V, const Eigen::VectorXd R, double t, double rho, Eigen::VectorXd v0, double eps_abs, double eps_rel){
  
  int p = U.rows();
  Eigen::MatrixXd R1 = R * v0.transpose(); 
  double temp = std::sqrt((U+V+R1).squaredNorm() + t * t);
  double eps = std::sqrt(p*p+1) * eps_abs + rho*temp*eps_rel;
  
  return eps;
}

////////////// Checks whether dual stopping criteria is met.
//// Inputs: Z, Z_u, W, W_u, phi, phi_u, U, V, R, t, rho
//// Returns: T if dual residual <= dual threshold
//// Added v0
bool Dual_Cri(const Eigen::MatrixXd Z, const Eigen::MatrixXd Z_u, const Eigen::MatrixXd W, const Eigen::MatrixXd W_u, double phi, double phi_u, 
              const Eigen::MatrixXd U, const Eigen::MatrixXd V, const Eigen::VectorXd R, double t, double rho, Eigen::VectorXd v0, double eps_abs, double eps_rel, bool verbose){
  
  double resi = Resid_Dual_S(Z, Z_u, W, W_u, phi, phi_u, rho);
  double eps = Eps_Dual_S(U, V, R, t, rho, v0, eps_abs, eps_rel);
  
  if (verbose){
    Rcpp::Rcout << "dual: " + std::to_string(resi) + " " + std::to_string(eps) << std::endl; 
  }
  
  return resi <= eps;
}

/////////////////////////////////////
/////// ADMM ALGORITHM CODE ////////
////////////////////////////////////
//// S: Sufficient statistic for covariance matrix; p x p symmetric
//// rho: ADMM scaled Lagrange parameter; positive scalr
//// lambda: l1 trace penalty; ppsitive scalar
//// Z_ini: Initial value for Z; W_ini: initial value for W; phi_ini: initival value for phi;
//// eps_thre: detault initial value and lower bound for phi; small positive scalar
//// eps_abs, eps_rel: parameters for stopping criteria; max_iter: maximum number of iterations
//// Z_max_iter, Z_conv_abs, Z_conv_rel: maximum number of steps and convergence thresholds for Z-phi minimization step (inner iteration)
//// Return: L, Z, W, and convergence status.

//' ADMM algorithm to estimate graph filter parameters and (normalized) graph Laplacian simultaneously, given v0.
//' 
//' @param SS an estimated covariance matrix, such as the sample covariance matrix.
//' @param V0 A given degree vector.
//' @param rho ADMM parameter. 
//' @param lambda Tuning parameter.
//' @param model A character specifying which type of Laplacian to use.
//' @param ZZ_ini An initial value for the ADMM algorithm.
//' @param WW_ini An initial value for the ADMM algorithm.
//' @param phi_ini An initial value for the ADMM algorithm.
//' @param eps_thre A small positive number
//' @param eps_abs ADMM stopping criterion
//' @param eps_rel ADMM stopping criterion
//' @param max_iter Maximum number of iterations to run
//' @param Z_max_iter Maximum number of iterations to run inner algorithm.
//' @param Z_conv_abs Convergence criterion for inner algorithm
//' @param Z_conv_rel Convergence criterion for inner algorithm
//' @param verbose Trace of ADMM algorithm
//' 
//' @returns
//' * `L`: A p by p matrix.
//' * `theta0`: A positive number.
//' * `theta1`: A positive number.
//' * `Z`: A p by p matrix.
//' * `phi`: A positive number.
//' * `W`: A p by p matrix.
//' * `conv:` A boolean indicating convergence (TRUE if converged).
//' @export
// [[Rcpp::export]]
List ADMM_Lap(const Rcpp::NumericMatrix& SS, Rcpp::NumericVector& V0, double rho, double lambda, std::string model, const Rcpp::NumericMatrix& ZZ_ini, const Rcpp::NumericMatrix& WW_ini, double phi_ini, double eps_thre, double eps_abs, double eps_rel, int max_iter, int Z_max_iter, double Z_conv_abs, double Z_conv_rel, bool verbose){
  
  // convert to Eigen: MatrixXd 
  const MapMat S = Rcpp::as<MapMat>(SS);
  const MapMat Z_ini = Rcpp::as<MapMat>(ZZ_ini); //********* not yet used 
  const MapMat W_ini = Rcpp::as<MapMat>(WW_ini); //********* not yet used 
  MapVec v0 = Rcpp::as<MapVec>(V0); 
  
  // Initialization: 
  int p = S.rows(); 
  // Here is where temp_ini would be, and Z, S, L, phi, theta initialization.
  Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(p,p);
  Eigen::MatrixXd W = Z;  //// W=Z
  Eigen::MatrixXd L = Z;   //// L=Z
  
  double phi = eps_thre;
  double theta = phi;
  
  Eigen::MatrixXd U = Z;  /// U=Z
  Eigen::MatrixXd V = Z;   ///V=Z 
  Eigen::VectorXd R = Eigen::VectorXd::Zero(p);
  
  double t = 0; // Initialize Objects used in While loop
  Eigen::MatrixXd L_u = L;
  double theta_u = theta; 
  double phi_u = phi;
  double t_u = t; 
  
  Eigen::MatrixXd Z_u = Z;
  Eigen::MatrixXd W_u = W; 
  Eigen::MatrixXd U_u = U; 
  Eigen::MatrixXd V_u = V;
  Eigen::VectorXd R_u = R;
  
  /// Prepare constant objects for L-theta minimization 
  List temp = L_theta_pre(S, v0, rho); // Returns a list of "Q"=Q, "Dx.inv" = Dx_inv, "Dx.inv_Sx" = Dx_inv_Sx, "dstar" = dstar
  Eigen::MatrixXd Q = temp["Q"];
  Eigen::MatrixXd QT = temp["QT"];
  
  Eigen::VectorXd Dx_inv = temp["Dx.inv"];
  Eigen::VectorXd Dx_inv_Sx = temp["Dx.inv.Sx"];
  double dstar = temp["dstar"];
  
  // Iteration:
  bool conv = false; 
  int iter = 0; 
  
  while(!conv && (iter<=max_iter)){ /// define L_u. etc., outside of the loop
    
    if(verbose){
      Rcpp::Rcout << "ADMM step: " + std::to_string(iter) << std::endl; 
    }
    
    // L-theta step: returns list with: "L" = L_new and "theta" = theta_new
    theta_u = L_theta_min(Q, QT, Dx_inv, Dx_inv_Sx, dstar, Z, W, U, V, R, phi, t, rho, lambda, L_u, v0, model);
    
    // Z-phi step: returns list with: "Z" = Z_new, "phi" = phi_new
    List temp_Z = Z_phi_Min(L_u, U, phi, theta_u, t, rho, eps_thre, Z_max_iter, Z_conv_abs, Z_conv_rel, verbose, Z_u);
    phi_u = temp_Z["phi"]; // Updated phi
    
    bool Z_conv_stat= temp_Z["status"]; // convergence status
    if (Z_conv_stat == false){
      throw std::runtime_error("Error: Z-phi step not converged");
    }
    
    // W step
    W_Min_S(L_u, V, model, eps_thre, W_u); 
    
    // Dual step
    t_u = dual_update(U, V, R, t, L_u, Z_u, W_u, theta_u, phi_u, U_u, V_u, R_u, v0); 
    
    // Convergence check
    //// Primal check at kth step
    bool pri_conv = Primal_Cri(L, Z, W, theta, phi, v0, eps_abs, eps_rel, verbose); 
    
    //// Dual check by comparing kth step with (k-1)th step. 
    if (pri_conv){
      bool dual_conv = Dual_Cri(Z, Z_u, W, W_u, phi, phi_u, U, V, R, t, rho, v0, eps_abs, eps_rel, verbose);
      if (dual_conv){
        conv = true;
      }
    }
    
    // Updates
    L = L_u;
    theta = theta_u;
    
    Z = Z_u;
    phi = phi_u;
    W = W_u; 
    
    U = U_u;
    V = V_u;
    R = R_u; 
    t = t_u; 
    
    // Update iteration counter
    iter++;
  }
  
  // Update theta1 and L
  double theta1 = L.diagonal().maxCoeff();
  L = L.array()/theta1; 
  
  return List::create(Named("L") = L, Named("theta0") = theta, Named("theta1")=theta1, Named("Z") = Z, Named("phi") = phi, Named("W") = W, Named("conv") = conv);
  
}

//' ADMM algorithm to estimate graph filter parameters and (normalized) graph Laplacian simultaneously, given v0 and zero pattern.
//' 
//' @param SS an estimated covariance matrix, such as the sample covariance matrix.
//' @param V0 A given degree vector.
//' @param rho ADMM parameter. 
//' @param AA A matrix encoding the zero-pattern of the (normalized) graph Laplacian.
//' @param lambda Tuning parameter.
//' @param model A character specifying which type of Laplacian to use.
//' @param ZZ_ini An initial value for the ADMM algorithm.
//' @param WW_ini An initial value for the ADMM algorithm.
//' @param phi_ini An initial value for the ADMM algorithm.
//' @param eps_thre A small positive number
//' @param eps_abs ADMM stopping criterion
//' @param eps_rel ADMM stopping criterion
//' @param max_iter Maximum number of iterations to run
//' @param Z_max_iter Maximum number of iterations to run inner algorithm.
//' @param Z_conv_abs Convergence criterion for inner algorithm
//' @param Z_conv_rel Convergence criterion for inner algorithm
//' @param verbose Trace of ADMM algorithm
//' 
//' @returns
//' * `L`: A p by p matrix.
//' * `theta0`: A positive number.
//' * `theta1`: A positive number.
//' * `Z`: A p by p matrix.
//' * `phi`: A positive number.
//' * `W`: A p by p matrix.
//' * `conv:` A boolean indicating convergence (TRUE if converged).
//' @export
// [[Rcpp::export]]
List ADMM_Lap_Zero(const Rcpp::NumericMatrix& SS, Rcpp::NumericVector& V0, double rho, const Rcpp::NumericMatrix& AA, std::string model, const Rcpp::NumericMatrix& ZZ_ini, const Rcpp::NumericMatrix& WW_ini, double phi_ini, double eps_thre, double eps_abs, double eps_rel, int max_iter, int Z_max_iter, double Z_conv_abs, double Z_conv_rel, bool verbose){
  
  // convert to Eigen: MatrixXd 
  const MapMat S = Rcpp::as<MapMat>(SS);
  const MapMat A = Rcpp::as<MapMat>(AA); 
  const MapMat Z_ini = Rcpp::as<MapMat>(ZZ_ini); //********* not yet used 
  const MapMat W_ini = Rcpp::as<MapMat>(WW_ini); //********* not yet used 
  MapVec v0 = Rcpp::as<MapVec>(V0); 
  
  // Initialization: 
  int p = S.rows(); 
  int lambda = 0; // For zero function, set lambda = 0
  // Here is where temp_ini would be, and Z, S, L, phi, theta initialization.
  Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(p,p);
  Eigen::MatrixXd W = Z.array()*A.array();  //// W=Z
  Eigen::MatrixXd L = Z;   //// L=Z
  
  double phi = eps_thre;
  double theta = phi;
  
  Eigen::MatrixXd U = Z;  /// U=Z
  Eigen::MatrixXd V = Z;   ///V=Z 
  Eigen::VectorXd R = Eigen::VectorXd::Zero(p);
  
  double t = 0; // Initialize Objects used in While loop
  Eigen::MatrixXd L_u = L;
  double theta_u = theta; 
  double phi_u = phi;
  double t_u = t; 
  
  Eigen::MatrixXd Z_u = Z;
  Eigen::MatrixXd W_u = W; 
  Eigen::MatrixXd U_u = U; 
  Eigen::MatrixXd V_u = V;
  Eigen::VectorXd R_u = R;
  
  /// Prepare constant objects for L-theta minimization 
  List temp = L_theta_pre(S, v0, rho); // Returns a list of "Q"=Q, "Dx.inv" = Dx_inv, "Dx.inv_Sx" = Dx_inv_Sx, "dstar" = dstar
  Eigen::MatrixXd Q = temp["Q"];
  Eigen::MatrixXd QT = temp["QT"];
  
  Eigen::VectorXd Dx_inv = temp["Dx.inv"];
  Eigen::VectorXd Dx_inv_Sx = temp["Dx.inv.Sx"];
  double dstar = temp["dstar"];
  
  // Iteration:
  bool conv = false; 
  int iter = 0; 
  
  while(!conv && (iter<=max_iter)){ /// define L_u. etc., outside of the loop
    
    if(verbose){
      Rcpp::Rcout << "ADMM step: " + std::to_string(iter) << std::endl; 
    }
    
    // L-theta step: returns list with: "L" = L_new and "theta" = theta_new
    theta_u = L_theta_min(Q, QT, Dx_inv, Dx_inv_Sx, dstar, Z, W, U, V, R, phi, t, rho, lambda, L_u, v0, model);
    
    // Z-phi step: returns list with: "Z" = Z_new, "phi" = phi_new
    List temp_Z = Z_phi_Min(L_u, U, phi, theta_u, t, rho, eps_thre, Z_max_iter, Z_conv_abs, Z_conv_rel, verbose, Z_u);
    phi_u = temp_Z["phi"]; // Updated phi
    
    bool Z_conv_stat= temp_Z["status"]; // convergence status
    if (Z_conv_stat == false){
      throw std::runtime_error("Error: Z-phi step not converged");
    }
    
    // W step
    W_Min_S_Zero(L_u, V, A, model, eps_thre, W_u); 
    
    // Dual step
    t_u = dual_update(U, V, R, t, L_u, Z_u, W_u, theta_u, phi_u, U_u, V_u, R_u, v0); 
    
    // Convergence check
    //// Primal check at kth step
    bool pri_conv = Primal_Cri(L, Z, W, theta, phi, v0, eps_abs, eps_rel, verbose); 
    
    //// Dual check by comparing kth step with (k-1)th step. 
    if (pri_conv){
      bool dual_conv = Dual_Cri(Z, Z_u, W, W_u, phi, phi_u, U, V, R, t, rho, v0, eps_abs, eps_rel, verbose);
      if (dual_conv){
        conv = true;
      }
    }
    
    // Updates
    L = L_u;
    theta = theta_u;
    
    Z = Z_u;
    phi = phi_u;
    W = W_u; 
    
    U = U_u;
    V = V_u;
    R = R_u; 
    t = t_u; 
    
    // Update iteration counter
    iter++;
  }
  
  // Update theta1 and L
  double theta1 = L.diagonal().maxCoeff();
  L = L.array()/theta1; 
  
  return List::create(Named("L") = L, Named("theta0") = theta, Named("theta1")=theta1, Named("Z") = Z, Named("phi") = phi, Named("W") = W, Named("conv") = conv);
  
}