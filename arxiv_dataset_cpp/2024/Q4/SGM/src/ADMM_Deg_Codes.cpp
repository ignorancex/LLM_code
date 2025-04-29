#include <Rcpp.h>
#include <RcppEigen.h>
#include <boost/math/tools/roots.hpp>
#include <limits>
using namespace Rcpp;
using namespace Eigen;
using namespace boost::math::tools;


typedef Eigen::Map<Eigen::MatrixXd> MapMat; 
typedef Eigen::Map<Eigen::VectorXd> MapVec;

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(BH)]]

///////////////////
// V Min function
// (Q, Lambda) = Eigendecomposition of S=Q' Lambda Q;
// lambda_s: leading eigenvalue of S
// w: p x 1: v-w = 0;
// u: p x 1, dual variable;
// rho: admm parameter
// return: Updated v

// Define the class for function, where root is used during v_min step
class func_root {
public:
  // Constructor to initialize member variables
  func_root(const Eigen::VectorXd& Lambda_, double lambda_s_, Eigen::VectorXd& tk_tilde_)
    : Lambda(Lambda_), lambda_s(lambda_s_), tk_tilde(tk_tilde_) {}
  
  // Operator function to compute the root-finding objective
  double operator()(double mu) const {
    Eigen::VectorXd Lambda_tilde = (Lambda.array() - lambda_s).square() + 2 * mu;
    Eigen::VectorXd Lambda_inv = 1.0 / Lambda_tilde.array();
    double val = (Lambda_inv.array() * tk_tilde.array()).square().sum() - 1.0;
    return val;
  }
  
private:
  // Member variables to store input data
  const Eigen::VectorXd& Lambda;
  double lambda_s;
  Eigen::VectorXd& tk_tilde;
};

// Actual v-step
void v_Min(const Eigen::MatrixXd& Q, const Eigen::VectorXd& Lambda, double lambda_s, double rho, Eigen::VectorXd& w, Eigen::VectorXd& u, bool verbose, Eigen::VectorXd& v){
  int p = Q.rows(); 
  Eigen::VectorXd tk = rho*(w-u); 
  Eigen::VectorXd tk_tilde = Q*tk; 
  
  // Define the root-finding function
  func_root f(Lambda, lambda_s, tk_tilde); 
  
  // Calculate the tolerance
  boost::math::tools::eps_tolerance<double> tol(64);
  boost::uintmax_t max_iter = 1000;
  double min_mu = 1e-6;
  double max_mu = 1e6;
  
  // Perform the root finding
  std::pair<double, double> result = boost::math::tools::toms748_solve(f, min_mu, max_mu, tol, max_iter);
  double mu_s = result.first;
  
  if (verbose) {
    Rcpp::Rcout << "v step root: " << mu_s << std::endl;
  }
  
  Eigen::VectorXd Lambda_inv_s = 1/((Lambda.array() - lambda_s).square() + 2*mu_s).array();
  Eigen::VectorXd temp = Lambda_inv_s.array()*tk_tilde.array(); 
  v = Q.transpose()*temp; 
}

///////////////////
// W Step
//////////////////
// v: p x 1 (v-w)=0
// u: p x 1 (dual variable)
// epsilon: small, positive value
// return: updated w

void w_Min(Eigen::VectorXd& v, Eigen::VectorXd& u, double epsilon, Eigen::VectorXd& w){
  Eigen::VectorXd temp = v+u; 
  w = temp; 
  w = (temp.array() < epsilon).select(epsilon,w);
}

// [[Rcpp::export]]
Rcpp::List ADMM_Deg(const Rcpp::NumericMatrix& s, double rho, double epsilon, double eps_abs, double eps_rel, int max_iter, bool verbose){
  
  const MapMat S = Rcpp::as<MapMat>(s); 
  
  
  int p = S.rows(); 
  
  // Eigendecomposition via symmetric structure
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(S);
  Eigen::MatrixXd Q = eigensolver.eigenvectors().transpose(); // vectors
  Eigen::VectorXd Lambda = eigensolver.eigenvalues(); // values 
  double lambda_s = Lambda.maxCoeff(); 
  
  // Initial values 
  Eigen::VectorXd w = Eigen::VectorXd::Ones(p)*1/std::sqrt(p); // Initial value
  Eigen::VectorXd u = Eigen::VectorXd::Zero(p); // Initial Value
  Eigen::VectorXd v = u; // Initial Value
  
  Eigen::VectorXd w_u(p);
  Eigen::VectorXd u_u(p);
  Eigen::VectorXd v_u(p); 
  
  bool conv = false; 
  int iter = 0;
  while ( (!conv) && iter<=max_iter){
    
    v_Min(Q, Lambda, lambda_s, rho, w, u, verbose, v_u); // v update
    w_Min(v_u, u, epsilon, w_u); // w update
    u_u = u+(v_u-w_u); // dual update
    
    // Convergence Criteria
    double res_prim = std::sqrt( (v_u-w_u).squaredNorm() );
    double res_dual = rho*std::sqrt((w-w_u).squaredNorm()); 
    
    double eps_prim = std::sqrt(2*p)*eps_abs + eps_rel*std::max(std::sqrt(v_u.squaredNorm()), std::sqrt(w_u.squaredNorm()));
    double eps_dual = std::sqrt(p)*eps_abs + eps_rel*rho*std::sqrt(u_u.squaredNorm()); 
    
    if (verbose){
      Rcpp::Rcout << "primal: " + std::to_string(res_prim) + " " + std::to_string(eps_prim) << std::endl; 
      Rcpp::Rcout << "dual: " + std::to_string(res_dual) + " " + std::to_string(eps_dual) << std::endl;
    }
    
    if (res_prim<=eps_prim && res_dual<= eps_dual){
      conv = true; 
    }
    
    iter++; // Update iteration
    
    // Update iterates 
    v = v_u; 
    w = w_u; 
    u = u_u;
  }
  
  return Rcpp::List::create(Named("v")=v, Named("w")=w, Named("deg") = v.array().square(), Named("conv") = conv);
}


//' ADMM for Degree estimation
//' 
//' Given a (normalized) graph laplacian, estimate the degree vector by an ADMM algorithm.  Same as `ADMM.Deg.L`. but written in C++ and uses a different root-solving algorithm.
//' 
//' @param L A  p by p (normalized) graph laplacian
//' @param rho ADMM parameter (positive number)
//' @param epsilon small positive number
//' @param eps_abs ADMM stopping criterion
//' @param eps_rel ADMM stopping criterion
//' @param max_iter Maximum number of iterations
//' @param verbose Trace of algorithm
//' 
//' @returns A list object
//' * `v`: A p by 1 matrix 
//' * `w`: A p by 1 matrix
//' * `deg`: A p by 1 matrix 
//' * `conv` A boolean indicating convergence (TRUE if converged)
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List ADMM_Deg_L(const Rcpp::NumericMatrix& L, double rho, double epsilon, double eps_abs, double eps_rel, int max_iter, bool verbose){
  
  const MapMat S = Rcpp::as<MapMat>(L); 
  
  
  int p = S.rows(); 
  
  // Eigendecomposition via symmetric structure
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(S);
  Eigen::MatrixXd Q = eigensolver.eigenvectors().transpose(); // vectors
  Eigen::VectorXd Lambda = eigensolver.eigenvalues(); // values 
  double lambda_s = 0; // Change to 0 for Laplacian case 
  
  // Initial values 
  Eigen::VectorXd w = Eigen::VectorXd::Ones(p)*1/std::sqrt(p); // Initial value
  Eigen::VectorXd u = Eigen::VectorXd::Zero(p); // Initial Value
  Eigen::VectorXd v = u; // Initial Value
  
  Eigen::VectorXd w_u(p);
  Eigen::VectorXd u_u(p);
  Eigen::VectorXd v_u(p); 
  
  bool conv = false; 
  int iter = 0;
  while ( (!conv) && iter<=max_iter){
    
    v_Min(Q, Lambda, lambda_s, rho, w, u, verbose, v_u); // v update
    w_Min(v_u, u, epsilon, w_u); // w update
    u_u = u+(v_u-w_u); // dual update
    
    // Convergence Criteria
    double res_prim = std::sqrt( (v_u-w_u).squaredNorm() );
    double res_dual = rho*std::sqrt((w-w_u).squaredNorm()); 
    
    double eps_prim = std::sqrt(2*p)*eps_abs + eps_rel*std::max(std::sqrt(v_u.squaredNorm()), std::sqrt(w_u.squaredNorm()));
    double eps_dual = std::sqrt(p)*eps_abs + eps_rel*rho*std::sqrt(u_u.squaredNorm()); 
    
    if (verbose){
      Rcpp::Rcout << "primal: " + std::to_string(res_prim) + " " + std::to_string(eps_prim) << std::endl; 
      Rcpp::Rcout << "dual: " + std::to_string(res_dual) + " " + std::to_string(eps_dual) << std::endl;
    }
    
    if (res_prim<=eps_prim && res_dual<= eps_dual){
      conv = true; 
    }
    
    iter++; // Update iteration
    
    // Update iterates 
    v = v_u; 
    w = w_u; 
    u = u_u;
  }
  
  return Rcpp::List::create(Named("v")=v, Named("w")=w, Named("deg") = v.array().square(), Named("conv") = conv);
}

