#include <RcppArmadillo.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>

#include "rng.h"
#include "tree.h"
#include "info.h"
#include "funs.h"
#include "bd.h"
#include "logging.h"
#include "param_draws.h"

using namespace Rcpp;
// Rstudios check's suggest not ignoring these
// #pragma GCC diagnostic ignored "-Wunused-parameter"
// #pragma GCC diagnostic ignored "-Wcomment"
// #pragma GCC diagnostic ignored "-Wformat"
// #pragma GCC diagnostic ignored "-Wsign-compare"

// y = m(x) + b(x)z + e, e~N(0, sigma^2_y

//x_con is the design matrix for m. It should have n = rows
//x_mod is the design matrix for b. It should have n = rows
//data should come in sorted with all trt first, then control cases

// [[Rcpp::export]]
List bcfoverparRcppClean(NumericVector y_, NumericVector z_, NumericVector w_,
                  NumericVector x_con_, NumericVector x_mod_, 
                  List x_con_info_list, List x_mod_info_list, 
                  int burn, int nd, int thin, //Draw nd*thin + burn samples, saving nd draws after burn-in
                  int ntree_mod, int ntree_con,
                  double lambda, double nu, //prior pars for sigma^2_y
                  double con_sd, // Var(m(x)) = con_sd^2 marginally a priori (approx)
                  double mod_sd, // Var(b(x)) = mod_sd^2 marginally a priori (approx)
                  double con_alpha, double con_beta,
                  double mod_alpha, double mod_beta,
                  CharacterVector treef_con_name_, CharacterVector treef_mod_name_,
                  int status_interval=100,
                  bool RJ= false, bool use_mscale=true, bool use_bscale=true, bool b_half_normal=true,
                  double trt_init = 1.0, int verbose=1)
{

  std::ofstream treef_con;
  std::ofstream treef_mod;
  
  std::string treef_con_name = as<std::string>(treef_con_name_);
  std::string treef_mod_name = as<std::string>(treef_mod_name_);
  
  if(not treef_con_name.empty()){
    Rcout << "Saving Trees to"  << std::endl;
    Rcout << treef_con_name  << std::endl;
    Rcout << treef_mod_name  << std::endl;

    treef_con.open(treef_con_name.c_str());
    treef_mod.open(treef_mod_name.c_str());
  }else{
    Rcout << "Not Saving Trees to file"  << std::endl;
  }
  
  RNGScope scope;
  RNG gen; //this one random number generator is used in all draws

  //double lambda = 1.0; //this one really needs to be set
  //double nu = 3.0;
  //double kfac=2.0; //original is 2.0
  
  Logger logger = Logger();  
  char logBuff[100];

  // Logger only used for detailed logging
  bool log_level = verbose > 1;

  logger.setLevel(log_level);
  
  logger.log("============================================================");
  logger.log(" Starting up BCF: ");
  logger.log("============================================================");
  if (log_level){
    logger.getVectorHead(y_, logBuff);
    Rcout << "y: " <<  logBuff << "\n";
    logger.getVectorHead(z_, logBuff);
    Rcout << "z: " <<  logBuff << "\n";
    logger.getVectorHead(w_, logBuff);
    Rcout << "w: " <<  logBuff << "\n";
  }

  logger.log("BCF is Weighted");

  // sprintf(logBuff, "Updating Moderate Tree: %d of %d");
  // logger.log(logBuff);
  logger.log("");

  /*****************************************************************************
  /* Read, format y
  *****************************************************************************/
  std::vector<double> y; //storage for y
  double miny = INFINITY, maxy = -INFINITY;
  sinfo allys;       //sufficient stats for all of y, use to initialize the bart trees.
  double allys_y2 = 0;

  for(NumericVector::iterator it=y_.begin(); it!=y_.end(); ++it) {
    y.push_back(*it);
    if(*it<miny) miny=*it;
    if(*it>maxy) maxy=*it;
    allys.sy += *it; // sum of y
    allys_y2 += (*it)*(*it); // sum of y^2
  }
  size_t n = y.size();
  allys.n = n;

  double ybar = allys.sy/n; //sample mean
  double shat = sqrt((allys_y2-n*ybar*ybar)/(n-1)); //sample standard deviation
  /*****************************************************************************
  /* Read, format  weights 
  *****************************************************************************/
  double* w = new double[n]; //y-(allfit-ftemp) = y-allfit+ftemp


  for(int j=0; j<n; j++) {
    w[j] = w_[j];
  }


  /*****************************************************************************
  /* Read, format X_con
  *****************************************************************************/
  //the n*p numbers for x are stored as the p for first obs, then p for second, and so on.
  std::vector<double> x_con;
  for(NumericVector::iterator it=x_con_.begin(); it!= x_con_.end(); ++it) {
    x_con.push_back(*it);
  }
  size_t p_con = x_con.size()/n;

  Rcout << "Using " << p_con << " control variables." << std::endl;

  //x cutpoints
  xinfo xi_con;

  xi_con.resize(p_con);
  for(int i=0; i<p_con; ++i) {
    NumericVector tmp = x_con_info_list[i];
    std::vector<double> tmp2;
    for(size_t j=0; j<tmp.size(); ++j) {
      tmp2.push_back(tmp[j]);
    }
    xi_con[i] = tmp2;
  }

  /*****************************************************************************
  /* Read, format X_mod
  *****************************************************************************/
  int ntrt = 0;
  for(size_t i=0; i<n; ++i) {
    if(z_[i]>0) ntrt += 1;
  }
  std::vector<double> x_mod;
  for(NumericVector::iterator it=x_mod_.begin(); it!= x_mod_.end(); ++it) {
    x_mod.push_back(*it);
  }
  size_t p_mod = x_mod.size()/n;

  Rcout << "Using " << p_mod << " potential effect moderators." << std::endl;

  //x cutpoints
  xinfo xi_mod;

  xi_mod.resize(p_mod);
  for(int i=0; i<p_mod; ++i) {
    NumericVector tmp = x_mod_info_list[i];
    std::vector<double> tmp2;
    for(size_t j=0; j<tmp.size(); ++j) {
      tmp2.push_back(tmp[j]);
    }
    xi_mod[i] = tmp2;
  }

  //  Rcout <<"\nburn,nd,number of trees: " << burn << ", " << nd << ", " << m << endl;
  //  Rcout <<"\nlambda,nu,kfac: " << lambda << ", " << nu << ", " << kfac << endl;

  /*****************************************************************************
  /* Setup the model
  *****************************************************************************/
  //--------------------------------------------------
  //trees
  std::vector<tree> t_mod(ntree_mod);
  for(size_t i=0;i<ntree_mod;i++) t_mod[i].setm(trt_init/(double)ntree_mod);

  std::vector<tree> t_con(ntree_con);
  for(size_t i=0;i<ntree_con;i++) t_con[i].setm(ybar/(double)ntree_con);

  //--------------------------------------------------
  //prior parameters
  // PX scale parameter for b: 
  double bscale_prec = 2;
  double bscale0 = -0.5;
  double bscale1 = 0.5;

  double mscale_prec = 1.0;
  double mscale = 1.0;
  double delta_con = 1.0;
  double delta_mod = 1.0;

  pinfo pi_mod;
  pi_mod.pbd = 1.0; //prob of birth/death move
  pi_mod.pb = .5; //prob of birth given  birth/death

  pi_mod.alpha = mod_alpha; //prior prob a bot node splits is alpha/(1+d)^beta, d is depth of node
  pi_mod.beta  = mod_beta;  //2 for bart means it is harder to build big trees.
  pi_mod.tau   = mod_sd/(sqrt(delta_mod)*sqrt((double) ntree_mod)); //sigma_mu, variance on leaf parameters
  pi_mod.sigma = shat; //resid variance is \sigma^2_y/bscale^2 in the backfitting update

  pinfo pi_con;
  pi_con.pbd = 1.0; //prob of birth/death move
  pi_con.pb = .5; //prob of birth given  birth/death

  pi_con.alpha = con_alpha;
  pi_con.beta  = con_beta;
  pi_con.tau   = con_sd/(sqrt(delta_con)*sqrt((double) ntree_con)); //sigma_mu, variance on leaf parameters

  pi_con.sigma = shat/fabs(mscale); //resid variance in backfitting is \sigma^2_y/mscale^2

  double sigma = shat;

  // @Peter This is where dinfo is initialized

  //--------------------------------------------------
  //dinfo for control function m(x)
//  Rcout << "ybar " << ybar << endl;
  double* allfit_con = new double[n]; //sum of fit of all trees
  for(size_t i=0;i<n;i++) allfit_con[i] = ybar;
  double* r_con = new double[n]; //y-(allfit-ftemp) = y-allfit+ftemp
  dinfo di_con;
  di_con.n=n; 
  di_con.p = p_con; 
  di_con.x = &x_con[0]; 
  di_con.y = r_con; //the y for each draw will be the residual

  //--------------------------------------------------
  //dinfo for trt effect function b(x)
  double* allfit_mod = new double[n]; //sum of fit of all trees
  for(size_t i=0;i<n;i++) allfit_mod[i] = (z_[i]*bscale1 + (1-z_[i])*bscale0)*trt_init;
  double* r_mod = new double[n]; //y-(allfit-ftemp) = y-allfit+ftemp
  dinfo di_mod;
  di_mod.n=n; 
  di_mod.p=p_mod; 
  di_mod.x = &x_mod[0]; 
  di_mod.y = r_mod; //the y for each draw will be the residual

  //--------------------------------------------------

  //--------------------------------------------------
  //storage for the fits
  double* allfit = new double[n]; //yhat
  for(size_t i=0;i<n;i++) {
    allfit[i] = allfit_mod[i] + allfit_con[i];
  }
  double* ftemp  = new double[n]; //fit of current tree

  NumericVector sigma_post(nd);
  NumericVector msd_post(nd);
  NumericVector bsd_post(nd);
  NumericVector b0_post(nd);
  NumericVector b1_post(nd);
  NumericMatrix m_post(nd,n);
  NumericMatrix yhat_post(nd,n);
  NumericMatrix b_post(nd,n);

  //  NumericMatrix spred2(nd,dip.n);


  // The default output precision is of C++ is 5 or 6 dp, depending on compiler.
  // I don't have much justification for 32, but it seems like a sensible number   
  int save_tree_precision = 32; 

  //save stuff to tree file
  if(not treef_con_name.empty()){
    treef_con << std::setprecision(save_tree_precision) << xi_con << endl; //cutpoints
    treef_con << ntree_con << endl;  //number of trees
    treef_con << di_con.p << endl;  //dimension of x's
    treef_con << nd << endl;
  
    treef_mod << std::setprecision(save_tree_precision) << xi_mod << endl; //cutpoints
    treef_mod << ntree_mod << endl;  //number of trees
    treef_mod << di_mod.p << endl;  //dimension of x's
    treef_mod << nd << endl;
  }

  //*****************************************************************************
  /* MCMC
   * note: the allfit objects are all carrying the appropriate scales
   */
  //*****************************************************************************
  Rcout << "\n============================================================\nBeginning MCMC:\n============================================================\n";
  time_t tp;
  int time1 = time(&tp);

  size_t save_ctr = 0;
  bool verbose_itr; 

  double* weight      = new double[n];
  double* weight_het  = new double[n];

  logger.setLevel(0);

  bool printTrees = verbose > 2;

  // Vector of references to which scale applies to each individual
  std::vector<std::reference_wrapper<double>> mscale_idx;
  std::vector<std::reference_wrapper<double>> bscale_idx;
  for (size_t i=0;i<n;i++) {
    mscale_idx.push_back(mscale);
    bscale_idx.push_back(i<ntrt ? bscale1 : bscale0);
  }

  // General info - shortcut for passing to functions
  ginfo ginfo = {.n       = n,
                 .ntrt    = ntrt,
                 .z_      = z_,
                 .y       = y,
                 .w       = w,
                 .gen     = gen,
                 .logger  = logger};

  winfo wi_con = {.ntree      = ntree_con,
                  .t          = t_con,
                  .sd         = con_sd,
                  .scale_prec = mscale_prec,
                  .scale_idx  = mscale_idx,
                  .xi         = xi_con, 
                  .di         = di_con, 
                  .pi         = pi_con,
                  .ri         = r_con,
                  .weight     = weight,
                  .delta      = delta_con};

  winfo wi_mod = {.ntree      = ntree_mod,
                  .t          = t_mod,
                  .sd         = mod_sd,
                  .scale_prec = bscale_prec,
                  .scale_idx  = bscale_idx,
                  .xi         = xi_mod, 
                  .di         = di_mod, 
                  .pi         = pi_mod,
                  .ri         = r_mod,
                  .weight     = weight_het,
                  .delta      = delta_mod};

  for(size_t iIter=0;iIter<(nd*thin+burn);iIter++) {
    // Only log iters that will be saved, and only if desired
    verbose_itr = (verbose >= 2 && (iIter>=burn) && (iIter % thin==0)) || verbose==4;

    if(verbose > 0){
        if(iIter%status_interval==0) {
            Rcout << "iteration: " << iIter << " sigma/SD(y): "<< sigma << endl;
        }
    }

    logger.setLevel(verbose_itr);

    log_iter("Start", iIter+1, nd*thin+burn, sigma, mscale, bscale0, bscale1, logger);
    
    log_fit(y, allfit, allfit_con, allfit_mod, logger, verbose_itr);

    for (int k=0; k<n; ++k){
      weight[k] = w[k]*mscale*mscale/(sigma * sigma); // for non-het case, weights need to be divided by sigma square to make it similar to phi
    }

    for(size_t k=0; k<ntrt; ++k) {
      weight_het[k] = w[k]*bscale1*bscale1/(sigma*sigma);
    }
    for(size_t k=ntrt; k<n; ++k) {
      weight_het[k] = w[k]*bscale0*bscale0/(sigma*sigma);
    }

    logger.log("=====================================");
    logger.log("- Tree Processing");
    logger.log("=====================================");

    update_trees("control",  
                  allfit, allfit_con, 
                  mscale, bscale0, bscale1,
                  ginfo, wi_con, verbose_itr && printTrees);

    update_trees("moderate",  
                  allfit, allfit_mod, 
                  mscale, bscale0, bscale1,
                  ginfo, wi_mod, verbose_itr && printTrees);

    logger.log("=====================================");
    logger.log("- MCMC iteration Cleanup");
    logger.log("=====================================");

    if(use_bscale) {
      update_bscale(bscale0, bscale1, 
                    b_half_normal, sigma,
                    allfit_con, allfit_mod,
                    ginfo, wi_mod, verbose_itr);
    }

    if(use_mscale) {
     update_mscale(mscale, sigma, 
                    allfit_con, allfit_mod,
                    ginfo, wi_con, verbose_itr);
    }

    if(use_mscale || use_bscale) {
      logger.log("Sync allfits after scale updates");
      for(size_t k=0; k<n; ++k) {
        allfit[k] = allfit_con[k] + allfit_mod[k];
      }
    }

    update_sigma(allfit, sigma, nu, lambda, mscale, pi_con, pi_mod, ginfo);

    if( ((iIter>=burn) & (iIter % thin==0)) )  {
      if(not treef_con_name.empty()){
        for(size_t j=0;j<ntree_con;j++) treef_con << std::setprecision(save_tree_precision) << t_con[j] << endl; // save trees
        for(size_t j=0;j<ntree_mod;j++) treef_mod << std::setprecision(save_tree_precision) << t_mod[j] << endl; // save trees
      }
      
      save_values(save_ctr, n, ntrt, msd_post, bsd_post, b0_post, b1_post, sigma_post,
                  mscale, bscale1, bscale0, sigma, m_post, yhat_post, b_post,
                  allfit, allfit_con, allfit_mod);
    }

    log_iter("End", iIter+1, nd*thin+burn, sigma, mscale, bscale0, bscale1, logger);
    
    log_fit(y, allfit, allfit_con, allfit_mod, logger, verbose_itr);

  } // end MCMC Loop

  int time2 = time(&tp);
  Rcout << "\n============================================================\n MCMC Complete \n============================================================\n";

  Rcout << "time for loop: " << time2 - time1 << endl;

  t_mod.clear(); t_con.clear();
  delete[] allfit;
  delete[] allfit_mod;
  delete[] allfit_con;
  delete[] r_mod;
  delete[] r_con;
  delete[] ftemp;
  
  if(not treef_con_name.empty()){
    treef_con.close();
    treef_mod.close();
  }
  
  return(List::create(_["yhat_post"] = yhat_post, _["m_post"] = m_post, _["b_post"] = b_post,
                      _["sigma"] = sigma_post, _["msd"] = msd_post, _["bsd"] = bsd_post, _["b0"] = b0_post, _["b1"] = b1_post
  ));
}
