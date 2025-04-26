// Todo: what is the best order/location (cpp vs h) of includes? Can I drop any of these?
#include "funs.h"
#include "bd.h"
#include "logging.h"
#include <cmath>
#include <iostream>
#include <RcppArmadillo.h>
#include "param_draws.h"

void log_trees(std::string step, tree& t, xinfo& xi, bool verbose, Logger& logger) {
    logger.log("Attempting to print tree " + step);
    if(verbose){
        t.pr(xi);
        Rcpp::Rcout << "\n\n";
      }
}

// Desired behavior of verbose is unclear to me, since it is hardcoded originally.
void update_trees(std::string context,
                  double* allfit, double* allfit_spec, 
                  double mscale, double bscale0, double bscale1,
                  ginfo& gi, winfo& wi, bool verbose) {
    char logBuff[100];
    double* ftemp  = new double[gi.n];

    for(size_t iTree=0;iTree<wi.ntree;iTree++) {

      gi.logger.log("==================================");
      sprintf(logBuff, "%d of %d", iTree + 1, wi.ntree);
      gi.logger.log("Updating " + context + " tree: " + logBuff);
      gi.logger.log("==================================");
      gi.logger.startContext();

      log_trees("pre update", wi.t[iTree], wi.xi, verbose, gi.logger);
      
      fit(wi.t[iTree], // tree& t
          wi.xi, // xinfo& xi
          wi.di, // dinfo& di
          ftemp);
      
      log_trees("post first call to fit", wi.t[iTree], wi.xi, verbose, gi.logger);

      for(size_t k=0;k<gi.n;k++) {
        if(ftemp[k] != ftemp[k]) {
          Rcpp::Rcout << context << " tree " << iTree <<" obs "<< k <<" "<< std::endl;
          Rcpp::Rcout << wi.t[iTree] << std::endl;
          Rcpp::stop("nan in ftemp");
        }

        allfit[k]      = allfit[k]      -wi.scale_idx[k]*ftemp[k];
        allfit_spec[k] = allfit_spec[k] -wi.scale_idx[k]*ftemp[k];
        
        wi.ri[k] = (gi.y[k]-allfit[k])/wi.scale_idx[k];

        if(wi.ri[k] != wi.ri[k]) {
          Rcpp::Rcout << (gi.y[k]-allfit[k]) << std::endl;
          Rcpp::Rcout << wi.scale_idx[k] << std::endl;
          Rcpp::Rcout << wi.ri[k] << std::endl;
          Rcpp::stop("NaN in resid");
        }
      }

      if(verbose){
        gi.logger.getVectorHead(wi.weight, logBuff);
        Rcpp::Rcout << "\n weight: " <<  logBuff << "\n\n";
      } 
      gi.logger.log("Starting birth / death processing");
      gi.logger.startContext();
      bd(wi.t[iTree], // tree& x
         wi.xi, // xinfo& xi
         wi.di, // dinfo& di
         wi.weight, // phi
         wi.pi, // pinfo& pi
         gi.gen,
         gi.logger); // RNG& gen
      gi.logger.stopContext();

      log_trees("post bd", wi.t[iTree], wi.xi, verbose, gi.logger);
      if(verbose){
        log_status(gi.z_, gi.y, allfit, wi.ri, mscale, bscale0, bscale1, gi.logger);
      }

      gi.logger.log("Starting to draw mu");
      gi.logger.startContext();

      drmu(wi.t[iTree],  // tree& x
           wi.xi, // xinfo& xi
           wi.di, // dinfo& di
           wi.pi, // pinfo& pi,
           wi.weight,
           gi.gen); // RNG& gen

      gi.logger.stopContext();

      log_trees("post drmu", wi.t[iTree], wi.xi, verbose, gi.logger);

      fit(wi.t[iTree],
          wi.xi,
          wi.di,
          ftemp);

      for(size_t k=0;k<gi.n;k++) {
        allfit[k] += wi.scale_idx[k]*ftemp[k];
        allfit_spec[k] += wi.scale_idx[k]*ftemp[k];
      }

      log_trees("post second call to fit", wi.t[iTree], wi.xi, verbose, gi.logger);
      
      gi.logger.stopContext();
    }
}

void calculate_rwww(int start, int stop, double s2, double scale, double* allfit_spec, double* allfit_alt, std::vector<double>& y, double* w, double& ww, double& rw) {
    for(size_t k=start; k<stop; ++k) {
        double scale_factor = (w[k]*allfit_spec[k]*allfit_spec[k])/(s2*scale*scale);
        
        if(scale_factor!=scale_factor) {
          Rcpp::Rcout << " scale_factor " << scale_factor << endl;
          Rcpp::stop("NaN in scale factor");
        }

        // numerator is what's unexplained by the other factor
        double r = (y[k] - allfit_alt[k])*scale/allfit_spec[k];
        
        if(r!=r) {
          Rcpp::Rcout << " individual " << k << " r " << r << endl;
          Rcpp::stop("NaN in r");
        }

        ww += scale_factor;
        rw += r*scale_factor;
    }
}

void draw_scale(double& scale, double scale_prec, double ww, double rw, RNG& gen, Logger& logger, bool verbose) {
    logger.startContext();

    double scale_old = scale;
    double scale_fc_var = 1/(ww + scale_prec);
    scale = scale_fc_var*rw + gen.normal(0., 1.)*sqrt(scale_fc_var);
    if(verbose){
        Rcpp::Rcout << "Original : " << scale_old << "\n";
        Rcpp::Rcout << "scale_prec : " << scale_prec << ", ww : " << ww << ", rw : " << rw << "\n";
        Rcpp::Rcout << "New : " << scale << "\n\n";
    }
    logger.stopContext();
}

void draw_delta(std::vector<tree>& t, pinfo& pi, double& delta, RNG& gen) {
  int ntree = t.size();
  double ssq = 0.0;
  tree::npv bnv;
  typedef tree::npv::size_type bvsz;
  double endnode_count = 0.0;

  for(size_t iTree=0;iTree<ntree;iTree++) {
    bnv.clear();
    t[iTree].getbots(bnv);
    bvsz nb = bnv.size();
    for(bvsz ii = 0; ii<nb; ++ii) {
      double mm = bnv[ii]->getm(); //node parameter
      ssq += mm*mm/(pi.tau*pi.tau);
      endnode_count += 1.0;
    }
  }
  delta = gen.gamma(0.5*(1. + endnode_count), 1.0)/(0.5*(1 + ssq));
}

void update_pi(winfo& wi, Logger& logger, bool verbose) {
    if(verbose){
        logger.log("Updating pi.tau");
        Rcpp::Rcout << "Original pi.tau : " <<  wi.pi.tau << "\n";
    }
      
    wi.pi.tau = wi.sd/(sqrt(wi.delta)*sqrt((double) wi.ntree));
      
    if(verbose){
        Rcpp::Rcout << "New pi.tau : " <<  wi.pi.tau << "\n\n";
    }
}

void update_mscale(double& mscale, double sigma,
                    double* allfit_con, double* allfit_mod,
                    ginfo& gi, winfo& wi, bool verbose) {
    double ww = 0.0;
    double rw = 0.0;
    double s2 = sigma*sigma;

    calculate_rwww(0, gi.n, s2, mscale, allfit_con, allfit_mod, gi.y, gi.w, ww, rw);

    double mscale_old = mscale;
    gi.logger.log("Drawing mscale");
    draw_scale(mscale, wi.scale_prec, ww, rw, gi.gen, gi.logger, verbose);

    for(size_t k=0; k<gi.n; ++k) {
        allfit_con[k] = allfit_con[k] * mscale / mscale_old;
    }

    draw_delta(wi.t, wi.pi, wi.delta, gi.gen) ;
    
    update_pi(wi, gi.logger, verbose);
}

void update_bscale(double& bscale0, double& bscale1,
                    bool b_half_normal, double sigma,
                    double* allfit_con, double* allfit_mod,
                    ginfo& gi, winfo& wi, bool verbose) {
    double ww0 = 0.0, ww1 = 0.0;
    double rw0 = 0.0, rw1 = 0.0;
    double s2 = sigma*sigma;

    calculate_rwww(0,       gi.ntrt, s2, bscale1, allfit_mod, allfit_con, gi.y, gi.w, ww1, rw1);
    calculate_rwww(gi.ntrt, gi.n,    s2, bscale0, allfit_mod, allfit_con, gi.y, gi.w, ww0, rw0);

    double bscale0_old = bscale0;
    double bscale1_old = bscale1;
    gi.logger.log("Drawing bscale1");
    draw_scale(bscale1, wi.scale_prec, ww1, rw1, gi.gen, gi.logger, verbose);
    gi.logger.log("Drawing bscale0");
    draw_scale(bscale0, wi.scale_prec, ww0, rw0, gi.gen, gi.logger, verbose);

    for(size_t k=0; k<gi.n; ++k) {
        double scale_ratio = (k<gi.ntrt) ? bscale1/bscale1_old : bscale0/bscale0_old;
        allfit_mod[k] = allfit_mod[k]*scale_ratio;
    }

    // Not currently accesible from R - hardcoded to true
    if(!b_half_normal) {
       draw_delta(wi.t, wi.pi, wi.delta, gi.gen) ;
    }

    update_pi(wi, gi.logger, verbose);
}

void update_sigma(double* allfit, double& sigma, double nu, double lambda, double mscale, pinfo& pi_con, pinfo& pi_mod, ginfo& ginfo) {
  size_t n = ginfo.n;
  std::vector<double>& y = ginfo.y;
  double* w = ginfo.w;
  RNG& gen = ginfo.gen;
  Logger& logger = ginfo.logger;

  logger.log("Draw sigma");
  double rss = 0.0;
  double restemp = 0.0;
  for(size_t k=0;k<n;k++) {
    restemp = y[k]-allfit[k];
    rss += w[k]*restemp*restemp;
  }

  sigma = sqrt((nu*lambda + rss)/gen.chi_square(nu+n));
  pi_con.sigma = sigma/fabs(mscale);
  pi_mod.sigma = sigma;
}

void save_values(size_t& save_ctr, int n, int ntrt,
                Rcpp::NumericVector& msd_post, Rcpp::NumericVector& bsd_post, 
                Rcpp::NumericVector& b0_post, Rcpp::NumericVector& b1_post, Rcpp::NumericVector& sigma_post,
                double mscale, double bscale1, double bscale0, double sigma,
                Rcpp::NumericMatrix& m_post, Rcpp::NumericMatrix& yhat_post, Rcpp::NumericMatrix& b_post,
                double* allfit, double* allfit_con, double* allfit_mod) {

  msd_post(save_ctr) = mscale;
  bsd_post(save_ctr) = bscale1-bscale0;
  b0_post(save_ctr)  = bscale0;
  b1_post(save_ctr)  = bscale1;
  sigma_post(save_ctr) = sigma;

  for(size_t k=0;k<n;k++) {
    m_post(save_ctr, k) = allfit_con[k];
    yhat_post(save_ctr, k) = allfit[k];
    double bscale = (k<ntrt) ? bscale1 : bscale0;
    b_post(save_ctr, k) = (bscale1-bscale0)*allfit_mod[k]/bscale;
  }

  save_ctr += 1;
}