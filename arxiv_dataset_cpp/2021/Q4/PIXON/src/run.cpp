
/*
 *  PIXON
 *  A Pixon-based method for reconstructing velocity-delay map in reverberation mapping.
 * 
 *  Yan-Rong Li, liyanrong@mail.ihep.ac.cn
 * 
 */
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <random>
#include <nlopt.hpp>
#include <fftw3.h>

#include "proto.hpp"
#include "utilities.hpp"
#include "cont_model.hpp"
#include "pixon_cont.hpp"
#include "drw_cont.hpp"
#include "tnc.h"

using namespace std;

int run(Config &cfg)
{
  Data cont, line;
  cont.load(cfg.fcont);  /* load cont data */
  line.load(cfg.fline);  /* load line data */

  /* continuum reconstruction */
  cout<<"Start cont reconstruction."<<endl;
  /* time extending of reconstruction, 1% of time span */
  double text_rec = 0.1 * fmax((cont.time[cont.size-1] - cont.time[0]), (line.time[line.size-1]-line.time[0]));
  /* time backward */
  double tback = fmax(cont.time[0] - (line.time[0] - cfg.tau_range_up - text_rec), text_rec);
  /* time forward */
  double tforward = fmax((line.time[line.size-1] - cfg.tau_range_low + text_rec) - cont.time[cont.size-1], text_rec);
  double sigmad, taud, syserr;

  /* use drw to reconstruct continuum */
  cont_model = new ContModel(cont, tback, tforward, cfg.tau_interval);
  cont_model->mcmc();
  cont_model->get_best_params();
  cont_model->recon();
  taud = exp(cont_model->best_params[2]);
  sigmad = exp(cont_model->best_params[1])*sqrt(taud);
  syserr = (exp(cont_model->best_params[0]) - 1.0) * cont_model->mean_error;
  
  int npixel;  /* number of pixels */
  int npixon_size, npixon_size0; 
  int ipositive_tau; /* index of zero lag */
  double *pimg;

  pixon_sub_factor = cfg.pixon_sub_factor;
  pixon_size_factor = cfg.pixon_size_factor;
  pixon_map_low_bound = cfg.pixon_map_low_bound;
  npixon_size0 = cfg.max_pixon_size*pixon_sub_factor/pixon_size_factor;

  /* number of pixels */
  npixel = (cfg.tau_range_up - cfg.tau_range_low) / (cont_model->cont_recon.time[1]-cont_model->cont_recon.time[0]);

  /* index at which positive lags starts */
  ipositive_tau = (0.0 - cfg.tau_range_low) / (cont_model->cont_recon.time[1]-cont_model->cont_recon.time[0]);

  /* used to restore image  */
  pimg = new double[npixel+1+cont_model->cont_recon.size+1];

  /* setup pixon type */
  switch(cfg.pixon_basis_type)
  {
    case 0:  /* parabloid */      
      pixon_function = PixonBasis::parabloid;
      pixon_norm = PixonBasis::parabloid_norm;
      break;

    case 1:  /* Gaussian */
      PixonBasis::norm_gaussian = sqrt(2.0*M_PI) * erf(3.0*pixon_size_factor/sqrt(2.0));

      pixon_function = PixonBasis::gaussian;
      pixon_norm = PixonBasis::gaussian_norm;
      break;
    
    case 2: /* modified Gaussian */
      PixonBasis::coeff1_modified_gaussian = exp(-0.5 * pixon_size_factor*3.0*pixon_size_factor*3.0);
      PixonBasis::coeff2_modified_gaussian = 1.0 - PixonBasis::coeff1_modified_gaussian;
      PixonBasis::norm_gaussian = (sqrt(2.0*M_PI) * erf(3.0*pixon_size_factor/sqrt(2.0)) 
                    - 2.0*3.0*pixon_size_factor * PixonBasis::coeff1_modified_gaussian)/PixonBasis::coeff2_modified_gaussian;
      
      pixon_function = PixonBasis::modified_gaussian;
      pixon_norm = PixonBasis::modified_gaussian_norm;
      break;
    
    case 3:  /* Lorentz */ 
      pixon_function = PixonBasis::lorentz;
      pixon_norm = PixonBasis::lorentz_norm;
      break;
    
    case 4: /* Wendland */
      pixon_function = PixonBasis::wendland;
      pixon_norm = PixonBasis::wendland_norm;
      break;
    
    case 5:  /* triangle */ 
      pixon_function = PixonBasis::triangle;
      pixon_norm = PixonBasis::triangle_norm;
      break;
    
    case 6:  /* top-hat */ 
      pixon_sub_factor = 1; /* enforce to 1 */
      pixon_function = PixonBasis::tophat;
      pixon_norm = PixonBasis::tophat_norm;
      break;
    
    default:  /* default */
      PixonBasis::norm_gaussian = sqrt(2.0*M_PI) * erf(3.0*pixon_size_factor/sqrt(2.0));

      pixon_function = PixonBasis::gaussian;
      pixon_norm = PixonBasis::gaussian_norm;
      break;
  }
  
  if(cfg.drv_lc_model == 0 || cfg.drv_lc_model == 3)
  {
    /* continuum free with pixon, line with pixon 
     * resp_pixon_uniform.txt, resp_pixon.txt
     * line_pixon_uniform.txt, line_pixon.txt
     * cont_pixon_uniform.txt, cont_pixon.txt
     */
    if(cfg.pixon_uniform)
    {
      npixon_size = npixon_size0;
      run_pixon_uniform(cont, cont_model->cont_recon, line, pimg, npixel, npixon_size, ipositive_tau, cfg);
    }
    else 
    {
      npixon_size = npixon_size0;
      run_pixon(cont, cont_model->cont_recon, line, pimg, npixel, npixon_size, ipositive_tau, cfg);
    }
  }
  if(cfg.drv_lc_model == 1 || cfg.drv_lc_model == 3)
  {
    /* continuum free with drw, line with pixon 
     * resp_drw_uniform.txt, resp_drw.txt
     * line_drw_uniform.txt, line_drw.txt
     * cont_drw_uniform.txt, cont_drw.txt
     */
    if(cfg.pixon_uniform)
    {
      npixon_size = npixon_size0;
      run_drw_uniform(cont, cont_model->cont_recon, line, pimg, npixel, npixon_size, ipositive_tau, sigmad, taud, syserr, cfg);
    }
    else 
    {
      npixon_size = npixon_size0;
      run_drw(cont, cont_model->cont_recon, line, pimg, npixel, npixon_size, ipositive_tau, sigmad, taud, syserr, cfg);
    }
  }
  if(cfg.drv_lc_model == 2 || cfg.drv_lc_model == 3)
  {
    /* continuum fixed with drw, line with pixon 
     * resp_contfix_uniform.txt, resp_contfix.txt
     * line_contfix_uniform.txt line_contfix.txt 
     */
    if(cfg.pixon_uniform)
    {
      npixon_size = npixon_size0;
      run_contfix_uniform(cont_model->cont_recon, line, pimg, npixel, npixon_size, ipositive_tau, cfg);
    }
    else 
    {
      npixon_size = npixon_size0;
      run_contfix(cont_model->cont_recon, line, pimg, npixel, npixon_size, ipositive_tau, cfg);
    }
  }

  delete[] pimg;
  return 0;
}

/*
 * continuum free with drw, line with pixon
 *
 */
void run_drw(Data& cont_data, Data& cont_recon, Data& line, double *pimg, int npixel, 
                    int& npixon_size, int ipositive_tau, double sigmad, double taud, double syserr, Config& cfg)
{
  cout<<"************************************************************"<<endl;
  cout<<"Start run_drw..."<<endl;
  cout<<"npixon_size:"<<npixon_size<<endl;
  int i, iter;
  bool flag;
  PixonDRW pixon(cont_data, cont_recon, line, npixel, npixon_size, sigmad, taud, syserr, ipositive_tau, cfg.sensitivity);
  void *args = (void *)&pixon;
  double f, f_old, num, num_old, chisq, chisq_old, df, dnum;

  int ndim = npixel + 1 + cont_recon.size + 1;  /* include one parameter for background */
  /* TNC */
  int rc, maxCGit = ndim, maxnfeval = cfg.nfeval_max, nfeval, niter;
  double eta = -1.0, stepmx = -1.0, accuracy =  cfg.tol, fmin = pixon.line.size, 
    ftol = cfg.tol, xtol = cfg.tol, pgtol = cfg.tol, rescale = -1.0;
  
  /* NLopt */
  nlopt::opt opt0(nlopt::LN_BOBYQA, ndim);
  vector<double> x(ndim), g(ndim), x_old(ndim);
  vector<double>low(ndim), up(ndim);

  /* bounds and initial values */
  /* transfer function */
  for(i=0; i<npixel; i++)
  {
    low[i] = -100.0;
    up[i] =   10.0;
    x[i] = log(1.0/(npixel * pixon.dt));
  }
  /* background */
  if(cfg.fix_bg)
  {
    low[npixel] = up[npixel] = x[npixel] = cfg.bg;
  }
  else 
  {
    low[npixel] = -1.0;
    up[npixel] = 1.0;
    x[npixel] = 0.0;
  }
  /* continuum reconstruction */
  for(i=npixel+1; i<ndim; i++)
  {
    low[i] = -10.0;
    up[i] =   10.0;
    x[i] = 0.0;
  }
  
  /* initial optimization */
  opt0.set_min_objective(func_nlopt_cont_drw, args);
  opt0.set_lower_bounds(low);
  opt0.set_upper_bounds(up);
  opt0.set_maxeval(1000);
  opt0.set_ftol_abs(cfg.tol);
  opt0.set_xtol_abs(cfg.tol);
  
  opt0.optimize(x, f);
  rc = tnc(ndim, x.data(), &f, g.data(), func_tnc_cont_drw, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
  
  f_old = f;
  num_old = pixon.compute_pixon_number();
  pixon.compute_rm_pixon(x.data());
  chisq_old = pixon.compute_chisquare(x.data());
  memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  cout<<f_old<<"  "<<num_old<<"  "<<chisq_old<<endl;

  /* then pixel-dependent pixon size */
  iter = 0;
  do
  {
    iter++;
    cout<<"===================iter:"<<iter<<"==================="<<endl;

    flag = pixon.update_pixon_map();
    if(!flag)
      break;

    num = pixon.compute_pixon_number();
    
    for(i=0; i<ndim; i++)
    {
      if(x[i] < low[i])
        x[i] = low[i];
      else if(x[i] > up[i])
        x[i] = up[i];
    }

    opt0.optimize(x, f);
    rc = tnc(ndim, x.data(), &f, g.data(), func_tnc_cont_drw, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
    
    pixon.compute_rm_pixon(x.data());
    chisq = pixon.compute_chisquare(x.data());
    cout<<f<<"  "<<num<<"  "<<chisq<<endl;

    if(f <= pixon.line.size)
    {
      memcpy(x_old.data(), x.data(), ndim*sizeof(double));
      break;
    }

    num_old = num;
    f_old = f;
    chisq_old = chisq;
    memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  }while(pixon.pfft.get_ipxion_min() >= pixon_map_low_bound); 

  cout<<"bg: "<<x_old[npixel]<<endl;
  pixon.compute_rm_pixon(x_old.data());
  ofstream fout;
  string fname;
  fname = "data/resp_drw.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<pixon.dt*(i-ipositive_tau)<<"  "<<pixon.image[i]<<"  "<<exp(x_old[i])<<endl;
  }
  fout.close();

  fname = "data/line_drw.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.line.size; i++)
  {
    fout<<pixon.line.time[i]<<"  "<<pixon.itline[i]*pixon.line.norm<<"   "<<pixon.itline[i] - line.flux[i]<<endl;
  }
  fout.close();

  fname = "data/line_drw_full.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fout<<pixon.cont.time[i]<<"  "<<pixon.rmline[i]*pixon.line.norm<<endl;
  }
  fout.close();
  
  fname = "data/cont_drw.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fout<<pixon.cont.time[i]<<" "<<pixon.cont.flux[i]*pixon.cont.norm<<"  "<<pixon.cont.error[i]*pixon.cont.norm<<endl;
  }
  fout.close();
  
  fname = "data/pixon_map_drw.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<(i-ipositive_tau)*pixon.dt<<"  "<<pixon.pfft.pixon_sizes[pixon.pixon_map[i]]<<endl;
  }
  fout.close();
  memcpy(pimg, x_old.data(), ndim*sizeof(double));
}

void run_drw_uniform(Data& cont_data, Data& cont_recon, Data& line, double *pimg, int npixel, 
                  int& npixon_size, int ipositive_tau, double sigmad, double taud, double syserr, Config& cfg)
{
  cout<<"************************************************************"<<endl;
  cout<<"Start run_drw_uniform..."<<endl;
  cout<<"npixon_size:"<<npixon_size<<endl;
  int i, iter;
  bool flag;
  PixonDRW pixon(cont_data, cont_recon, line, npixel, npixon_size, sigmad, taud, syserr, ipositive_tau, cfg.sensitivity);
  void *args = (void *)&pixon;
  double f, f_old, num, num_old, chisq, chisq_old, df, dnum;

  int ndim = npixel + 1 + cont_recon.size + 1;  /* include one parameter for background */
  /* TNC */
  int rc, maxCGit = ndim, maxnfeval = cfg.nfeval_max, nfeval, niter;
  double eta = -1.0, stepmx = -1.0, accuracy =  cfg.tol, fmin = pixon.line.size, 
    ftol = cfg.tol, xtol = cfg.tol, pgtol = cfg.tol, rescale = -1.0;
  
  /* NLopt */
  nlopt::opt opt0(nlopt::LN_BOBYQA, ndim);
  vector<double> x(ndim), g(ndim), x_old(ndim);
  vector<double>low(ndim), up(ndim);

  /* bounds and initial values */
  /* transfer function */
  for(i=0; i<npixel; i++)
  {
    low[i] = -100.0;
    up[i] =   10.0;
    x[i] = log(1.0/(npixel * pixon.dt));
  }
  /* background */
  if(cfg.fix_bg)
  {
    low[npixel] = up[npixel] = x[npixel] = cfg.bg;
  }
  else 
  {
    low[npixel] = -1.0;
    up[npixel] = 1.0;
    x[npixel] = 0.0;
  }
  /* continuum reconstruction */
  for(i=npixel+1; i<ndim; i++)
  {
    low[i] = -10.0;
    up[i] =   10.0;
    x[i] = 0.0;
  }

  opt0.set_min_objective(func_nlopt_cont_drw, args);
  opt0.set_lower_bounds(low);
  opt0.set_upper_bounds(up);
  opt0.set_maxeval(1000);
  opt0.set_ftol_abs(cfg.tol);
  opt0.set_xtol_abs(cfg.tol);
  
  opt0.optimize(x, f);
  rc = tnc(ndim, x.data(), &f, g.data(), func_tnc_cont_drw, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
  
  f_old = f;
  num_old = pixon.compute_pixon_number();
  pixon.compute_rm_pixon(x.data());
  chisq_old = pixon.compute_chisquare(x.data());
  memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  cout<<f_old<<"  "<<num_old<<"  "<<chisq_old<<endl;

  iter = 0;
  while(npixon_size>pixon_map_low_bound+1)
  {
    iter++;
    cout<<"===================iter:"<<iter<<"==================="<<endl;
    npixon_size--;
    cout<<"npixon_size:"<<npixon_size<<",  size: "<<pixon.pfft.pixon_sizes[npixon_size-1]<<endl;

    pixon.reduce_pixon_map_all();
    num = pixon.compute_pixon_number();

    for(i=0; i<ndim; i++)
    {
      if(x[i] < low[i])
        x[i] = low[i];
      else if(x[i] > up[i])
        x[i] = up[i];
    }

    opt0.optimize(x, f);
    rc = tnc(ndim, x.data(), &f, g.data(), func_tnc_cont_drw, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
    
    pixon.compute_rm_pixon(x.data());
    chisq = pixon.compute_chisquare(x.data());
    cout<<f<<"  "<<num<<"  "<<chisq<<endl;

    if(f <= fmin)
    {
      memcpy(x_old.data(), x.data(), ndim*sizeof(double));
      break;
    }
    
    df = f-f_old;
    dnum = num - num_old;

    if(-df < dnum * (1.0 + cfg.sensitivity/sqrt(2.0*num)))
    {
      /* pixon size goes back to previous value */
      pixon.increase_pixon_map_all();
      break;
    }

    num_old = num;
    f_old = f;
    chisq_old = chisq;
    memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  }

  cout<<"bg: "<<x_old[npixel]<<endl;
  pixon.compute_rm_pixon(x_old.data());
  ofstream fout;
  string fname;
  fname = "data/resp_drw_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<pixon.dt*(i-ipositive_tau)<<"  "<<pixon.image[i]<<"  "<<exp(x_old[i])<<endl;
  }
  fout.close();

  fname = "data/line_drw_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.line.size; i++)
  {
    fout<<pixon.line.time[i]<<"  "<<pixon.itline[i]*pixon.line.norm<<"   "<<pixon.itline[i] - line.flux[i]<<endl;
  }
  fout.close();

  fname = "data/line_drw_uniform_full.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fout<<pixon.cont.time[i]<<"  "<<pixon.rmline[i]*pixon.line.norm<<endl;
  }
  fout.close();

  fname = "data/cont_drw_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fout<<pixon.cont.time[i]<<" "<<pixon.cont.flux[i]*pixon.cont.norm<<"  "<<pixon.cont.error[i]*pixon.cont.norm<<endl;
  }
  fout.close();
  
  fname = "data/pixon_map_drw_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<(i-ipositive_tau)*pixon.dt<<"  "<<pixon.pfft.pixon_sizes[pixon.pixon_map[i]]<<endl;
  }
  fout.close();

  memcpy(pimg, x_old.data(), ndim*sizeof(double));
}

/* set continuum free and use pixons to model continuum, pixel-dependent pixon sizes for RM */
void run_pixon(Data& cont_data, Data& cont_recon, Data& line, double *pimg, int npixel, 
                    int& npixon_size, int ipositive_tau, Config& cfg)
{
  cout<<"************************************************************"<<endl;
  cout<<"Start run_pixon..."<<endl;
  cout<<"npixon_size:"<<npixon_size<<endl;
  bool flag;
  int i, iter;
  int npixon_size_cont = 10;
  PixonCont pixon(cont_data, cont_recon, line, npixel, npixon_size, npixon_size_cont, ipositive_tau, cfg.sensitivity);
  void *args = (void *)&pixon;
  double f, f_old, num, num_old, chisq, chisq_old, df, dnum;

  /* TNC */
  int rc, maxCGit = cont_recon.size, maxnfeval = cfg.nfeval_max, nfeval, niter;
  double eta = -1.0, stepmx = -1.0, accuracy =  cfg.tol, fmin = pixon.cont_data.size, 
    ftol = cfg.tol, xtol = cfg.tol, pgtol = cfg.tol, rescale = -1.0;
  
  /* NLopt */
  nlopt::opt opt0(nlopt::LN_BOBYQA, cont_recon.size);
  vector<double> x_cont(cont_recon.size), g_cont(cont_recon.size), x_old_cont(cont_recon.size);
  vector<double> low_cont(cont_recon.size), up_cont(cont_recon.size);

  /* bounds and initial values */
  for(i=0; i<cont_recon.size; i++)
  {
    low_cont[i] = fmax(0.0, cont_recon.flux[i] - 5.0 * cont_recon.error[i]);
    up_cont[i] =            cont_recon.flux[i] + 5.0 * cont_recon.error[i];
    x_cont[i] = cont_recon.flux[i];
  }
  
  opt0.set_min_objective(func_nlopt_cont, args);
  opt0.set_lower_bounds(low_cont);
  opt0.set_upper_bounds(up_cont);
  opt0.set_maxeval(1000);
  opt0.set_ftol_abs(cfg.tol);
  opt0.set_xtol_abs(cfg.tol);
  
  opt0.optimize(x_cont, f);
  rc = tnc(cont_recon.size, x_cont.data(), &f, g_cont.data(), func_tnc_cont, args, 
      low_cont.data(), up_cont.data(), NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
  
  f_old = f;
  num_old = pixon.compute_pixon_number_cont();
  pixon.compute_cont(x_cont.data());
  chisq_old = pixon.compute_chisquare_cont(x_cont.data());
  memcpy(x_old_cont.data(), x_cont.data(), cont_recon.size*sizeof(double));
  cout<<f_old<<"  "<<num_old<<"  "<<chisq_old<<endl;
 
  while(npixon_size_cont>2)
  {
    npixon_size_cont--;
    cout<<"npixon_size_cont:"<<npixon_size_cont<<",  size: "<<pixon.pfft_cont.pixon_sizes[npixon_size_cont-1]<<endl;
    
    pixon.reduce_ipixon_cont();
    num = pixon.compute_pixon_number_cont();
    
    for(i=0; i<cont_recon.size; i++)
    {
      if(x_cont[i] < low_cont[i])
        x_cont[i] = low_cont[i];
      else if(x_cont[i] > up_cont[i])
        x_cont[i] = up_cont[i];
    }

    opt0.optimize(x_cont, f);
    rc = tnc(cont_recon.size, x_cont.data(), &f, g_cont.data(), func_tnc_cont, args, 
      low_cont.data(), up_cont.data(), NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
    
    pixon.compute_cont(x_cont.data());
    chisq = pixon.compute_chisquare_cont(x_cont.data());
    cout<<f<<"  "<<num<<"  "<<chisq<<endl;

    if(f <= fmin)
    {
      memcpy(x_old_cont.data(), x_cont.data(), cont_recon.size*sizeof(double));
      break;
    }
    
    df = f-f_old;
    dnum = num - num_old;

    if(-df < dnum * (1.0 + 1.0/sqrt(2.0*num)))
    {
      pixon.increase_ipixon_cont();
      break;
    }

    num_old = num;
    f_old = f;
    chisq_old = chisq;
    memcpy(x_old_cont.data(), x_cont.data(), cont_recon.size*sizeof(double));
  }
  
  pixon.compute_cont(x_old_cont.data());
  ofstream fp;
  fp.open("data/cont_recon_pixon.txt");
  for(i=0; i<cont_recon.size; i++)
  {
    fp<<pixon.cont.time[i]<<" "<<pixon.image_cont[i]*pixon.cont.norm<<endl;
  }
  fp.close();

  cout<<"Start to RM"<<endl;
  /* then continuum and line reverberation */
  pixon.cont.set_data(pixon.image_cont);
  /* TNC */
  int ndim = npixel + 1 + pixon.cont.size;
  maxCGit = ndim;
  fmin = pixon.line.size + pixon.cont_data.size;

  /* NLopt */
  nlopt::opt opt1(nlopt::LN_BOBYQA, ndim);
  vector<double> x(ndim), g(ndim), x_old(ndim);
  vector<double> low(ndim), up(ndim);

  /* bounds and initial values */
  /* tranfer function */
  for(i=0; i<npixel; i++)
  {
    low[i] = -100.0;
    up[i] =  10.0;
    x[i] = log(1.0/(npixel * pixon.dt));
  }
  /* background */
  if(cfg.fix_bg)
  {
    low[npixel] = up[npixel] = x[npixel] = cfg.bg;
  }
  else 
  {
    low[npixel] = -1.0;
    up[npixel] = 1.0;
    x[npixel] = 0.0;
  }
  /* continuum reconstruction */
  for(i=0; i<pixon.cont.size; i++)
  {
    low[i+npixel+1] = fmax(0.0, cont_recon.flux[i] - 5.0 * cont_recon.error[i]);
    up[i+npixel+1] =            cont_recon.flux[i] + 5.0 * cont_recon.error[i];
    x[i+npixel+1] = pixon.cont.flux[i];
  }
  
  opt1.set_min_objective(func_nlopt_cont_rm, args);
  opt1.set_lower_bounds(low);
  opt1.set_upper_bounds(up);
  opt1.set_maxeval(1000);
  opt1.set_ftol_abs(cfg.tol);
  opt1.set_xtol_abs(cfg.tol);
  
  for(i=0; i<ndim; i++)
  {
    if(x[i] < low[i])
      x[i] = low[i];
    else if(x[i] > up[i])
      x[i] = up[i];
  }

  opt1.optimize(x, f);
  rc = tnc(ndim, x.data(), &f, g.data(), func_tnc_cont_rm, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
    
  f_old = f;
  num_old = pixon.compute_pixon_number();
  pixon.compute_rm_pixon(x.data());
  chisq_old = pixon.compute_chisquare(x.data());
  memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  cout<<f_old<<"  "<<num_old<<"  "<<chisq_old<<endl;
  
  /* then pixel-dependent pixon size */
  iter = 0;
  do
  {
    iter++;
    cout<<"===================iter:"<<iter<<"==================="<<endl;
    
    flag = pixon.update_pixon_map();
    if(!flag)
      break;

    num = pixon.compute_pixon_number();
    
    for(i=0; i<ndim; i++)
    {
      if(x[i] < low[i])
        x[i] = low[i];
      else if(x[i] > up[i])
        x[i] = up[i];
    }

    opt1.optimize(x, f);
    rc = tnc(ndim, x.data(), &f, g.data(), func_tnc_cont_rm, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
    
    pixon.compute_rm_pixon(x.data());
    chisq = pixon.compute_chisquare(x.data());
    cout<<f<<"  "<<num<<"  "<<chisq<<endl;

    if(f <= fmin)
    {
      memcpy(x_old.data(), x.data(), ndim*sizeof(double));
      break;
    }

    num_old = num;
    f_old = f;
    chisq_old = chisq;
    memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  }while(pixon.pfft.get_ipxion_min() >= pixon_map_low_bound); 
  
  cout<<"bg: "<<x_old[npixel]<<endl;
  pixon.compute_rm_pixon(x_old.data());
  ofstream fout;
  string fname;
  fname = "data/resp_pixon.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<pixon.dt*(i-ipositive_tau)<<"  "<<pixon.image[i]<<"  "<<exp(x_old[i])<<endl;
  }
  fout.close();

  fname = "data/line_pixon.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.line.size; i++)
  {
    fout<<pixon.line.time[i]<<"  "<<pixon.itline[i]*pixon.line.norm<<"   "<<pixon.itline[i] - line.flux[i]<<endl;
  }
  fout.close();

  fname = "data/line_pixon_full.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fout<<pixon.cont.time[i]<<"  "<<pixon.rmline[i]*pixon.line.norm<<endl;
  }
  fout.close();
  
  fname = "data/cont_pixon.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fout<<pixon.cont.time[i]<<" "<<pixon.image_cont[i]*pixon.cont.norm<<endl;
  }
  fout.close();
  
  fname = "data/pixon_map_pixon.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<(i-ipositive_tau)*pixon.dt<<"  "<<pixon.pfft.pixon_sizes[pixon.pixon_map[i]]<<endl;
  }
  fout.close();

  memcpy(pimg, x_old.data(), ndim*sizeof(double));
}

/* set continuum free and use pixons to model continuum, uniform pixon sizes for RM */
void run_pixon_uniform(Data& cont_data, Data& cont_recon, Data& line, double *pimg, 
                            int npixel, int& npixon_size, int ipositive_tau, Config& cfg)
{
  cout<<"************************************************************"<<endl;
  cout<<"Start run_pixon_uniform..."<<endl;
  cout<<"npixon_size:"<<npixon_size<<endl;
  int i, iter;
  int npixon_size_cont = 10;
  PixonCont pixon(cont_data, cont_recon, line, npixel, npixon_size, npixon_size_cont, ipositive_tau, cfg.sensitivity);
  void *args = (void *)&pixon;
  double f, f_old, num, num_old, chisq, chisq_old, df, dnum;

  /* TNC */
  int rc, maxCGit = cont_recon.size, maxnfeval = cfg.nfeval_max, nfeval, niter;
  double eta = -1.0, stepmx = -1.0, accuracy =  cfg.tol, fmin = pixon.cont_data.size, 
    ftol = cfg.tol, xtol = cfg.tol, pgtol = cfg.tol, rescale = -1.0;
  
  /* NLopt */
  nlopt::opt opt0(nlopt::LN_BOBYQA, cont_recon.size);
  vector<double> x_cont(cont_recon.size), g_cont(cont_recon.size), x_old_cont(cont_recon.size);
  vector<double> low_cont(cont_recon.size), up_cont(cont_recon.size);

  /* bounds and initial values */
  for(i=0; i<cont_recon.size; i++)
  {
    low_cont[i] = fmax(0.0, cont_recon.flux[i] - 5.0 * cont_recon.error[i]);
    up_cont[i] =            cont_recon.flux[i] + 5.0 * cont_recon.error[i];
    x_cont[i] = cont_recon.flux[i];
  }
  
  opt0.set_min_objective(func_nlopt_cont, args);
  opt0.set_lower_bounds(low_cont);
  opt0.set_upper_bounds(up_cont);
  opt0.set_maxeval(1000);
  opt0.set_ftol_abs(cfg.tol);
  opt0.set_xtol_abs(cfg.tol);
  
  opt0.optimize(x_cont, f);
  rc = tnc(cont_recon.size, x_cont.data(), &f, g_cont.data(), func_tnc_cont, args, 
      low_cont.data(), up_cont.data(), NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
  
  f_old = f;
  num_old = pixon.compute_pixon_number_cont();
  pixon.compute_cont(x_cont.data());
  chisq_old = pixon.compute_chisquare_cont(x_cont.data());
  memcpy(x_old_cont.data(), x_cont.data(), cont_recon.size*sizeof(double));
  cout<<f_old<<"  "<<num_old<<"  "<<chisq_old<<endl;
 
  while(npixon_size_cont>2)
  {
    npixon_size_cont--;
    cout<<"npixon_size_cont:"<<npixon_size_cont<<",  size: "<<pixon.pfft_cont.pixon_sizes[npixon_size_cont-1]<<endl;
    
    pixon.reduce_ipixon_cont();
    num = pixon.compute_pixon_number_cont();
    
    for(i=0; i<cont_recon.size; i++)
    {
      if(x_cont[i] < low_cont[i])
        x_cont[i] = low_cont[i];
      else if(x_cont[i] > up_cont[i])
        x_cont[i] = up_cont[i];
    }

    opt0.optimize(x_cont, f);
    rc = tnc(cont_recon.size, x_cont.data(), &f, g_cont.data(), func_tnc_cont, args, 
      low_cont.data(), up_cont.data(), NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
    
    pixon.compute_cont(x_cont.data());
    chisq = pixon.compute_chisquare_cont(x_cont.data());
    cout<<f<<"  "<<num<<"  "<<chisq<<endl;

    if(f <= fmin)
    {
      memcpy(x_old_cont.data(), x_cont.data(), cont_recon.size*sizeof(double));
      break;
    }
    
    df = f-f_old;
    dnum = num - num_old;

    if(-df < dnum * (1.0 + 1.0/sqrt(2.0*num)))
    {
      pixon.increase_ipixon_cont();
      break;
    }

    num_old = num;
    f_old = f;
    chisq_old = chisq;
    memcpy(x_old_cont.data(), x_cont.data(), cont_recon.size*sizeof(double));
  }
  
  pixon.compute_cont(x_old_cont.data());
  ofstream fp;
  fp.open("data/cont_recon_pixon_uniform.txt");
  for(i=0; i<cont_recon.size; i++)
  {
    fp<<pixon.cont.time[i]<<" "<<pixon.image_cont[i]*pixon.cont.norm<<endl;
  }
  fp.close();

  cout<<"Start to RM"<<endl;
  /* then continuum and line reverberation */
  pixon.cont.set_data(pixon.image_cont);
  /* TNC */
  int ndim = npixel + 1 + pixon.cont.size;
  maxCGit = ndim;
  fmin = pixon.line.size + pixon.cont_data.size;

  /* NLopt */
  nlopt::opt opt1(nlopt::LN_BOBYQA, ndim);
  vector<double> x(ndim), g(ndim), x_old(ndim);
  vector<double> low(ndim), up(ndim);

  /* bounds and initial values */
  /* transfer function */
  for(i=0; i<npixel; i++)
  {
    low[i] = -100.0;
    up[i] =  10.0;
    x[i] = log(1.0/(npixel * pixon.dt));
  }
  /* background */
  if(cfg.fix_bg)
  {
    low[npixel] = up[npixel] = x[npixel] = cfg.bg;
  }
  else 
  {
    low[npixel] = -1.0;
    up[npixel] = 1.0;
    x[npixel] = 0.0;
  }
  /* continuuum reconstruction */
  for(i=0; i<pixon.cont.size; i++)
  {
    low[i+npixel+1] = fmax(0.0, cont_recon.flux[i] - 5.0 * cont_recon.error[i]);
    up[i+npixel+1] =            cont_recon.flux[i] + 5.0 * cont_recon.error[i];
    x[i+npixel+1] = pixon.cont.flux[i];
  }
  
  opt1.set_min_objective(func_nlopt_cont_rm, args);
  opt1.set_lower_bounds(low);
  opt1.set_upper_bounds(up);
  opt1.set_maxeval(1000);
  opt1.set_ftol_abs(cfg.tol);
  opt1.set_xtol_abs(cfg.tol);
  
  for(i=0; i<ndim; i++)
  {
    if(x[i] < low[i])
      x[i] = low[i];
    else if(x[i] > up[i])
      x[i] = up[i];
  }

  opt1.optimize(x, f);
  rc = tnc(ndim, x.data(), &f, g.data(), func_tnc_cont_rm, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
    
  f_old = f;
  num_old = pixon.compute_pixon_number();
  pixon.compute_rm_pixon(x.data());
  chisq_old = pixon.compute_chisquare(x.data());
  memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  cout<<f_old<<"  "<<num_old<<"  "<<chisq_old<<endl;
  
  iter = 0;
  while(npixon_size>pixon_map_low_bound+1)
  {
    iter++;
    cout<<"===================iter:"<<iter<<"==================="<<endl;
    npixon_size--;
    cout<<"npixon_size:"<<npixon_size<<",  size: "<<pixon.pfft.pixon_sizes[npixon_size-1]<<endl;

    pixon.reduce_pixon_map_all();
    num = pixon.compute_pixon_number();

    for(i=0; i<ndim; i++)
    {
      if(x[i] < low[i])
        x[i] = low[i];
      else if(x[i] > up[i])
        x[i] = up[i];
    }

    opt1.optimize(x, f);
    rc = tnc(ndim, x.data(), &f, g.data(), func_tnc_cont_rm, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
    
    pixon.compute_rm_pixon(x.data());
    chisq = pixon.compute_chisquare(x.data());
    cout<<f<<"  "<<num<<"  "<<chisq<<endl;

    if(f <= fmin)
    {
      memcpy(x_old.data(), x.data(), ndim*sizeof(double));
      break;
    }
    
    df = f-f_old;
    dnum = num - num_old;

    if(-df < dnum * (1.0 + cfg.sensitivity/sqrt(2.0*num)))
    {
      /* pixon size goes back to previous value */
      pixon.increase_pixon_map_all();
      break;
    }

    num_old = num;
    f_old = f;
    chisq_old = chisq;
    memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  }
  
  cout<<"bg: "<<x_old[npixel]<<endl;
  pixon.compute_rm_pixon(x_old.data());
  ofstream fout;
  string fname;
  fname = "data/resp_pixon_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<pixon.dt*(i-ipositive_tau)<<"  "<<pixon.image[i]<<"   "<<exp(x_old[i])<<endl;
  }
  fout.close();

  fname = "data/line_pixon_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.line.size; i++)
  {
    fout<<pixon.line.time[i]<<"  "<<pixon.itline[i]*pixon.line.norm<<"   "<<pixon.itline[i] - line.flux[i]<<endl;
  }
  fout.close();

  fname = "data/line_pixon_uniform_full.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fout<<pixon.cont.time[i]<<"  "<<pixon.rmline[i]*pixon.line.norm<<endl;
  }
  fout.close();
  
  fname = "data/cont_pixon_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fp.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fp<<pixon.cont.time[i]<<" "<<pixon.image_cont[i]*pixon.cont.norm<<endl;
  }
  fp.close();

  fname = "data/pixon_map_pixon_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<(i-ipositive_tau)*pixon.dt<<"  "<<pixon.pfft.pixon_sizes[pixon.pixon_map[i]]<<endl;
  }
  fout.close();

  memcpy(pimg, x_old.data(), ndim*sizeof(double));
}

/* set continuum fixed from a drw reconstruction and use pixel dependent pixon sizes for RM */
void run_contfix(Data& cont, Data& line, double *pimg, int npixel, int& npixon_size, int ipositive_tau, Config& cfg)
{
  cout<<"************************************************************"<<endl;
  cout<<"Start run_contfix..."<<endl;
  cout<<"npixon_size:"<<npixon_size<<endl;
  int i, iter;
  Pixon pixon(cont, line, npixel, npixon_size, ipositive_tau, cfg.sensitivity);
  void *args = (void *)&pixon;
  bool flag;
  double f, f_old, num, num_old, chisq, chisq_old, df, dnum;
  
  int ndim = npixel + 1;  /* include one parameter for background */
  /* TNC */
  int rc, maxCGit = ndim, maxnfeval = cfg.nfeval_max, nfeval, niter;
  double eta = -1.0, stepmx = -1.0, accuracy =  cfg.tol, fmin = pixon.line.size, 
    ftol = cfg.tol, xtol = cfg.tol, pgtol = cfg.tol, rescale = -1.0;
  
  /* NLopt */
  nlopt::opt opt0(nlopt::LN_BOBYQA, ndim);
  vector<double> x(ndim), g(ndim), x_old(ndim);
  vector<double>low(ndim), up(ndim);

  /* bounds and initial values */
  /* transfer function */
  for(i=0; i<npixel; i++)
  {
    low[i] = -100.0;
    up[i] =  10.0;
    x[i] = log(1.0/(npixel * pixon.dt));
  }
  /* background */
  if(cfg.fix_bg)
  {
    low[npixel] = up[npixel] = x[npixel] = cfg.bg;
  }
  else 
  {
    low[npixel] = -1.0;
    up[npixel] = 1.0;
    x[npixel] = 0.0;
  }

  opt0.set_min_objective(func_nlopt, args);
  opt0.set_lower_bounds(low);
  opt0.set_upper_bounds(up);
  opt0.set_maxeval(1000);
  opt0.set_ftol_abs(cfg.tol);
  opt0.set_xtol_abs(cfg.tol);
  
  opt0.optimize(x, f);
  rc = tnc(ndim, x.data(), &f, g.data(), func_tnc, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
  
  f_old = f;
  num_old = pixon.compute_pixon_number();
  pixon.compute_rm_pixon(x.data());
  chisq_old = pixon.compute_chisquare(x.data());
  memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  cout<<f_old<<"  "<<num_old<<"  "<<chisq_old<<endl;

  /* then pixel-dependent pixon size */
  iter = 0;
  do
  {
    iter++;
    cout<<"===================iter:"<<iter<<"==================="<<endl;
    
    flag = pixon.update_pixon_map();
    if(!flag)
      break;

    num = pixon.compute_pixon_number();
    
    for(i=0; i<ndim; i++)
    {
      if(x[i] < low[i])
        x[i] = low[i];
      else if(x[i] > up[i])
        x[i] = up[i];
    }

    opt0.optimize(x, f);
    rc = tnc(ndim, x.data(), &f, g.data(), func_tnc, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
    
    pixon.compute_rm_pixon(x.data());
    chisq = pixon.compute_chisquare(x.data());
    cout<<f<<"  "<<num<<"  "<<chisq<<endl;

    if(f <= pixon.line.size)
    {
      memcpy(x_old.data(), x.data(), ndim*sizeof(double));
      break;
    }

    df = f-f_old;
    dnum = num - num_old;   

    num_old = num;
    f_old = f;
    chisq_old = chisq;
    memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  }while(pixon.pfft.get_ipxion_min() >= pixon_map_low_bound); 

  cout<<"bg: "<<x_old[npixel]<<endl;
  
  pixon.compute_rm_pixon(x_old.data());
  ofstream fout;
  string fname;
  fname = "data/resp_contfix.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<pixon.dt*(i-ipositive_tau)<<"  "<<pixon.image[i]<<"   "<<exp(x_old[i])<<endl;
  }
  fout.close();
  
  fname = "data/line_contfix.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.line.size; i++)
  {
    fout<<pixon.line.time[i]<<"  "<<pixon.itline[i]*pixon.line.norm<<"   "<<pixon.itline[i] - line.flux[i]<<endl;
  }
  fout.close();

  fname = "data/line_contfix_full.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fout<<pixon.cont.time[i]<<"  "<<pixon.rmline[i]*pixon.line.norm<<endl;
  }
  fout.close();

  fname = "data/pixon_map_contfix.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<(i-ipositive_tau)*pixon.dt<<"  "<<pixon.pfft.pixon_sizes[pixon.pixon_map[i]]<<endl;
  }
  fout.close();
  
  memcpy(pimg, x_old.data(), ndim*sizeof(double));
}

/* set continuum fixed from a drw reconstruction and use uniform pixon sizes for RM */
void run_contfix_uniform(Data& cont, Data& line, double *pimg, int npixel, int& npixon_size, int ipositive_tau, Config& cfg)
{
  cout<<"************************************************************"<<endl;
  cout<<"Start run_contfix_uniform..."<<endl;
  cout<<"npixon_size:"<<npixon_size<<endl;
  int i;
  Pixon pixon(cont, line, npixel, npixon_size, ipositive_tau, cfg.sensitivity);
  void *args = (void *)&pixon;
  double f, f_old, num_old, num, df, dnum, chisq, chisq_old;
  int iter;
 
  int ndim = npixel + 1;
  /* TNC */
  int rc, maxCGit = ndim, maxnfeval = cfg.nfeval_max, nfeval, niter;
  double eta = -1.0, stepmx = -1.0, accuracy = cfg.tol, fmin = pixon.line.size, 
    ftol = cfg.tol, xtol = cfg.tol, pgtol = cfg.tol, rescale = -1.0;

  /* NLopt */
  nlopt::opt opt0(nlopt::LN_BOBYQA, ndim);
  vector<double> x(ndim), g(ndim), x_old(ndim);
  vector<double> low(ndim), up(ndim);
  
  /* bounds and initial values */
  /* transfer function */
  for(i=0; i<npixel; i++)
  {
    low[i] = -100.0;
    up[i] =  10.0;
    x[i] = log(1.0/(npixel * pixon.dt));
  }
  /* background */
  if(cfg.fix_bg)
  {
    low[npixel] = up[npixel] = x[npixel] = cfg.bg;
  }
  else 
  {
    low[npixel] = -1.0;
    up[npixel] = 1.0;
    x[npixel] = 0.0;
  }

  /* NLopt settings */
  opt0.set_min_objective(func_nlopt, args);
  opt0.set_lower_bounds(low);
  opt0.set_upper_bounds(up);
  opt0.set_maxeval(1000);
  opt0.set_ftol_abs(cfg.tol);
  opt0.set_xtol_abs(cfg.tol);
   
  opt0.optimize(x, f);
  rc = tnc(ndim, x.data(), &f, g.data(), func_tnc, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
  
  f_old = f;
  num_old = pixon.compute_pixon_number();
  pixon.compute_rm_pixon(x.data());
  chisq_old = pixon.compute_chisquare(x.data());
  memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  cout<<f_old<<"  "<<num_old<<"  "<<chisq_old<<endl;

  iter = 0;
  while(npixon_size>pixon_map_low_bound+1)
  {
    iter++;
    cout<<"===================iter:"<<iter<<"==================="<<endl;
    npixon_size--;
    cout<<"npixon_size:"<<npixon_size<<",  size: "<<pixon.pfft.pixon_sizes[npixon_size-1]<<endl;

    pixon.reduce_pixon_map_all();
    num = pixon.compute_pixon_number();
    
    for(i=0; i<ndim; i++)
    {
      if(x[i] < low[i])
        x[i] = low[i];
      else if(x[i] > up[i])
        x[i] = up[i];
    }

    opt0.optimize(x, f);
    rc = tnc(ndim, x.data(), &f, g.data(), func_tnc, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
    
    pixon.compute_rm_pixon(x.data());
    chisq = pixon.compute_chisquare(x.data());
    cout<<f<<"  "<<num<<"  "<<chisq<<endl;

    if(f <= pixon.line.size)
    {
      memcpy(x_old.data(), x.data(), ndim*sizeof(double));
      break;
    }
    
    df = f-f_old;
    dnum = num - num_old;

    if(-df < dnum * (1.0 + cfg.sensitivity/sqrt(2.0*num)))
    {
      /* pixon size goes back to previous value */
      pixon.increase_pixon_map_all();
      break;
    }

    num_old = num;
    f_old = f;
    chisq_old = chisq;
    memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  }
  
  cout<<"bg: "<<x_old[npixel]<<endl;
  
  pixon.compute_rm_pixon(x_old.data());
  ofstream fout;
  string fname;
  fname = "data/resp_contfix_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<pixon.dt*(i-ipositive_tau)<<"   "<<pixon.image[i]<<"  "<<exp(x_old[i])<<endl;
  }
  fout.close();
  
  fname = "data/line_contfix_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.line.size; i++)
  {
    fout<<pixon.line.time[i]<<"  "<<pixon.itline[i]*pixon.line.norm<<"   "<<pixon.itline[i] - line.flux[i]<<endl;
  }
  fout.close();

  fname = "data/line_contfix_uniform_full.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fout<<pixon.cont.time[i]<<"  "<<pixon.rmline[i]*pixon.line.norm<<endl;
  }
  fout.close();
  
  fname = "data/pixon_map_contfix_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<(i-ipositive_tau)*pixon.dt<<"  "<<pixon.pfft.pixon_sizes[pixon.pixon_map[i]]<<endl;
  }
  fout.close();

  memcpy(pimg, x_old.data(), ndim*sizeof(double));

}