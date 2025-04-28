/*
 *  PIXON
 *  A Pixon-based method for reconstructing velocity-delay map in reverberation mapping.
 * 
 *  Yan-Rong Li, liyanrong@mail.ihep.ac.cn
 * 
 */

#include <iostream>
#include <fstream> 
#include <vector>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <random>
#include <float.h>

/* dnest header file */
#include <dnestvars.h>

#include "utilities.hpp"
#include "cont_model.hpp"


using namespace std;

ContModel* cont_model;

double prob_cont(const void *model, const void *arg)
{
  double prob = 0.0;
  int i, info;
  double *pm = (double *)model;
  double tau, sigma2, lndet, syserr;
  double *Lbuf, *ybuf, *y, *yq, *Cq, *W, *D, *phi;
  int nq = cont_model->nq;
  Data & cont = cont_model->cont;
  double * workspace = cont_model->workspace;
  double * Larr_data = cont_model->Larr_data;
  
  syserr = (exp(pm[0])-1.0)*cont_model->mean_error;
  tau = exp(pm[2]);
  sigma2 = exp(2.0*pm[1]) * tau;

  Lbuf = workspace;
  ybuf = Lbuf + cont.size*nq;
  y = ybuf +  cont.size;
  yq = y +  cont.size;
  Cq = yq + nq;
  W = Cq + nq*nq;
  D = W +  cont.size;
  phi = D +  cont.size;

  compute_semiseparable_drw(cont.time, cont.size, sigma2, 1.0/tau, cont.error, syserr, W, D, phi);
  lndet = 0.0;
  for(i=0; i<cont.size; i++)
  lndet += log(D[i]);
 
  /* calculate L^T*C^-1*L */
  multiply_mat_semiseparable_drw(Larr_data, W, D, phi, cont.size, nq, sigma2, Lbuf);
  multiply_mat_MN_transposeA(Larr_data, Lbuf, Cq, nq, nq, cont.size);

  /* calculate L^T*C^-1*y */
  multiply_matvec_semiseparable_drw(cont.flux, W, D, phi, cont.size, sigma2, ybuf);
  multiply_mat_MN_transposeA(Larr_data, ybuf, yq, nq, 1, cont.size);

  inverse_pomat(Cq, nq, &info);
  multiply_mat_MN(Cq, yq, ybuf, nq, 1, nq);

  Chol_decomp_L(Cq, nq, &info);
  multiply_matvec(Cq, &pm[3], nq, yq);
  for(i=0; i<nq; i++)
    yq[i] += ybuf[i];
  
  multiply_matvec_MN(Larr_data, cont.size, nq, yq, ybuf);
  for(i=0; i<cont.size; i++)
  {
    y[i] = cont.flux[i] - ybuf[i];
  }
  
  /* y^T x C^-1 x y*/
  multiply_matvec_semiseparable_drw(y, W, D, phi, cont.size, sigma2, ybuf);
  for(i=0; i<cont.size; i++)
  {
    prob += y[i] * ybuf[i];
  }
  prob *= -0.5;
  prob += -0.5*lndet;
  return prob;
}

void from_prior_cont(void *model, const void *arg)
{
  int i;
  double *pm = (double *)model;

  for(i=0; i<cont_model->num_params; i++)
  {
    if(cont_model->par_prior_model[i] == GAUSSIAN )
    {
      pm[i] = dnest_randn() * cont_model->par_prior_gaussian[i][1] + cont_model->par_prior_gaussian[i][0];
      dnest_wrap(&pm[i], cont_model->par_range_model[i][0], cont_model->par_range_model[i][1]);
    }
    else
    {
      pm[i] = cont_model->par_range_model[i][0] + dnest_rand()*(cont_model->par_range_model[i][1] - cont_model->par_range_model[i][0]);
    }
  }

  for(i=0; i<cont_model->num_params; i++)
  {
    if(cont_model->par_fix[i] == 1)
      pm[i] = cont_model->par_fix_val[i];
  }
}

void print_particle_cont(FILE *fp, const void *model, const void *arg)
{
  int i;
  double *pm = (double *)model;

  for(i=0; i<cont_model->num_params; i++)
  {
    fprintf(fp, "%e ", pm[i] );
  }
  fprintf(fp, "\n");
}

double perturb_cont(void *model, const void *arg)
{
  double *pm = (double *)model;
  double logH = 0.0, limit1, limit2, width, rnd;
  int which, which_level;
  
  /* sample variability parameters more frequently */
  do
  {
    which = dnest_rand_int(cont_model->num_params);
  }while(cont_model->par_fix[which] == 1);
 
  width = ( cont_model->par_range_model[which][1] - cont_model->par_range_model[which][0] );


  if(cont_model->par_prior_model[which] == GAUSSIAN)
  {
    logH -= (-0.5*pow((pm[which] - cont_model->par_prior_gaussian[which][0])/cont_model->par_prior_gaussian[which][1], 2.0) );
    pm[which] += dnest_randh() * width;
    dnest_wrap(&pm[which], cont_model->par_range_model[which][0], cont_model->par_range_model[which][1]);
    logH += (-0.5*pow((pm[which] - cont_model->par_prior_gaussian[which][0])/cont_model->par_prior_gaussian[which][1], 2.0) );
  }
  else
  {
    pm[which] += dnest_randh() * width;
    dnest_wrap(&(pm[which]), cont_model->par_range_model[which][0], cont_model->par_range_model[which][1]);
  }
  return logH;
}

/*===========================================================*/
ContModel::ContModel()
{
  num_params = 0;
  workspace = NULL;
  workspace_uv = NULL;
  Larr_data = NULL;
  par_range_model = NULL;
  par_fix = NULL;
  par_fix_val = NULL;
  par_prior_model = NULL;
  par_prior_model = NULL;
  best_params = NULL;
  best_params_std = NULL;
  USmat = NULL;
  PEmat1 = NULL;
  PEmat2 = NULL;
  PSmat = NULL;

  dnest_free_fptrset(fptrset);
}
ContModel::ContModel(Data& cont_in, double tback, double tforward, double tau_interval)
  :cont(cont_in)
{
  int i;
  nq = 1;
  compute_mean_error();

  /* continuum reconstruction */
  int n;
  double dt = (cont.time[cont.size-1] - cont.time[0])/(cont.size-1);

  dt = fmin(dt, tau_interval);
  n = (cont.time[cont.size - 1] + tforward - (cont.time[0] - tback))/dt;
  cout<<"size: "<<n<<endl;
  cont_recon.set_size(n);
  cont_recon.set_norm(cont_in.norm);
  for(i=0; i<cont_recon.size; i++)
  {
    cont_recon.time[i] = cont.time[0] - tback + dt * i;
  }
  size_max = fmax(cont.size, cont_recon.size);
  workspace = new double[size_max*15];
  workspace_uv = new double[2*cont_recon.size];
  Larr_data = new double[cont.size*nq];
  for(i=0; i<cont.size; i++)
  {
    Larr_data[i*nq+0] = 1.0;
  }
  USmat = new double [cont.size * cont_recon.size];
  PEmat1 = new double [cont.size * cont_recon.size];
  PEmat2 = new double [cont_recon.size * cont_recon.size];
  PSmat = new double [cont_recon.size * cont_recon.size];

  /* dnest configuration */
  num_params_drw = 3; /* syserr, sigma, and tau */
  num_params_var = num_params_drw + nq;
  num_params = num_params_var;
  par_range_model = new double * [num_params];
  for(i=0; i<num_params; i++)
  {
    par_range_model[i] = new double [2];
  }
  par_fix = new int [num_params];
  par_fix_val = new double [num_params];

  par_prior_model = new int [num_params];
  par_prior_gaussian = new double * [num_params];
  for(i=0; i<num_params; i++)
  {
    par_prior_gaussian[i] = new double [2];
  }

  i=0; /* systematic error */
  par_range_model[i][0] = log(1.0);
  par_range_model[i][1] = log(1.0 + 10.0);
  par_prior_model[i] = UNIFORM;
  par_prior_gaussian[i][0] = 0.0;
  par_prior_gaussian[i][1] = 0.0;

  i=1; /* sigma */
  par_range_model[i][0] = log(1.0e-6);
  par_range_model[i][1] = log(1.0);
  par_prior_model[i] = UNIFORM;
  par_prior_gaussian[i][0] = 0.0;
  par_prior_gaussian[i][1] = 0.0;
  
  i=2; /* tau */
  par_range_model[i][0] = log(1.0);
  par_range_model[i][1] = log(1.0e4);
  par_prior_model[i] = UNIFORM;
  par_prior_gaussian[i][0] = 0.0;
  par_prior_gaussian[i][1] = 0.0;

  i=3; /*  q */
  par_range_model[i][0] = -5.0;
  par_range_model[i][1] = 5.0;
  par_prior_model[i] = UNIFORM;
  par_prior_gaussian[i][0] = 0.0;
  par_prior_gaussian[i][1] = 0.0;

  for(i=0; i<num_params; i++)
  {
    par_fix[i] = 0;
    par_fix_val[i] = -DBL_MAX;
  }
  par_fix[0] = 1;
  par_fix_val[0] = log(1.0);

  best_params = new double [num_params];
  best_params_std = new double [num_params];

  uniform_dist = new uniform_real_distribution<double>(0.0, 1.0);
  normal_dist = new normal_distribution<double>(0.0, 1.0);

  fptrset = dnest_malloc_fptrset();
  fptrset->from_prior = from_prior_cont;
  fptrset->perturb = perturb_cont;
  fptrset->print_particle = print_particle_cont;
  fptrset->log_likelihoods_cal = prob_cont;
}

ContModel::~ContModel()
{
  int i;

  num_params = 0;
  delete[] workspace;
  delete[] workspace_uv;
  delete[] Larr_data;
  delete[] USmat;
  delete[] PEmat1;
  delete[] PEmat2;
  delete[] PSmat;

  for(i=0; i<num_params; i++)
  {
    delete[] par_range_model[i];
    delete[] par_prior_gaussian[i];
  }
  delete[] par_range_model;
  delete[] par_prior_gaussian;
  delete[] best_params;
  delete[] best_params_std;
}

void ContModel::compute_mean_error()
{
  int i;
  mean_error = 0.0;
  for(i=0; i<cont.size; i++)
  {
    mean_error += cont.error[i];
  }
  mean_error /= cont.size;
}

void ContModel::mcmc()
{
  int i, argc=0;
  char **argv;
  double logz_con;

  argv = new char * [9];
  for(i=0; i<9; i++)
  {
    argv[i] = new char [256];
  }
  
  strcpy(argv[argc++], "dnest");
  strcpy(argv[argc++], "-s");
  strcpy(argv[argc], "./");
  strcat(argv[argc++], "/data/restart_dnest.txt");

  logz_con = dnest(argc, argv, fptrset, num_params, "data/", 1000, 0.1, NULL);

  for(i=0; i<9; i++)
  {
    delete[] argv[i];
  }
  delete[] argv;
}

void ContModel::get_best_params()
{
  int i, j, num_ps;
  FILE *fp;
  char posterior_sample_file[256];
  double *post_model, *posterior_sample;
  double *pm, *pmstd;

  strcpy(posterior_sample_file, "data/posterior_sample.txt");

  /* open file for posterior sample */
  fp = fopen(posterior_sample_file, "r");
  if(fp == NULL)
  {
    fprintf(stderr, "# Error: Cannot open file %s.\n", posterior_sample_file);
    exit(0);
  }

  /* read number of points in posterior sample */
  if(fscanf(fp, "# %d", &num_ps) < 1)
  {
    fprintf(stderr, "# Error: Cannot read file %s.\n", posterior_sample_file);
    exit(0);
  }
  printf("# Number of points in posterior sample: %d\n", num_ps);

  post_model = new double[num_params];
  posterior_sample = new double[num_ps * num_params];
  
  for(i=0; i<num_ps; i++)
  {
    for(j=0; j<num_params; j++)
    {
      if(fscanf(fp, "%lf", (double *)post_model + j) < 1)
      {
        fprintf(stderr, "# Error: Cannot read file %s.\n", posterior_sample_file);
        exit(0);
      }
    }
    fscanf(fp, "\n");

    memcpy(posterior_sample+i*num_params, post_model, num_params*sizeof(double));

  }
  fclose(fp);

  /* calcaulte mean and standard deviation of posterior samples. */
  pm = (double *)best_params;
  pmstd = (double *)best_params_std;
  for(j=0; j<num_params; j++)
  {
    pm[j] = pmstd[j] = 0.0;
  }
  for(i=0; i<num_ps; i++)
  {
    for(j =0; j<num_params; j++)
      pm[j] += *((double *)posterior_sample + i*num_params + j );
  }

  for(j=0; j<num_params; j++)
    pm[j] /= num_ps;

  for(i=0; i<num_ps; i++)
  {
    for(j=0; j<num_params; j++)
      pmstd[j] += pow( *((double *)posterior_sample + i*num_params + j ) - pm[j], 2.0 );
  }

  for(j=0; j<num_params; j++)
  {
    if(num_ps > 1)
      pmstd[j] = sqrt(pmstd[j]/(num_ps-1.0));
    else
      pmstd[j] = 0.0;
  }  

  for(j = 0; j<num_params; j++)
    printf("Best params %d %f +- %f\n", j, *((double *)best_params + j), 
                                           *((double *)best_params_std + j) ); 
  
  /* calculate the median values */
  double *param_buf;
  param_buf = new double [num_ps];
  for(j=0; j<num_params; j++)
  {
    for(i=0; i<num_ps; i++)
    {
      param_buf[i] = *((double *)posterior_sample + i*num_params + j );
    }
    qsort(param_buf, num_ps, sizeof(double), compare);
    *((double *)best_params + j) = param_buf[num_ps/2];
    printf("meidan: %f\n", param_buf[num_ps/2]);
  }

  delete[] param_buf;
  delete[] post_model;
  delete[] posterior_sample;
}

void ContModel::recon()
{
  double *Lbuf, *ybuf, *y, *Cq, *yq, *W, *D, *phi, *u, *v;
  double syserr;

  double *pm = (double *)best_params;
  double sigma, sigma2, tau, alpha;
  int i, info;

  syserr = (exp(pm[0]) - 1.0) * mean_error;  // systematic error 
  tau = exp(pm[2]);
  sigma = exp(pm[1]) * sqrt(tau);
  sigma2 = sigma*sigma;
  alpha = 1.0;
  
  Lbuf = workspace;
  ybuf = Lbuf + cont.size*nq; 
  y = ybuf + size_max;
  Cq = y + size_max;
  yq = Cq + nq*nq;
  W = yq + nq;
  D = W + size_max;
  phi = D + size_max;
  u = phi + size_max;
  v = u + cont_recon.size;

  compute_semiseparable_drw(cont.time, cont.size, sigma2, 1.0/tau, cont.error, syserr, W, D, phi);
  // Cq^-1 = L^TxC^-1xL
  multiply_mat_semiseparable_drw(Larr_data, W, D, phi, cont.size, nq, sigma2, Lbuf);
  multiply_mat_MN_transposeA(Larr_data, Lbuf, Cq, nq, nq, cont.size);

  // L^TxC^-1xy
  multiply_matvec_semiseparable_drw(cont.flux, W, D, phi, cont.size, sigma2, ybuf);
  multiply_mat_MN_transposeA(Larr_data, ybuf, yq, nq, 1, cont.size);

  // (hat q) = Cqx(L^TxC^-1xy)
  inverse_pomat(Cq, nq, &info);
  multiply_mat_MN(Cq, yq, ybuf, nq, 1, nq);

  // q = uq + (hat q)
  Chol_decomp_L(Cq, nq, &info);
  multiply_matvec(Cq, &pm[num_params_drw], nq, yq);
  for(i=0; i<nq; i++)
    yq[i] += ybuf[i];
  
  // y = yc - Lxq
  multiply_matvec_MN(Larr_data, cont.size, nq, yq, ybuf);
  for(i=0; i<cont.size; i++)
  {
    y[i] = cont.flux[i] - ybuf[i];
  }
  
  set_covar_Umat(sigma, tau, alpha);
  // (hat s) = SxC^-1xy
  multiply_matvec_semiseparable_drw(y, W, D, phi, cont.size, sigma2, ybuf);
  multiply_matvec_MN(USmat, cont_recon.size, cont.size, ybuf, cont_recon.flux);

  // SxC^-1xS^T
  multiply_mat_transposeB_semiseparable_drw(USmat, W, D, phi, cont.size, cont_recon.size, sigma2, PEmat1);
  multiply_mat_MN(USmat, PEmat1, PEmat2, cont_recon.size, cont_recon.size, cont.size);

  for(i=0; i<cont_recon.size; i++)
  {
    cont_recon.error[i] = sqrt(sigma2 + syserr*syserr - PEmat2[i*cont_recon.size + i]);
  }

  //set_covar_Pmat(sigma, tau, alpha);
  //for(i=0; i<cont_recon.size * cont_recon.size; i++)
  //{
  //  PSmat[i] = PSmat[i] - PEmat2[i];
  //}
  //for(i=0; i<cont_recon.size; i++)
  //{
  //  cont_recon.error[i] = sqrt(PSmat[i*cont_recon.size + i]);
  //}

  for(i=0; i<cont_recon.size; i++)
  {
    cont_recon.flux[i] += yq[0];
  }

  ofstream fout;
  fout.open("data/cont_recon_drw.txt");
  for(i=0; i<cont_recon.size; i++)
  {
    fout<<cont_recon.time[i]<<"  "<<cont_recon.flux[i]*cont_recon.norm<<"   "<<cont_recon.error[i]*cont_recon.norm<<endl;
  }
  fout.close();
}

/* 
 * covariance matrix 
 * Q = [S^-1 + N^-1] = Nx[S+N]^-1xS
 * with reconstructed errors N
 * 
 */
void ContModel::recon(const void *model)
{
  double *Lbuf, *ybuf, *y, *Cq, *yq, *W, *D, *phi, *u, *v;
  double syserr;

  double *pm = (double *)model;
  double sigma, sigma2, tau, alpha;
  int i, j, info;

  syserr = (exp(pm[0]) - 1.0) * mean_error;  // systematic error 
  tau = exp(pm[2]);
  sigma = exp(pm[1]) * sqrt(tau);
  sigma2 = sigma*sigma;
  alpha = 1.0;
  
  Lbuf = workspace;
  ybuf = Lbuf + cont.size*nq; 
  y = ybuf + size_max;
  Cq = y + size_max;
  yq = Cq + nq*nq;
  W = yq + nq;
  D = W + size_max;
  phi = D + size_max;
  u = phi + size_max;
  v = u + cont_recon.size;

  compute_semiseparable_drw(cont.time, cont.size, sigma2, 1.0/tau, cont.error, syserr, W, D, phi);
  // Cq^-1 = L^TxC^-1xL
  multiply_mat_semiseparable_drw(Larr_data, W, D, phi, cont.size, nq, sigma2, Lbuf);
  multiply_mat_MN_transposeA(Larr_data, Lbuf, Cq, nq, nq, cont.size);

  // L^TxC^-1xy
  multiply_matvec_semiseparable_drw(cont.flux, W, D, phi, cont.size, sigma2, ybuf);
  multiply_mat_MN_transposeA(Larr_data, ybuf, yq, nq, 1, cont.size);

  // (hat q) = Cqx(L^TxC^-1xy)
  inverse_pomat(Cq, nq, &info);
  multiply_mat_MN(Cq, yq, ybuf, nq, 1, nq);

  // q = uq + (hat q)
  Chol_decomp_L(Cq, nq, &info);
  multiply_matvec(Cq, &pm[num_params_drw], nq, yq);
  for(i=0; i<nq; i++)
    yq[i] += ybuf[i];
  
  // y = yc - Lxq
  multiply_matvec_MN(Larr_data, cont.size, nq, yq, ybuf);
  for(i=0; i<cont.size; i++)
  {
    y[i] = cont.flux[i] - ybuf[i];
  }
  
  set_covar_Umat(sigma, tau, alpha);
  // (hat s) = SxC^-1xy
  multiply_matvec_semiseparable_drw(y, W, D, phi, cont.size, sigma2, ybuf);
  multiply_matvec_MN(USmat, cont_recon.size, cont.size, ybuf, cont_recon.flux);

  // SxC^-1xS^T
  multiply_mat_transposeB_semiseparable_drw(USmat, W, D, phi, cont.size, cont_recon.size, sigma2, PEmat1);
  multiply_mat_MN(USmat, PEmat1, PEmat2, cont_recon.size, cont_recon.size, cont.size);

  for(i=0; i<cont_recon.size; i++)
  {
    cont_recon.error[i] = sqrt(sigma2 + syserr*syserr - PEmat2[i*cont_recon.size + i]);
  }

  set_covar_Pmat(sigma, tau, alpha);
  compute_semiseparable_drw(cont_recon.time, cont_recon.size, sigma2, 1.0/tau, cont_recon.error, 0.0, W, D, phi);

  // Q = [S^-1 + N^-1]^-1 = N x [S+N]^-1 x S
  multiply_mat_semiseparable_drw(PSmat, W, D, phi, cont_recon.size, cont_recon.size, sigma2, PEmat2);
  for(i=0; i<cont_recon.size; i++)
  {
    for(j=0; j<=i; j++)
    {
      PSmat[i*cont_recon.size + j] = PSmat[j*cont_recon.size + i] = cont_recon.error[i] * cont_recon.error[i] * PEmat2[i*cont_recon.size+j];
    }
  }  
  // Q^1/2
  Chol_decomp_L(PSmat, cont_recon.size, &info);
  multiply_matvec(PSmat, &pm[num_params_var], cont_recon.size, y);

  for(i=0; i<cont_recon.size; i++)
  {
    cont_recon.flux[i] += y[i] + yq[0];
  }

  ofstream fout;
  fout.open("data/cont_recon_drw.txt");
  for(i=0; i<cont_recon.size; i++)
  {
    fout<<cont_recon.time[i]<<"  "<<cont_recon.flux[i]*cont_recon.norm<<"   "<<cont_recon.error[i]*cont_recon.norm<<endl;
  }
  fout.close();
}

/*
 * different covariance matrix
 * S - SxC^-1xS
 */
void ContModel::recon2(const void *model)
{
  double *Lbuf, *ybuf, *y, *Cq, *yq, *W, *D, *phi, *u, *v;
  double syserr;

  double *pm = (double *)model;
  double sigma, sigma2, tau, alpha;
  int i, info;

  syserr = (exp(pm[0]) - 1.0) * mean_error;  // systematic error 
  tau = exp(pm[2]);
  sigma = exp(pm[1]) * sqrt(tau);
  sigma2 = sigma*sigma;
  alpha = 1.0;
  
  Lbuf = workspace;
  ybuf = Lbuf + cont.size*nq; 
  y = ybuf + size_max;
  Cq = y + size_max;
  yq = Cq + nq*nq;
  W = yq + nq;
  D = W + size_max;
  phi = D + size_max;
  u = phi + size_max;
  v = u + cont_recon.size;

  compute_semiseparable_drw(cont.time, cont.size, sigma2, 1.0/tau, cont.error, syserr, W, D, phi);
  // Cq^-1 = L^TxC^-1xL
  multiply_mat_semiseparable_drw(Larr_data, W, D, phi, cont.size, nq, sigma2, Lbuf);
  multiply_mat_MN_transposeA(Larr_data, Lbuf, Cq, nq, nq, cont.size);

  // L^TxC^-1xy
  multiply_matvec_semiseparable_drw(cont.flux, W, D, phi, cont.size, sigma2, ybuf);
  multiply_mat_MN_transposeA(Larr_data, ybuf, yq, nq, 1, cont.size);

  // (hat q) = Cqx(L^TxC^-1xy)
  inverse_pomat(Cq, nq, &info);
  multiply_mat_MN(Cq, yq, ybuf, nq, 1, nq);

  // q = uq + (hat q)
  Chol_decomp_L(Cq, nq, &info);
  multiply_matvec(Cq, &pm[num_params_drw], nq, yq);
  for(i=0; i<nq; i++)
    yq[i] += ybuf[i];
  
  // y = yc - Lxq
  multiply_matvec_MN(Larr_data, cont.size, nq, yq, ybuf);
  for(i=0; i<cont.size; i++)
  {
    y[i] = cont.flux[i] - ybuf[i];
  }
  
  set_covar_Umat(sigma, tau, alpha);
  // (hat s) = SxC^-1xy
  multiply_matvec_semiseparable_drw(y, W, D, phi, cont.size, sigma2, ybuf);
  multiply_matvec_MN(USmat, cont_recon.size, cont.size, ybuf, cont_recon.flux);

  // SxC^-1xS^T
  multiply_mat_transposeB_semiseparable_drw(USmat, W, D, phi, cont.size, cont_recon.size, sigma2, PEmat1);
  multiply_mat_MN(USmat, PEmat1, PEmat2, cont_recon.size, cont_recon.size, cont.size);

  set_covar_Pmat(sigma, tau, alpha);
  for(i=0; i<cont_recon.size * cont_recon.size; i++)
  {
    PSmat[i] = PSmat[i] - PEmat2[i];
  }
  for(i=0; i<cont_recon.size; i++)
  {
    cont_recon.error[i] = sqrt(PSmat[i*cont_recon.size + i]);
  }
  Chol_decomp_L(PSmat, cont_recon.size, &info);
  multiply_matvec(PSmat, &pm[num_params_var], cont_recon.size, y);

  for(i=0; i<cont_recon.size; i++)
  {
    cont_recon.flux[i] += y[i] + yq[0];
  }

  ofstream fout;
  fout.open("data/cont_recon_drw.txt");
  for(i=0; i<cont_recon.size; i++)
  {
    fout<<cont_recon.time[i]<<"  "<<cont_recon.flux[i]*cont_recon.norm<<"   "<<cont_recon.error[i]*cont_recon.norm<<endl;
  }
  fout.close();
}

void ContModel::set_covar_Umat(double sigma, double tau, double alpha)
{
  double t1, t2;
  int i, j;
 
  for(i=0; i<cont_recon.size; i++)
  {
    t1 = cont_recon.time[i];
    for(j=0; j<cont.size; j++)
    {
      t2 = cont.time[j];
      USmat[i*cont.size+j] = sigma*sigma * exp (- pow (fabs(t1-t2) / tau, alpha) );
    }
  }
  return;
}

void ContModel::set_covar_Pmat(double sigma, double tau, double alpha)
{
  double t1, t2;
  int i, j;
 
  for(i=0; i<cont_recon.size; i++)
  {
    t1 = cont_recon.time[i];
    for(j=0; j<=i; j++)
    {
      t2 = cont_recon.time[j];
      PSmat[i*cont_recon.size+j] = sigma*sigma* exp (- pow (fabs(t1-t2) / tau, alpha));
      PSmat[j*cont_recon.size+i] = PSmat[i*cont_recon.size+j];
    }
  }
  return;
}