/*
 *  PIXON
 *  A Pixon-based method for reconstructing velocity-delay map in reverberation mapping.
 * 
 *  Yan-Rong Li, liyanrong@mail.ihep.ac.cn
 * 
 */
#include "mathfun.h"
#include "utilities.hpp"
#include "drw_cont.hpp"

PixonDRW::PixonDRW()
{
  grad_chisq_cont = NULL;
  workspace = NULL;
  workspace_uv = NULL;
  Larr_data = NULL;
  USmat = NULL;
}

PixonDRW::PixonDRW(
   Data& cont_data_in, Data& cont_in, Data& line_data_in, 
   int npixel_in,  int npixon_size_max_in, double sigmad_in, double taud_in, double syserr_in,
   int ipositive_in, double sensitivity_in
  )
  :Pixon(cont_in, line_data_in, npixel_in, npixon_size_max_in, ipositive_in, sensitivity_in),
   cont_data(cont_data_in),
   sigmad(sigmad_in), taud(taud_in), syserr(syserr_in)
{
  int i;

  nq = 1;
  size_max = fmax(cont.size, cont_data.size);
  workspace = new double[size_max*15];
  workspace_uv = new double[2*cont.size];
  Larr_data = new double[cont_data.size*nq];
  for(i=0; i<cont_data.size; i++)
  {
    Larr_data[i*nq+0] = 1.0;
  }
  USmat = new double [cont_data.size * cont.size];
  PQmat = new double [cont.size * cont.size];
  D_data = new double[cont_data.size];
  W_data = new double[cont_data.size];
  phi_data = new double[cont_data.size];
  Cq = new double[nq*nq];
  QLmat = new double[nq*cont.size];
  qhat = new double[nq];
  D_recon = new double[cont.size];
  W_recon = new double[cont.size];
  phi_recon = new double[cont.size];

  grad_chisq_cont = new double[cont_in.size+nq];

  compute_matrix();
  //compute_matrix2();
}

PixonDRW::~PixonDRW()
{
  int i;
  delete[] grad_chisq_cont;

  delete[] workspace;
  delete[] workspace_uv;
  delete[] Larr_data;
  delete[] USmat;
  delete[] D_data;
  delete[] W_data;
  delete[] phi_data;
  delete[] Cq;
  delete[] QLmat;
  delete[] qhat;
  delete[] D_recon;
  delete[] W_recon;
  delete[] phi_recon;
}

/* compute rm amd pixon convolutions */
void PixonDRW::compute_rm_pixon(const double *x)
{
  compute_cont(x + npixel + 1);
  rmfft.set_data(cont.flux, cont.size);
  Pixon::compute_rm_pixon(x);
}

double PixonDRW::compute_chisquare(const double *x)
{
  /* calculate chi square */
  chisq = Pixon::compute_chisquare(x);
  return chisq;
}

/* calculate prior of uq and us, which are Gaussian variables */
double PixonDRW::compute_prior(const double *x)
{
  /* include prior, minimizing, so drop the minus sign */
  int i;
  prior = 0.0;
  for(i=0; i<cont.size+nq; i++)
  {
    prior += x[npixel+1+i]*x[npixel+1+i];
  }
  return prior;
}

double PixonDRW::compute_post(const double *x)
{
  return compute_chisquare(x) + compute_prior(x);
}

void PixonDRW::compute_post_grad(const double *x)
{
  Pixon::compute_chisquare_grad(x);       /* derivative of chisq_line with respect to transfer function */

  /* derivative of chisq_line with respect to continuum */
  int i, j;
  double *res_mat = workspace;
  double grad_in, tj, ti, Iintp;
  
  for(i=0; i<cont.size; i++)
  {
    ti = cont.time[i];
    grad_in = 0.0;
    for(j=0; j<line.size; j++)
    {
      tj = line.time[j];
      Iintp = interp_image(tj - ti);
      grad_in += residual[j]/line.error[j]/line.error[j] * Iintp;
    }
    res_mat[i] = grad_in *  2.0;
  }
  // w.r.t uq
  multiply_mat_MN(QLmat, res_mat, grad_chisq_cont, nq, 1, cont.size);
  // w.r.t us
  multiply_mat_MN(PQmat, res_mat, grad_chisq_cont+nq, cont.size, 1, cont.size);

  /* grad of prior */
  for(i=0; i<cont.size+nq; i++)
  {
    grad_chisq_cont[i] += 2.0 * x[npixel+1+i];
  }
}

double PixonDRW::compute_mem(const double *x)
{
  mem = Pixon::compute_mem(x);
  return mem;
}

void PixonDRW::compute_mem_grad(const double *x)
{
  Pixon::compute_mem_grad(x);
}

void PixonDRW::compute_matrix()
{
  double *PEmat1, *PEmat2, *PSmat;
  double *CL, *ybuf, *y, *yq, *u, *v;

  double sigmad2, alpha;
  int i, j, info;
  
  PEmat1 = new double [cont_data.size * cont.size];
  PEmat2 = new double [cont.size * cont.size];
  PSmat = new double [cont.size * cont.size];

  sigmad2 = sigmad*sigmad;
  alpha = 1.0;

  CL = workspace;
  ybuf = CL + cont_data.size*nq; 
  y = ybuf + size_max;
  yq = y + size_max;
  u = yq + nq;
  v = u + cont.size;

  compute_semiseparable_drw(cont_data.time, cont_data.size, sigmad2, 1.0/taud, cont_data.error, syserr, W_data, D_data, phi_data);
  // Cq^-1 = L^TxC^-1xL
  multiply_mat_semiseparable_drw(Larr_data, W_data, D_data, phi_data, cont_data.size, nq, sigmad2, CL);
  multiply_mat_MN_transposeA(Larr_data, CL, Cq, nq, nq, cont_data.size);

  // L^TxC^-1xy
  multiply_matvec_semiseparable_drw(cont_data.flux, W_data, D_data, phi_data, cont_data.size, sigmad2, ybuf);
  multiply_mat_MN_transposeA(Larr_data, ybuf, yq, nq, 1, cont_data.size);

  // (hat q) = Cqx(L^TxC^-1xy)
  inverse_pomat(Cq, nq, &info);
  multiply_mat_MN(Cq, yq, qhat, nq, 1, nq);

  // Cq^1/2
  Chol_decomp_L(Cq, nq, &info);

  set_covar_Umat(sigmad, taud, alpha);
  // SxC^-1xS^T
  multiply_mat_transposeB_semiseparable_drw(USmat, W_data, D_data, phi_data, cont_data.size, cont.size, sigmad2, PEmat1);
  multiply_mat_MN(USmat, PEmat1, PEmat2, cont.size, cont.size, cont_data.size);

  // assign errors
  for(i=0; i<cont.size; i++)
  {
    cont.error[i] = sqrt(sigmad2 - PEmat2[i*cont.size + i]);
  }

  // Cq^1/2 x (L - SxC^-1xL)^T
  multiply_mat_MN(USmat, CL, PEmat2, cont.size, nq, cont_data.size);
  for(i=0; i<cont.size; i++)
  {
    PEmat1[i] = 1.0 - PEmat2[i];
  }
  multiply_mat_MN_transposeB(Cq, PEmat1, QLmat, nq, cont.size, nq);

  set_covar_Pmat(sigmad, taud, alpha, PSmat);
  compute_semiseparable_drw(cont.time, cont.size, sigmad2, 1.0/taud, cont.error, 0.0, W_recon, D_recon, phi_recon);

  // Q = [S^-1 + N^-1]^-1 = N x [S+N]^-1 x S
  multiply_mat_semiseparable_drw(PSmat, W_recon, D_recon, phi_recon, cont.size, cont.size, sigmad2, PEmat2);
  for(i=0; i<cont.size; i++)
  {
    for(j=0; j<=i; j++)
    {
      PQmat[i*cont.size + j] = PQmat[j*cont.size + i] = cont.error[i] * cont.error[i] * PEmat2[i*cont.size+j];
    }
  }  
  // Q^1/2
  Chol_decomp_L(PQmat, cont.size, &info);

  delete[] PEmat1;
  delete[] PEmat2;
  delete[] PSmat;
}

/*
 * different cocariance matrix 
 * S-SxC^-1xS
 * 
 */
void PixonDRW::compute_matrix2()
{
  double *PEmat1, *PEmat2, *PSmat;
  double *CL, *ybuf, *y, *yq, *u, *v;

  double sigmad2, alpha;
  int i, j, info;
  
  PEmat1 = new double [cont_data.size * cont.size];
  PEmat2 = new double [cont.size * cont.size];
  PSmat = new double [cont.size * cont.size];

  sigmad2 = sigmad*sigmad;
  alpha = 1.0;

  CL = workspace;
  ybuf = CL + cont_data.size*nq; 
  y = ybuf + size_max;
  yq = y + size_max;
  u = yq + nq;
  v = u + cont.size;

  compute_semiseparable_drw(cont_data.time, cont_data.size, sigmad2, 1.0/taud, cont_data.error, syserr, W_data, D_data, phi_data);
  // Cq^-1 = L^TxC^-1xL
  multiply_mat_semiseparable_drw(Larr_data, W_data, D_data, phi_data, cont_data.size, nq, sigmad2, CL);
  multiply_mat_MN_transposeA(Larr_data, CL, Cq, nq, nq, cont_data.size);

  // L^TxC^-1xy
  multiply_matvec_semiseparable_drw(cont_data.flux, W_data, D_data, phi_data, cont_data.size, sigmad2, ybuf);
  multiply_mat_MN_transposeA(Larr_data, ybuf, yq, nq, 1, cont_data.size);

  // (hat q) = Cqx(L^TxC^-1xy)
  inverse_pomat(Cq, nq, &info);
  multiply_mat_MN(Cq, yq, qhat, nq, 1, nq);

  // Cq^1/2
  Chol_decomp_L(Cq, nq, &info);

  set_covar_Umat(sigmad, taud, alpha);
  // SxC^-1xS^T
  multiply_mat_transposeB_semiseparable_drw(USmat, W_data, D_data, phi_data, cont_data.size, cont.size, sigmad2, PEmat1);
  multiply_mat_MN(USmat, PEmat1, PEmat2, cont.size, cont.size, cont_data.size);

  set_covar_Pmat(sigmad, taud, alpha, PSmat);
  for(i=0; i<cont.size * cont.size; i++)
  {
    PQmat[i] = PSmat[i] - PEmat2[i];
  }
  // assign errors
  for(i=0; i<cont.size; i++)
  {
    cont.error[i] = sqrt(PQmat[i*cont.size + i]);
  }
  // Q^1/2
  Chol_decomp_L(PQmat, cont.size, &info);

  // Cq^1/2 x (L - SxC^-1xL)^T
  multiply_mat_MN(USmat, CL, PEmat2, cont.size, nq, cont_data.size);
  for(i=0; i<cont.size; i++)
  {
    PEmat1[i] = 1.0 - PEmat2[i];
  }
  multiply_mat_MN_transposeB(Cq, PEmat1, QLmat, nq, cont.size, nq);

  delete[] PEmat1;
  delete[] PEmat2;
  delete[] PSmat;
}

void PixonDRW::compute_cont(const double *x)
{
  double *ybuf, *y, *yq;

  double sigmad2, alpha;
  int i, j, info;
  
  double *pm = (double *)x;
  sigmad2 = sigmad*sigmad;
  alpha = 1.0;

  ybuf = workspace; 
  y = ybuf + size_max;
  yq = y + size_max;

  // q = uq + (hat q)
  multiply_matvec(Cq, pm, nq, yq);
  for(i=0; i<nq; i++)
    yq[i] += qhat[i];
  
  // y = yc - Lxq
  multiply_matvec_MN(Larr_data, cont_data.size, nq, yq, ybuf);
  for(i=0; i<cont_data.size; i++)
  {
    y[i] = cont_data.flux[i] - ybuf[i];
  }
  
  // (hat s) = SxC^-1xy
  multiply_matvec_semiseparable_drw(y, W_data, D_data, phi_data, cont_data.size, sigmad2, ybuf);
  multiply_matvec_MN(USmat, cont.size, cont_data.size, ybuf, cont.flux);

  multiply_matvec(PQmat, pm+nq, cont.size, y); 

  for(i=0; i<cont.size; i++)
  {
    cont.flux[i] += y[i] + yq[0];
  }
  return;
}

void PixonDRW::set_covar_Umat(double sigma, double tau, double alpha)
{
  double t1, t2;
  int i, j;
 
  for(i=0; i<cont.size; i++)
  {
    t1 = cont.time[i];
    for(j=0; j<cont_data.size; j++)
    {
      t2 = cont_data.time[j];
      USmat[i*cont_data.size+j] = sigma*sigma * exp (- pow (fabs(t1-t2) / tau, alpha) );
    }
  }
  return;
}

void PixonDRW::set_covar_Pmat(double sigma, double tau, double alpha, double *PSmat)
{
  double t1, t2;
  int i, j;
 
  for(i=0; i<cont.size; i++)
  {
    t1 = cont.time[i];
    for(j=0; j<=i; j++)
    {
      t2 = cont.time[j];
      PSmat[i*cont.size+j] = sigma*sigma* exp (- pow (fabs(t1-t2) / tau, alpha));
      PSmat[j*cont.size+i] = PSmat[i*cont.size+j];
    }
  }
  return;
}

/* function for nlopt */
double func_nlopt_cont_drw(const vector<double> &x, vector<double> &grad, void *f_data)
{
  PixonDRW *pixon = (PixonDRW *)f_data;
  double chisq, mem;

  pixon->compute_rm_pixon(x.data());
  if (!grad.empty()) 
  {
    int i;
    
    pixon->compute_post_grad(x.data());
    pixon->compute_mem_grad(x.data());
    
    for(i=0; i<pixon->npixel+1; i++)
      grad[i] = pixon->grad_chisq[i] + pixon->grad_mem[i];

    for(i=pixon->npixel+1; i<(int)grad.size(); i++)
      grad[i] = pixon->grad_chisq_cont[i-pixon->npixel-1];
  }
  chisq = pixon->compute_post(x.data());
  mem = pixon->compute_mem(x.data());
  return chisq + mem;
}


/* function for tnc */
int func_tnc_cont_drw(double x[], double *f, double g[], void *state)
{
  PixonDRW *pixon = (PixonDRW *)state;
  int i;
  double chisq, mem;

  pixon->compute_rm_pixon(x);
  pixon->compute_post_grad(x);
  pixon->compute_mem_grad(x);
  
  chisq = pixon->compute_post(x);
  mem = pixon->compute_mem(x);

  *f = chisq + mem;

  for(i=0; i<pixon->npixel+1; i++)
    g[i] = pixon->grad_chisq[i] + pixon->grad_mem[i];

  for(i=pixon->npixel+1; i<pixon->cont.size + pixon->nq + pixon->npixel+1; i++)
    g[i] = pixon->grad_chisq_cont[i - pixon->npixel - 1];

  return 0;
}