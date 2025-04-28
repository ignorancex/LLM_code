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

#include "utilities.hpp"

int pixon_size_factor;
int pixon_sub_factor;
int pixon_map_low_bound;

using namespace std;

/*==================================================================*/
/* class configuration */
Config::Config()
{
  pixon_basis_type = 0;
  pixon_uniform = true;
  drv_lc_model = 0;

  fix_bg = false;
  bg = 0.0;

  fcont = "data/con.txt";
  fline = "data/line.txt";

  pixon_size_factor = 1;
  pixon_sub_factor = 1;
  pixon_map_low_bound = pixon_sub_factor - 1;
}
Config::~Config()
{

}

void Config::load_cfg(string fname)
{
  ifstream is(fname);
  configparser::ConfigParser<char> param;
	param.parse(is);
  //param.generate(std::cout);

  if(!configparser::extract(param.sections["param"]["pixon_basis_type"], pixon_basis_type))
  {
    pixon_basis_type = 0;
  }
  if(!configparser::extract(param.sections["param"]["pixon_uniform"], pixon_uniform))
  {
    pixon_uniform = true;
  }
  if(!configparser::extract(param.sections["param"]["drv_lc_model"], drv_lc_model))
  {
    drv_lc_model = 0;
  }
  configparser::extract(param.sections["param"]["fcont"], fcont);
  if(fcont.empty())
  {
    cout<<"fcont is not defined!"<<endl;
    cout<<"exit!"<<endl;
    exit(0);
  }
  configparser::extract(param.sections["param"]["fline"], fline);
  if(fline.empty())
  {
    cout<<"fline is not defined!"<<endl;
    cout<<"exit!"<<endl;
    exit(0);
  }
  if(!configparser::extract(param.sections["param"]["tau_range_low"], tau_range_low))
  {
    tau_range_low = 0.0;
  }
  if(!configparser::extract(param.sections["param"]["tau_range_up"], tau_range_up))
  {
    tau_range_up = 10.0;
  }
  if(!configparser::extract(param.sections["param"]["tau_interval"], tau_interval))
  {
    tau_interval = 1.0;
  }
  if(!configparser::extract(param.sections["param"]["fix_bg"], fix_bg))
  {
    fix_bg = false;
  }
  if(!configparser::extract(param.sections["param"]["bg"], bg))
  {
    bg = 0.0;
  }

  if(!configparser::extract(param.sections["param"]["tol"], tol))
  {
    tol = 1.0e-6;
  }
  if(!configparser::extract(param.sections["param"]["nfeval_max"], nfeval_max))
  {
    nfeval_max = 10000;
  }

  if(!configparser::extract(param.sections["param"]["pixon_sub_factor"], pixon_sub_factor))
  {
    pixon_sub_factor = 1;
  }
  if(!configparser::extract(param.sections["param"]["pixon_size_factor"], pixon_size_factor))
  {
    pixon_size_factor = 1;
  }
  if(!configparser::extract(param.sections["param"]["max_pixon_size"], max_pixon_size))
  {
    max_pixon_size = 10;
  }
  if(!configparser::extract(param.sections["param"]["sensitivity"], sensitivity))
  {
    sensitivity = 10;
  }
  pixon_map_low_bound = pixon_sub_factor - 1;

  if(drv_lc_model < 0 || drv_lc_model > 3)
  {
    cout<<"Incorrect configuration drv_lc_model."<<endl;
  }
}

void Config::print_cfg()
{
  ofstream fout;
  fout.open("data/param_input");
  fout<<setw(24)<<left<<"pixon_basis_type"<<" = "<<pixon_basis_type<<endl;
  fout<<setw(24)<<left<<boolalpha<<"pixon_uniform"<<" = "<<pixon_uniform<<endl;
  fout<<setw(24)<<left<<"drv_lc_model"<<" = "<<drv_lc_model<<endl;
  fout<<setw(24)<<left<<"fcont"<<" = "<<fcont<<endl;
  fout<<setw(24)<<left<<"fline"<<" = "<<fline<<endl;
  fout<<setw(24)<<left<<"tau_range_low"<<" = "<<tau_range_low<<endl;
  fout<<setw(24)<<left<<"tau_range_up"<<" = "<<tau_range_up<<endl;
  fout<<setw(24)<<left<<"tau_interval"<<" = "<<tau_interval<<endl;
  fout<<setw(24)<<left<<boolalpha<<"fix_bg"<<" = "<<fix_bg<<endl;
  fout<<setw(24)<<left<<"bg"<<" = "<<bg<<endl;
  fout<<setw(24)<<left<<"tol"<<" = "<<tol<<endl;
  fout<<setw(24)<<left<<"nfeval_max"<<" = "<<nfeval_max<<endl;
  fout<<setw(24)<<left<<"pixon_sub_factor"<<" = "<<pixon_sub_factor<<endl;
  fout<<setw(24)<<left<<"pixon_size_factor"<<" = "<<pixon_size_factor<<endl;
  fout<<setw(24)<<left<<"pixon_map_low_bound"<<" = "<<pixon_map_low_bound<<endl;
  fout<<setw(24)<<left<<"max_pixon_size"<<" = "<<max_pixon_size<<endl;
  fout<<setw(24)<<left<<"sensitivity"<<" = "<<sensitivity<<endl;
  fout.close();
}

/*==================================================================*/
/* class PixonBasis */

double PixonBasis::norm_gaussian= sqrt(2*M_PI) * erf(3.0/sqrt(2.0));
double PixonBasis::coeff1_modified_gaussian = exp(-0.5*9.0);
double PixonBasis::coeff2_modified_gaussian =(1.0 - exp(-0.5*9.0));
double PixonBasis::norm_modified_gaussian= (sqrt(2*M_PI) * erf(3.0/sqrt(2.0)) - 2*3.0*exp(-0.5*9.0))/PixonBasis::coeff2_modified_gaussian;

string PixonBasis::pixonbasis_name[] = {"parabloid", "Gaussian", "modified Gaussian", "Lorentz", "Wendland", "triangle","top-hat"};

/* modified gaussian function, truncated at factor * psize */
double PixonBasis::gaussian(double x, double y, double psize)
{
  if(fabs(y-x) <= pixon_size_factor * psize)
    return gaussian_norm(psize) * exp( -0.5*(y-x)*(y-x)/(psize/3*psize/3) );
  else 
    return 0.0;
}
double PixonBasis::gaussian_norm(double psize)
{
  return 1.0/(norm_gaussian*psize/3.0);
}

/* modified gaussian function, truncated at factor * psize */
double PixonBasis::modified_gaussian(double x, double y, double psize)
{
  if(fabs(y-x) <= pixon_size_factor * psize)
    return gaussian_norm(psize)/coeff2_modified_gaussian * (exp( -0.5*(y-x)*(y-x)/(psize/3*psize/3) ) - coeff1_modified_gaussian);
  else 
    return 0.0;
}
double PixonBasis::modified_gaussian_norm(double psize)
{
  return 1.0/(norm_modified_gaussian*psize/3.0);
}

/* prarabloid function, truncated at factor * psize */
double PixonBasis::parabloid(double x, double y, double psize)
{
  double d = fabs(y-x)/(pixon_size_factor * psize);
  if( d <= 1.0)
    return parabloid_norm(psize) * (1.0 - d*d);
  else 
    return 0.0;
}
double PixonBasis::parabloid_norm(double psize)
{
  return 1.0/(psize * 4.0/3.0 * pixon_size_factor);
}

double PixonBasis::tophat(double x, double y, double psize)
{
  if(fabs(y-x) <= pixon_size_factor * psize)
    return parabloid_norm(psize);
  else 
    return 0.0;
}
double PixonBasis::tophat_norm(double psize)
{
  return 1.0/(psize * 2.0*pixon_size_factor);
}

double PixonBasis::triangle(double x, double y, double psize)
{
  double d = fabs(y-x)/(pixon_size_factor * psize);
  if(d <= 1.0)
    return triangle_norm(psize) * (1.0 - d);
  else 
    return 0.0;
}
double PixonBasis::triangle_norm(double psize)
{
  return 1.0/(pixon_size_factor * psize);
}
double PixonBasis::lorentz(double x, double y, double psize)
{
  double d = fabs(y-x)/(psize);
  if(d <= pixon_size_factor)
    return lorentz_norm(psize) / (1.0 + d*d*3*3);
  else 
    return 0.0;
}
double PixonBasis::lorentz_norm(double psize)
{
  return 1.0/(2.0*psize/3.0 * atan(pixon_size_factor*3.0));
}

double PixonBasis::wendland(double x, double y, double psize)
{
  double d = fabs(y-x)/psize;
  if( d <= 1.0 )
    return wendland_norm(psize) * pow((1.0 - d), 4.0) * (4.0*d + 1);
  else 
    return 0.0;
}
double PixonBasis::wendland_norm(double psize)
{
  return 1.5/psize;
}
/*==================================================================*/
/* class Data */
Data::Data()
{
  size = 0;
  time = flux = error = NULL;
  norm = 1.0;
}

/* constructor with a size of n */
Data::Data(int n)
{
  if(n > 0)
  {
    size = n;
    time = new double[size];
    flux = new double[size];
    error = new double[size];
    norm = 1.0;
  }
  else
  {
    cout<<"Data size must be positive."<<endl;
    exit(0);
  }
}

Data::~Data()
{
  if(size > 0)
  {
    size = 0;
    delete[] time;
    delete[] flux;
    delete[] error;
  }
}

Data::Data(Data& data)
{ 
  size = data.size;
  time = new double[size];
  flux = new double[size];
  error = new double[size];
  memcpy(time, data.time, size*sizeof(double));
  memcpy(flux, data.flux, size*sizeof(double));
  memcpy(error, data.error, size*sizeof(double));

  norm = data.norm;
}

Data& Data::operator = (Data& data)
{
  if(size != data.size)
  {
    if(size > 0)
    {
      delete[] time;
      delete[] flux;
      delete[] error;
    }
    size = data.size;
    time = new double[size];
    flux = new double[size];
    error = new double[size];
    memcpy(time, data.time, size*sizeof(double));
    memcpy(flux, data.flux, size*sizeof(double));
    memcpy(error, data.error, size*sizeof(double));

    norm = data.norm;
  }
  else 
  {
    memcpy(time, data.time, size*sizeof(double));
    memcpy(flux, data.flux, size*sizeof(double));
    memcpy(error, data.error, size*sizeof(double));

    norm = data.norm;
  }
  return *this;
}

void Data::set_size(int n)
{
  if(size != n)
  {
    if(size > 0)
    {
      delete[] time;
      delete[] flux;
      delete[] error;
    }
    size = n;
    time = new double[size];
    flux = new double[size];
    error = new double[size];

    norm = 1.0;
  }
}

void Data::load(const string& fname)
{
  ifstream fin;
  string line;
  int i;
  
  /* first determine number of lines */
  fin.open(fname);
  if(!fin.good())
  {
    cout<<fname<<" does not exist!"<<endl;
    exit(0);
  }
  i = 0;
  while(1)
  {
    getline(fin, line);
    if(fin.good())
    {
      i++;
    }
    else
    {
      break;
    }
  }

  /* delete old data if sizes do not match */
  if(i != size)
  {
    if(size > 0)
    {
      delete[] time;
      delete[] flux;
      delete[] error;
    }
    size = i;
  }
  cout<<"file \""+fname+"\" has "<<size<<" lines."<<endl;

  /* allocate memory */
  time = new double[size];
  flux = new double[size];
  error = new double[size];

  /* now read data */
  fin.clear();  // clear flags
  fin.seekg(0); // go to the beginning
  for(i=0; i<size; i++)
  {
    fin>>time[i]>>flux[i]>>error[i];
    if(fin.fail())
    {
      cout<<"# Error in reading the file \""+fname+"\", no enough points in line "<<i<<"."<<endl;
      exit(0);
    }
  }
  fin.close();

  normalize();
}

void Data::set_data(double *data)
{
  memcpy(flux, data, size*sizeof(double));
}

void Data::set_norm(double norm_in)
{
  norm = norm_in;
}

void Data::normalize()
{
  int i;
  norm = 0.0;
  for(i=0; i<size; i++)
  {
    norm += flux[i];
  }
  norm /= size;
  for(i=0; i<size; i++)
  {
    flux[i] /= norm;
    error[i] /= norm;
  }
}

/*==================================================================*/
/* class DataFFT */
DataFFT::DataFFT()
{
  nd = npad = nd_fft = nd_fft_cal = 0;
  data_fft = resp_fft = conv_fft = NULL;
  data_real = resp_real = conv_real = NULL;
}

DataFFT::DataFFT(int nd_in, double fft_dx, int npad_in)
      :nd(nd_in), npad(npad_in) 
{
  int i;

  nd_fft = nd + npad;
  nd_fft_cal = nd_fft/2 + 1;

  data_fft = (fftw_complex *) fftw_malloc((nd_fft_cal) * sizeof(fftw_complex));
  resp_fft = (fftw_complex *) fftw_malloc((nd_fft_cal) * sizeof(fftw_complex));
  conv_fft = (fftw_complex *) fftw_malloc((nd_fft_cal) * sizeof(fftw_complex));

  data_real = new double[nd_fft];
  resp_real = new double[nd_fft];
  conv_real = new double[nd_fft];
      
  pdata = fftw_plan_dft_r2c_1d(nd_fft, data_real, data_fft, FFTW_PATIENT);
  presp = fftw_plan_dft_r2c_1d(nd_fft, resp_real, resp_fft, FFTW_PATIENT);
  pback = fftw_plan_dft_c2r_1d(nd_fft, conv_fft, conv_real, FFTW_PATIENT);
      
  /* normalization */
  fft_norm = fft_dx/nd_fft;

  for(i=0; i < nd_fft; i++)
  {
    data_real[i] = resp_real[i] = conv_real[i] = 0.0;
  }
}

DataFFT::DataFFT(Data& cont, int npad_in)
      :nd(cont.size), npad(npad_in)
{
  int i;

  nd_fft = nd + npad;
  nd_fft_cal = nd_fft/2 + 1;

  data_fft = (fftw_complex *) fftw_malloc((nd_fft_cal) * sizeof(fftw_complex));
  resp_fft = (fftw_complex *) fftw_malloc((nd_fft_cal) * sizeof(fftw_complex));
  conv_fft = (fftw_complex *) fftw_malloc((nd_fft_cal) * sizeof(fftw_complex));

  data_real = new double[nd_fft];
  resp_real = new double[nd_fft];
  conv_real = new double[nd_fft];
      
  pdata = fftw_plan_dft_r2c_1d(nd_fft, data_real, data_fft, FFTW_PATIENT);
  presp = fftw_plan_dft_r2c_1d(nd_fft, resp_real, resp_fft, FFTW_PATIENT);
  pback = fftw_plan_dft_c2r_1d(nd_fft, conv_fft, conv_real, FFTW_PATIENT);
      
  fft_norm = (cont.time[1] - cont.time[0]) / nd_fft;

  for(i=0; i < nd_fft; i++)
  {
    data_real[i] = resp_real[i] = conv_real[i] = 0.0;
  }
}

DataFFT& DataFFT::operator = (DataFFT& df)
{
  if(nd != df.nd)
  {
    if(nd > 0)
    {
      fftw_free(data_fft);
      fftw_free(resp_fft);
      fftw_free(conv_fft);

      delete[] data_real;
      delete[] resp_real;
      delete[] conv_real;

      fftw_destroy_plan(pdata);
      fftw_destroy_plan(presp);
      fftw_destroy_plan(pback);
    }
        
    int i;
    nd = df.nd;
    npad = df.npad;
    nd_fft = nd + npad;
    nd_fft_cal = nd_fft/2 + 1;
  
    data_fft = (fftw_complex *) fftw_malloc((nd_fft_cal) * sizeof(fftw_complex));
    resp_fft = (fftw_complex *) fftw_malloc((nd_fft_cal) * sizeof(fftw_complex));
    conv_fft = (fftw_complex *) fftw_malloc((nd_fft_cal) * sizeof(fftw_complex));
  
    data_real = new double[nd_fft];
    resp_real = new double[nd_fft];
    conv_real = new double[nd_fft];
        
    pdata = fftw_plan_dft_r2c_1d(nd_fft, data_real, data_fft, FFTW_PATIENT);
    presp = fftw_plan_dft_r2c_1d(nd_fft, resp_real, resp_fft, FFTW_PATIENT);
    pback = fftw_plan_dft_c2r_1d(nd_fft, conv_fft, conv_real, FFTW_PATIENT);
        
    fft_norm = df.fft_norm;
  
    for(i=0; i < nd_fft; i++)
    {
      data_real[i] = resp_real[i] = conv_real[i] = 0.0;
    }
  }
  return *this;
}

DataFFT::~DataFFT()
{
  if(nd > 0)
  {
    nd = 0;

    fftw_free(data_fft);
    fftw_free(resp_fft);
    fftw_free(conv_fft);

    delete[] data_real;
    delete[] resp_real;
    delete[] conv_real;

    fftw_destroy_plan(pdata);
    fftw_destroy_plan(presp);
    fftw_destroy_plan(pback);
  }
}

/* convolution with resp, output to conv */
void DataFFT::convolve_simple(double *conv)
{
  int i;
  for(i=0; i<nd_fft_cal; i++)
  {
    conv_fft[i][0] = data_fft[i][0]*resp_fft[i][0] - data_fft[i][1]*resp_fft[i][1];
    conv_fft[i][1] = data_fft[i][0]*resp_fft[i][1] + data_fft[i][1]*resp_fft[i][0];
  }
  fftw_execute(pback);

  /* normalize */
  for(i=0; i<nd_fft; i++)
  {
    conv_real[i] *= fft_norm;
  }

  /* copy back, the first npad points are discarded */
   memcpy(conv, conv_real, nd*sizeof(double));      
   return;
}

void DataFFT::set_resp_real(const double *resp, int nall, int ipositive)
{
  /* positive-lag part */
  memcpy(resp_real, resp+ipositive, (nall - ipositive)*sizeof(double));

  /* negative-lag part */
  int i; 
  for(i=0; i<ipositive; i++)
  {
    resp_real[nd_fft-ipositive+i] = resp[i];
  }
  fftw_execute(presp);
}

/*==================================================================*/
/* class RMFFT */
RMFFT::RMFFT(int n, double dx, int npad_in)
      :DataFFT(n, dx, npad_in)
{
}

RMFFT::RMFFT(int n, double *cont, double dx, int npad_in)
      :DataFFT(n, dx, npad_in)
{
  /* fft of cont setup only once */
  memcpy(data_real, cont, nd*sizeof(double));
  fftw_execute(pdata);
}
    
RMFFT::RMFFT(Data& cont, int npad_in):DataFFT(cont, npad_in)
{
  memcpy(data_real, cont.flux, nd*sizeof(double));
  fftw_execute(pdata);
}

void RMFFT::set_data(Data & cont)
{
  memcpy(data_real, cont.flux, cont.size*sizeof(double));
  fftw_execute(pdata);
}

void RMFFT::set_data(double *data, int n)
{
  memcpy(data_real, data, n*sizeof(double));
  fftw_execute(pdata);
}

/* convolution with resp, output to conv */
void RMFFT::convolve(const double *resp, int n, double *conv)
{
  /* fft of resp */
  memcpy(resp_real, resp, n * sizeof(double));
  fftw_execute(presp);
  
  DataFFT::convolve_simple(conv);
  return;
}

/* convolution with resp, output to conv */
void RMFFT::convolve_bg(const double *resp, int n, double *conv, double bg)
{
  /* fft of resp */
  memcpy(resp_real, resp, n * sizeof(double));
  fftw_execute(presp);
  
  DataFFT::convolve_simple(conv);

  int i;
  for(i=0; i<nd; i++)
  {
    conv[i] += bg;
  }
  return;
}

/* convolution with resp, output to conv */
void RMFFT::convolve_bg(const double *resp, int n, int ipositive, double *conv, double bg)
{
  /* fft of resp */
  set_resp_real(resp, n, ipositive);
  
  DataFFT::convolve_simple(conv);

  int i;
  for(i=0; i<nd; i++)
  {
    conv[i] += bg;
  }
  return;
}

/*==================================================================*/
/* class PixonFFT */
PixonFFT::PixonFFT()
{
  npixon_size_max = ipixon_min = 0;
  pixon_sizes = NULL;
  pixon_sizes_num = NULL;
  conv_tmp = NULL;
}
PixonFFT::PixonFFT(int npixel_in, int npixon_size_max_in)
      :DataFFT(npixel_in, 1.0, npixon_size_max_in*pixon_size_factor), npixon_size_max(npixon_size_max_in)
{
  int i;

  ipixon_min = npixon_size_max-1;
  pixon_sizes = new double[npixon_size_max];
  pixon_sizes_num = new double[npixon_size_max];
  conv_tmp = new double[nd];
  for(i=0; i<npixon_size_max; i++)
  {
    pixon_sizes[i] = (i+1)*1.0/pixon_sub_factor;
    pixon_sizes_num[i] = 0;
  }
  /* assume that all pixels have the largest pixon size */
  pixon_sizes_num[ipixon_min] = npixel_in;
}

PixonFFT::~PixonFFT()
{
  if(npixon_size_max > 0)
  {
    npixon_size_max = 0;
    delete[] pixon_sizes;
    delete[] pixon_sizes_num;
    delete[] conv_tmp;
  }
}

void PixonFFT::convolve(const double *pseudo_img, int *pixon_map, double *conv)
{
  int ip, j;
  double psize, norm;

  /* fft of pseudo image */
  memcpy(data_real, pseudo_img, nd*sizeof(double));
  fftw_execute(pdata);

  /* loop over all pixon sizes */
  for(ip=ipixon_min; ip<npixon_size_max; ip++)
  {
    if(pixon_sizes_num[ip] > 0)
    {
      psize = pixon_sizes[ip];
      /* setup resp */
      norm = 0.0;
      for(j=0; j<nd_fft/2; j++)
      {
        resp_real[j] = pixon_function(j, 0, psize);
        norm += resp_real[j];
      }
      for(j=nd_fft-1; j>=nd_fft/2; j--)
      {
        resp_real[j] = pixon_function(j, nd_fft, psize);
        norm += resp_real[j];
      }
      fftw_execute(presp);
      
      DataFFT::convolve_simple(conv_tmp);
      for(j=0; j<nd; j++)
      {
        if(pixon_map[j] == ip)
          conv[j] = conv_tmp[j] / norm;
      }
    }
  }
}

void PixonFFT::convolve_pixon_diff_low(const double *pseudo_img, int *pixon_map, double *conv)
{
  int ip, j;
  double psize, psize_low;

  /* fft of pseudo image */
  memcpy(data_real, pseudo_img, nd*sizeof(double));
  fftw_execute(pdata);

  /* loop over all pixon sizes */
  for(ip=ipixon_min; ip<npixon_size_max; ip++)
  {
    if(pixon_sizes_num[ip] > 0)
    {
      psize = pixon_sizes[ip];
      psize_low = pixon_sizes[ip-1];
      /* setup resp */
      for(j=0; j<nd_fft/2; j++)
      {
        resp_real[j] = pixon_function(j, 0, psize) - pixon_function(j, 0, psize_low);
      }
      for(j=nd_fft-1; j>=nd_fft/2; j--)
      {
        resp_real[j] = pixon_function(j, nd_fft, psize) - pixon_function(j, nd_fft, psize_low);
      }
      fftw_execute(presp);
      
      DataFFT::convolve_simple(conv_tmp);
      for(j=0; j<nd; j++)
      {
        if(pixon_map[j] == ip)
          conv[j] = conv_tmp[j];
      }
    }
  }
}

void PixonFFT::convolve_pixon_diff_up(const double *pseudo_img, int *pixon_map, double *conv)
{
  int ip, j;
  double psize, psize_up;

  /* fft of pseudo image */
  memcpy(data_real, pseudo_img, nd*sizeof(double));
  fftw_execute(pdata);

  /* loop over all pixon sizes */
  for(ip=ipixon_min; ip<npixon_size_max; ip++)
  {
    if(pixon_sizes_num[ip] > 0)
    {
      psize = pixon_sizes[ip];
      psize_up = pixon_sizes[ip+1];
      /* setup resp */
      for(j=0; j<nd_fft/2; j++)
      {
        resp_real[j] = pixon_function(j, 0, psize) - pixon_function(j, 0, psize_up);
      }
      for(j=nd_fft-1; j>=nd_fft/2; j--)
      {
        resp_real[j] = pixon_function(j, nd_fft, psize) - pixon_function(j, nd_fft, psize_up);
      }
      fftw_execute(presp);
      
      DataFFT::convolve_simple(conv_tmp);
      for(j=0; j<nd; j++)
      {
        if(pixon_map[j] == ip)
          conv[j] = conv_tmp[j];
      }
    }
  }
}

/* reduce the minimum pixon size */
void PixonFFT::reduce_pixon_min()
{
  if(ipixon_min > 0)
  {
    ipixon_min--;
  }
  else 
  {
    cout<<"reach minimumly allowed pixon sizes!"<<endl;
    exit(0);
  }
}

/* reduce the minimum pixon size */
void PixonFFT::increase_pixon_min()
{
  if(ipixon_min < npixon_size_max-1)
  {
    ipixon_min++;
  }
  else 
  {
    cout<<"reach maximumly allowed pixon sizes!"<<endl;
    exit(0);
  }
}

int PixonFFT::get_ipxion_min()
{
  return ipixon_min;
}

/*==================================================================*/
/* class PixonUniFFT */
PixonUniFFT::PixonUniFFT()
{
  npixon_size_max = ipixon_min = 0;
  pixon_sizes = NULL;
}
PixonUniFFT::PixonUniFFT(int npixel_in, int npixon_size_max_in)
      :DataFFT(npixel_in, 1.0, npixon_size_max_in*pixon_size_factor), npixon_size_max(npixon_size_max_in)
{
  int i;

  ipixon_min = npixon_size_max-1;
  pixon_sizes = new double[npixon_size_max];
  for(i=0; i<npixon_size_max; i++)
  {
    pixon_sizes[i] = (i+1)*1.0/pixon_sub_factor;
  }
}

PixonUniFFT::~PixonUniFFT()
{
  if(npixon_size_max > 0)
  {
    npixon_size_max = 0;
    delete[] pixon_sizes;
  }
}

void PixonUniFFT::convolve(const double *pseudo_img, int ipixon, double *conv)
{
  int ip, j;
  double psize, norm;

  /* fft of pseudo image */
  memcpy(data_real, pseudo_img, nd*sizeof(double));
  fftw_execute(pdata);

  psize = pixon_sizes[ipixon];
  /* setup resp */
  norm = 0.0;
  for(j=0; j<nd_fft/2; j++)
  {
    resp_real[j] = pixon_function(j, 0, psize);
    norm += resp_real[j];
  }
  for(j=nd_fft-1; j>=nd_fft/2; j--)
  {
    resp_real[j] = pixon_function(j, nd_fft, psize);
    norm += resp_real[j];
  }
  fftw_execute(presp);
  
  DataFFT::convolve_simple(conv);
  for(j=0; j<nd; j++)
  {
    conv[j] = conv[j] / norm;
  }
}

/* reduce the minimum pixon size */
void PixonUniFFT::reduce_pixon_min()
{
  if(ipixon_min > 0)
  {
    ipixon_min--;
  }
  else 
  {
    cout<<"reach minimumly allowed pixon sizes!"<<endl;
    exit(0);
  }
}

/* reduce the minimum pixon size */
void PixonUniFFT::increase_pixon_min()
{
  if(ipixon_min < npixon_size_max-1)
  {
    ipixon_min++;
  }
  else 
  {
    cout<<"reach maximumly allowed pixon sizes!"<<endl;
    exit(0);
  }
}

int PixonUniFFT::get_ipxion_min()
{
  return ipixon_min;
}
/*==================================================================*/
/* class Pixon */

Pixon::Pixon()
{
  npixel = 0;
  bg = 0.0;
  pixon_map = NULL; 
  pixon_map_updated = NULL;
  image = pseudo_image = NULL;
  rmline = NULL;
  itline = NULL;
  residual = NULL;
  grad_chisq = NULL;
  grad_pixon_low = NULL;
  grad_pixon_up = NULL;
  grad_mem = NULL;
  grad_mem_pixon_low = NULL;
  resp_pixon = NULL;
  conv_pixon = NULL;
}

Pixon::Pixon(Data& cont_in, Data& line_in, int npixel_in,  int npixon_size_in, int ipositive_in, double sensitivity_in)
  :cont(cont_in), line(line_in), rmfft(cont_in, fmax(npixel_in-ipositive_in, ipositive_in)), 
   pfft(npixel_in, npixon_size_in), npixel(npixel_in),
   bg(0.0), ipositive(ipositive_in), sensitivity(sensitivity_in)
{
  pixon_map = new int[npixel];
  pixon_map_updated = new bool[npixel];
  image = new double[npixel];
  pseudo_image = new double[npixel];
  rmline = new double[cont.size];
  itline = new double[line.size];
  residual = new double[line.size];
  grad_pixon_low = new double[npixel];
  grad_pixon_up = new double[npixel];
  grad_chisq = new double[npixel+1];
  grad_mem = new double[npixel+1];
  grad_mem_pixon_low = new double[npixel];
  grad_mem_pixon_up = new double[npixel];
  resp_pixon = new double[cont.size];
  conv_pixon = new double[cont.size];

  dt = cont.time[1]-cont.time[0];  /* time interval width of continuum light curve */
  int i;
  for(i=0; i<npixel; i++)
  {
    pixon_map[i] = npixon_size_in-1;  /* set the largest pixon size */
  }

  tau0 = 0.0 - ipositive * dt;
}

Pixon::~Pixon()
{
  if(npixel > 0)
  {
    npixel = 0;
    delete[] pixon_map;
    delete[] pixon_map_updated;
    delete[] image;
    delete[] pseudo_image;
    delete[] rmline;
    delete[] itline;
    delete[] residual;
    delete[] grad_chisq;
    delete[] grad_pixon_low;
    delete[] grad_pixon_up;
    delete[] grad_mem;
    delete[] grad_mem_pixon_low;
    delete[] grad_mem_pixon_up;
    delete[] resp_pixon;
    delete[] conv_pixon;
  }
}

/* interpolate image, note the negative time lags */
double Pixon::interp_image(double t)
{
  int it;
  it = (t - tau0)/dt;

  if(it < 0 || it >= npixel -1)
    return 0.0;

  return image[it] + (image[it+1] - image[it])/dt * (t - (it-ipositive)*dt);
}

/* linear line interplolation  */
double Pixon::interp_line(double t)
{
  int it;

  it = (t - cont.time[0])/dt;

  if(it < 0)
    return rmline[0];
  else if(it >= cont.size -1)
    return rmline[cont.size -1];

  return rmline[it] + (rmline[it+1] - rmline[it])/dt * (t - cont.time[it]);
}

/* linear cont interplolation  */
double Pixon::interp_cont(double t)
{
  int it;

  it = (t - cont.time[0])/dt;

  if(it < 0)
    return cont.flux[0];
  else if(it >= cont.size -1)
    return cont.flux[cont.size-1];

  return cont.flux[it] + (cont.flux[it+1] - cont.flux[it])/dt * (t - cont.time[it]);
}

double Pixon::interp_pixon(double t)
{
  int it;

  it = (t - cont.time[0])/dt;

  if(it < 0)
    return conv_pixon[0];
  else if(it >= cont.size -1)
    return conv_pixon[cont.size-1];

  return conv_pixon[it] + (conv_pixon[it+1] - conv_pixon[it])/dt * (t - cont.time[it]);
}

/* compute rm amd pixon convolutions */
void Pixon::compute_rm_pixon(const double *x)
{
  int i;
  double t;
  /* convolve with pixons */
  for(i=0; i<npixel; i++)
  {
    pseudo_image[i] = exp(x[i]);
  }
  pfft.convolve(pseudo_image, pixon_map, image);
  bg = x[npixel];
  
  /* enforce positive image */
  for(i=0; i<npixel; i++)
  {
    if(image[i] <= 0.0)
      image[i] = EPS;
  }
  
  /* reverberation mapping */
  rmfft.convolve_bg(image, npixel, ipositive, rmline, bg);

  /* interpolation */
  for(i=0; i<line.size; i++)
  {
    t = line.time[i];
    itline[i] = interp_line(t);
    residual[i] = itline[i] - line.flux[i];
  }
}

/* compute chi square */
double Pixon::compute_chisquare(const double *x)
{
  int i;

  /* calculate chi square */
  chisq = 0.0;
  for(i=0; i<line.size; i++)
  {
    chisq += (residual[i] * residual[i])/(line.error[i] * line.error[i]);
  }
  return chisq;
}

/* compute entropy */
double Pixon::compute_mem(const double *x)
{
  double Itot, num, alpha;
  int i;

  Itot = 0.0;
  for(i=0; i<npixel; i++)
  {
    Itot += image[i];
  }
  
  num = compute_pixon_number();
  alpha = log(num)/log(npixel);

  mem = 0.0;
  for(i=0; i<npixel; i++)
  {
    mem += (image[i]/Itot) * log(image[i]/Itot);
  }
  mem *= 2.0*alpha;
  return mem;
}

/* compute gradient of chi square line */
void Pixon::compute_chisquare_grad(const double *x)
{
  int i, k, j;
  double psize, t, grad_in, grad_out;
  for(i=0; i<npixel; i++)
  {   
    for(j=0; j<npixel; j++)
    {
      psize = pfft.pixon_sizes[pixon_map[j]];
      resp_pixon[j] = pixon_function(j, i, psize);
    }
    rmfft.set_resp_real(resp_pixon, npixel, ipositive);
    //rmfft.convolve(resp_pixon, npixel, conv_pixon);
    rmfft.convolve_simple(conv_pixon);

    grad_out = 0.0;
    for(k=0; k<line.size; k++)
    {
      t = line.time[k];
      grad_in = interp_pixon(t);
      grad_out += grad_in * residual[k]/line.error[k]/line.error[k];
    }
    grad_chisq[i] = grad_out * 2.0 * pseudo_image[i];
  }
  
  /* with respect to background */
  grad_out = 0.0;
  for(k=0; k<line.size; k++)
  {
    grad_out += residual[k]/line.error[k]/line.error[k];
  }
  grad_chisq[npixel] = grad_out * 2.0;
}

/* calculate chisqure gradient with respect to pixon size 
 * when pixon size decreases, chisq decreases, 
 * so chisq gradient is positive 
 */
void Pixon::compute_chisquare_grad_pixon_low()
{
  int i, k, j;
  double psize, psize_low, t, grad_in, grad_out, K, grad_size, tau;
  int jrange1, jrange2;

  pfft.convolve_pixon_diff_low(pseudo_image, pixon_map, conv_pixon);
  for(i=0; i<npixel; i++)
  {   
    grad_size = conv_pixon[i];
    grad_out = 0.0;
    tau = tau0 + i * dt;
    for(k=0; k<line.size; k++)
    {
      t = line.time[k];
      grad_in = interp_cont(t - tau);
      grad_out += grad_in * residual[k]/line.error[k]/line.error[k];
    }
    grad_pixon_low[i] = grad_out * 2.0 * grad_size;
  }
}

/* calculate chisqure gradient with respect to pixon size 
 * when pixon size increases, chisq increases, 
 * so chisq gradient is positive too
 */
void Pixon::compute_chisquare_grad_pixon_up()
{
  int i, k, j;
  double psize, psize_up, t, tau, grad_in, grad_out, grad_size, K;
  int jrange1, jrange2;

  pfft.convolve_pixon_diff_up(pseudo_image, pixon_map, conv_pixon);
  for(i=0; i<npixel; i++)
  {
    grad_size = conv_pixon[i];
    grad_out = 0.0;
    tau = tau0 + i * dt;
    for(k=0; k<line.size; k++)
    {
      t = line.time[k];
      grad_in = interp_cont(t - tau);
      grad_out += grad_in * residual[k]/line.error[k]/line.error[k];
    }
    grad_pixon_up[i] = grad_out * 2.0 * grad_size;
  }
}

/* calculate entropy gradient
 *
 */
void Pixon::compute_mem_grad(const double *x)
{
  double Itot, num, alpha, grad_in, psize, K;
  int i, j;
  Itot = 0.0;
  for(i=0; i<npixel; i++)
  {
    Itot += image[i];
  }
  num = compute_pixon_number();
  alpha = log(num)/log(npixel);
  
  for(i=0; i<npixel; i++)
  {       
    grad_in = 0.0;
    for(j=0; j<npixel; j++)
    {
      psize = pfft.pixon_sizes[pixon_map[j]];
      K = pixon_function(j, i, psize);
      grad_in += (1.0 + log(image[j]/Itot)) * K;
    }
    grad_mem[i] = 2.0 * alpha * pseudo_image[i] * grad_in / Itot;
  }
  /* with respect to bg is zero */
  grad_mem[npixel] = 0.0;
}

/* calculate entropy gradient with respect to pixon size 
 * when pixon size decreases
 */
void Pixon::compute_mem_grad_pixon_low()
{
  double Itot, num, alpha, grad_in, psize, psize_low, K;
  int i, j, jrange1, jrange2, grad_size;
  Itot = 0.0;
  for(i=0; i<npixel; i++)
  {
    Itot += image[i];
  }
  num = compute_pixon_number();
  alpha = log(num)/log(npixel);

  pfft.convolve_pixon_diff_low(pseudo_image, pixon_map, conv_pixon);
  for(i=0; i<npixel; i++)
  {       
    grad_size = conv_pixon[i];
    grad_in = (1.0 + log(image[i]/Itot));
    grad_mem_pixon_low[i] = 2.0* alpha * grad_in * grad_size / Itot;
  }
}

/* calculate entropy gradient with respect to pixon size 
 * when pixon size increases
 */
void Pixon::compute_mem_grad_pixon_up()
{
  double Itot, num, alpha, grad_in, psize, psize_up, K;
  int i, j, jrange1, jrange2, grad_size;
  Itot = 0.0;
  for(i=0; i<npixel; i++)
  {
    Itot += image[i];
  }
  num = compute_pixon_number();
  alpha = log(num)/log(npixel);

  pfft.convolve_pixon_diff_up(pseudo_image, pixon_map, conv_pixon);
  for(i=0; i<npixel; i++)
  {       
    grad_size = conv_pixon[i];
    grad_in = (1.0 + log(image[i]/Itot));
    grad_mem_pixon_up[i] = 2.0* alpha * grad_in * grad_size / Itot;
  }
}

double Pixon::compute_pixon_number()
{
  int i;
  double num, psize;
    
  num = 0.0;
  for(i=0; i<npixel; i++)
  {
    psize = pfft.pixon_sizes[pixon_map[i]];
    num += pixon_norm(psize);
  }
  return num;
}

void Pixon::reduce_pixon_map_all()
{
  int i;
  pfft.pixon_sizes_num[pfft.ipixon_min] = 0;
  pfft.reduce_pixon_min();
  pfft.pixon_sizes_num[pfft.ipixon_min] = npixel;
  for(i=0; i<npixel; i++)
  {
    pixon_map[i]--;
  }
}

bool Pixon::reduce_pixon_map_uniform()
{
  int i;
  bool flag = false;
  for(i=0; i<npixel; i++)
  {
    if(pixon_map[i] > pixon_map_low_bound + 1)
    {
      reduce_pixon_map(i);
      flag = true;
    }
  }

  return flag;
}

void Pixon::increase_pixon_map_all()
{
  int i;
  pfft.pixon_sizes_num[pfft.ipixon_min] = 0;
  pfft.increase_pixon_min();
  pfft.pixon_sizes_num[pfft.ipixon_min] = npixel;
  for(i=0; i<npixel; i++)
  {
    pixon_map[i]++;
  }
}

void Pixon::reduce_pixon_map(int ip)
{
  pfft.pixon_sizes_num[pixon_map[ip]]--;
  pixon_map[ip]--;
  pfft.pixon_sizes_num[pixon_map[ip]]++;
  if(pfft.ipixon_min > pixon_map[ip])
  {
    pfft.ipixon_min = pixon_map[ip];
  }
}

void Pixon::increase_pixon_map(int ip)
{
  pfft.pixon_sizes_num[pixon_map[ip]]--;
  pixon_map[ip]++;
  pfft.pixon_sizes_num[pixon_map[ip]]++;
}

bool Pixon::update_pixon_map()
{
  int i;
  double psize, psize_low, dnum_low, num;
  bool flag=false;

  cout<<"update pixon map."<<endl;
  compute_chisquare_grad_pixon_low();
  compute_mem_grad_pixon_low();
  for(i=0; i<npixel; i++)
  {
    pixon_map_updated[i] = false;
    if(pixon_map[i] > pixon_map_low_bound + 1)
    {
      psize = pfft.pixon_sizes[pixon_map[i]];
      psize_low = pfft.pixon_sizes[pixon_map[i]-1];
      num = pixon_norm(psize);
      dnum_low = pixon_norm(psize_low) - num;
      if( grad_pixon_low[i] + grad_mem_pixon_low[i] > dnum_low  * (1.0 + sensitivity/sqrt(2.0*num)))
      {
        reduce_pixon_map(i);
        pixon_map_updated[i] = true;
        cout<<"decrease "<< i <<"-th pixel to "<<pfft.pixon_sizes[pixon_map[i]]<<endl;
        flag=true;
      }
    }
  }
  return flag;
}

bool Pixon::increase_pixon_map()
{
  int i;
  double psize, psize_up, dnum_up, num;
  bool flag=false;

  cout<<"update pixon map."<<endl;
  compute_chisquare_grad_pixon_up();
  compute_mem_grad_pixon_up();
  for(i=0; i<npixel; i++)
  {
    if(pixon_map[i] < pfft.npixon_size_max - 1)
    {
      psize = pfft.pixon_sizes[pixon_map[i]];
      psize_up = pfft.pixon_sizes[pixon_map[i]+1];
      num = pixon_norm(psize);
      dnum_up = num - pixon_norm(psize_up);
      if(grad_pixon_up[i] + grad_mem_pixon_up[i] <= dnum_up )
      {
        increase_pixon_map(i);
        cout<<"increase "<< i <<"-th pixel to "<<pfft.pixon_sizes[pixon_map[i]]<<endl;
        flag=true;
      }
    }
  }
  return flag;
}
/*==================================================================*/
/* pixon functions */
PixonFunc pixon_function;
PixonNorm pixon_norm;

/* function for nlopt */
double func_nlopt(const vector<double> &x, vector<double> &grad, void *f_data)
{
  Pixon *pixon = (Pixon *)f_data;
  double chisq, mem;

  pixon->compute_rm_pixon(x.data());
  if (!grad.empty()) 
  {
    int i;
    
    pixon->compute_chisquare_grad(x.data());
    pixon->compute_mem_grad(x.data());
    
    for(i=0; i<(int)grad.size(); i++)
      grad[i] = pixon->grad_chisq[i] + pixon->grad_mem[i];
  }
  chisq = pixon->compute_chisquare(x.data());
  mem = pixon->compute_mem(x.data());
  return chisq + mem;
}

/* function for tnc */
int func_tnc(double x[], double *f, double g[], void *state)
{
  Pixon *pixon = (Pixon *)state;
  int i;
  double chisq, mem;

  pixon->compute_rm_pixon(x);
  pixon->compute_chisquare_grad(x);
  pixon->compute_mem_grad(x);
  
  chisq = pixon->compute_chisquare(x);
  mem = pixon->compute_mem(x);

  *f = chisq + mem;

  for(i=0; i<pixon->npixel+1; i++)
    g[i] = pixon->grad_chisq[i] + pixon->grad_mem[i];

  return 0;
}