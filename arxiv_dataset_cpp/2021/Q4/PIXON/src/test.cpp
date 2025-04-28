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
#include "tnc.h"

using namespace std;

void test_nlopt()
{
  Data cont, line;
  string fcon, fline;
  fcon="data/con.txt";
  cont.load(fcon);
  fline = "data/line.txt";
  line.load(fline);

  int npixel = cont.size*0.3, npixon = 8;
  int i;
  Pixon pixon(cont, line, npixel, npixon);
  nlopt::opt opt0(nlopt::GN_ISRES, npixel);
  nlopt::opt opt1(nlopt::LN_BOBYQA, npixel);
  nlopt::opt opt2(nlopt::LD_SLSQP, npixel);
  vector<double> x(npixel), x0(npixel), x_old(npixel);
  void *args = (void *)&pixon;
  double minf_old, num_old, minf, num;
  double *image=new double[npixel], *itline=new double[line.size];
  
  for(i=0; i<npixel; i++)
  {
    x[i] = log(1.0/(npixel * pixon.dt));
    //x[i] = log(1.0/sqrt(2.0*M_PI)/10.0 * exp( - 0.5*pow((pixon.dt*i - 300.0)/10.0, 2.0) ) + 1.0e-10);
  }
  x0 = x;

  opt0.set_min_objective(func_nlopt, args);
  opt0.set_lower_bounds(-100.0);
  opt0.set_upper_bounds(1.0);
  opt0.set_maxeval(10000);
  //opt0.set_xtol_rel(1e-11);
  //opt0.set_ftol_rel(1e-11);
  opt0.set_ftol_abs(1.0e-15);
  opt0.set_xtol_abs(1.0e-15);

  opt1.set_min_objective(func_nlopt, args);
  opt1.set_lower_bounds(-100.0);
  opt1.set_upper_bounds(1.0);
  opt1.set_maxeval(10000);
  //opt1.set_xtol_rel(1e-11);
  //opt1.set_ftol_rel(1e-11);
  opt1.set_ftol_abs(1.0e-15);
  opt1.set_xtol_abs(1.0e-15);

  opt2.set_min_objective(func_nlopt, args);
  opt2.set_lower_bounds(-100.0);
  opt2.set_upper_bounds(1.0);
  opt2.set_maxeval(10000);
  //opt2.set_xtol_rel(1e-11);
  //opt2.set_ftol_rel(1e-11);
  opt2.set_ftol_abs(1.0e-15);
  opt2.set_xtol_abs(1.0e-15);

  num_old = pixon.compute_pixon_number();
  //opt0.optimize(x, minf_old);
  opt1.optimize(x, minf_old);
  opt2.optimize(x, minf_old);
  cout<<minf_old<<"  "<<num_old<<endl;
  memcpy(image, pixon.image, npixel*sizeof(double));
  memcpy(itline, pixon.itline, line.size*sizeof(double));
  while(npixon>1)
  {
    npixon--;
    cout<<"npixon:"<<npixon<<endl;

    pixon.reduce_pixon_map_all();
    num = pixon.compute_pixon_number();
    //opt0.optimize(x, minf);
    opt1.optimize(x, minf);
    opt2.optimize(x, minf);
    cout<<minf<<"  "<<num<<endl;

    if(minf_old - minf < num - num_old)
      break;

    num_old = num;
    minf_old = minf;
    memcpy(image, pixon.image, npixel*sizeof(double));
    memcpy(itline, pixon.itline, line.size*sizeof(double));
    x_old = x;
  }
  
  ofstream fout;
  double xr;
  fout.open("data/resp_nlopt.txt");
  for(i=0; i<npixel; i++)
  {
    xr = exp(-0.5 * pow(pixon.dt*i-300.0, 2)/(50.0*50.0)) * 1.0/sqrt(2.0*M_PI)/50.0;
    fout<<pixon.dt*i<<"  "<<exp(x_old[i])<<"  "<<image[i]<<"  "<<pixon_function(pixon.dt*i, 300.0, 50.0)<<"  "<<xr<<endl;
  }
  fout.close();

  fout.open("data/line_sim_nlopt.txt");
  for(i=0; i<line.size; i++)
  {
    fout<<line.time[i]<<"  "<<itline[i]<<endl;
  }
  fout.close();
  
  delete[] image;
  delete[] itline;
}


void test()
{
  Data cont, line;
  string fcon, fline;
  fcon={"data/con.txt"};
  cont.load(fcon);
  fline = "data/line.txt";
  line.load(fline);

  int i, nr = 100, np=13;
  double *resp, *delay,*conv;
  double sigma = 20.0;
  double *pseudo_img, *conv_img;
  int *pixon_map;
  
  default_random_engine generator;
  uniform_int_distribution<int> distribution(0,np-1);

  resp = new double[nr];
  delay = new double[nr];
  conv = new double[cont.size];

  pseudo_img = new double[nr];
  pixon_map = new int[nr];
  conv_img = new double[nr];

  for(i=0; i<nr; i++)
  {
    delay[i] = cont.time[i] - cont.time[0];
    resp[i] = 1.0/sqrt(2.0*M_PI)/sigma * exp(-0.5 * (delay[i] - 200.0) * (delay[i] - 200.0)/sigma/sigma);
  }
  
  RMFFT rmfft(cont);
  rmfft.convolve(resp, nr, conv);
  
  for(i=0; i<nr; i++)
  {
    pseudo_img[i] = resp[i];
    pixon_map[i] = np-1;
  }
  PixonFFT pfft(nr, np);
  pfft.convolve(pseudo_img, pixon_map, conv_img);

  ofstream fout;
  fout.open("data/conv.txt");
  for(i=0; i<cont.size; i++)
  {
    fout<<cont.time[i]<<"  "<<conv[i]<<endl;
  }
  fout.close();
  fout.open("data/resp.txt");
  for(i=0; i<nr; i++)
  {
    fout<<delay[i]<<" "<<resp[i]<<endl;
  }
  fout.close();

  fout.open("data/conv_img.txt");
  for(i=0; i<nr; i++)
  {
    fout<<pseudo_img[i]<<"  "<<conv_img[i]<<endl;
  }
  fout.close();

  delete[] resp;
  delete[] delay;
  delete[] conv;

  delete[] pseudo_img;
  delete[] pixon_map;
  delete[] conv_img;
}