#include<iostream>
#include<vector>
#include<cmath>
#include<cstdlib>
#include"vvector.h"
#include <iomanip>
#include <fstream>


extern "C" void dgetrf_(int*, int*, double*, int*, int*, int*);
extern "C" void dgetri_(int* , double* , int* , int* , double* , int* , int* );
extern "C" void dgetrs_(char*, int*, int*, double*, int*, int*, double*, int*, int*);
extern "C" void dgesvd_(char*, char*, int*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int*, int*);
//extern "C" void dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

using namespace std;
const double pi = 3.1415926535897932384626;

//basis funtions
double phi(int m,vvector<double> X){
    double x=X[0];
    double y=X[1];
    if(m == 0){
        return 1.0;
    }
    else if(m == 1){
        return x;
    }
    else if(m == 2){
        return y;
    }
    else if(m == 3){
        return x*x;
    }
    else if(m == 4){
        return x*y;
    }
    else if(m == 5){
        return y*y;
    }
    else if(m == 6){
        return x*x*x;
    }
    else if(m == 7){
        return x*x*y;
    }
    else if(m == 8){
        return x*y*y;
    }
    else if(m == 9){
        return y*y*y;
    }
    else if(m == 10){
        return x*x*x*x;
    }
    else if(m == 11){
        return x*x*x*y;
    }
    else if(m == 12){
        return x*x*y*y;
    }
    else if(m == 13){
        return x*y*y*y;
    }
    else if(m == 14){
        return y*y*y*y;
    }
    else if(m == 15){
        return x*x*x*x*x;
    }
    else if(m == 16){
        return x*x*x*x*y;
    }
    else if(m == 17){
        return x*x*x*y*y;
    }
    else if(m == 18){
        return x*x*y*y*y;
    }
    else if(m == 19){
        return x*y*y*y*y;
    }
    else if(m == 20){
        return y*y*y*y*y;
    }
}

double Iphi(int m, double delta){
    if (m==0){
    return pi*pow(delta,2);
    }
    else if (m == 3){
        return 1.0/4.0*pi*pow(delta,4);
    }
    else if (m == 5){
        return 1.0/4.0*pi*pow(delta,4);
    }
    else if (m == 10){
        return 1.0/8.0*pi*pow(delta,6);
    }
    else if (m == 12){
        return 1.0/24.0*pi*pow(delta,6);
    }
    else if (m == 14){
        return 1.0/8.0*pi*pow(delta,6);
    }
    else {
        return 0.0;
    }
}

double weight(double r,double epsilon){
    if(r > epsilon) return 0;
    else return pow((1-r/epsilon),4);
}

void inverse(double* A, int N)
{
    int *IPIV = new int[N+1];
    int LWORK = N*N;
    double *WORK = new double[LWORK];
    int INFO;
    dgetrf_(&N,&N,A,&N,IPIV,&INFO);
    dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);
    delete[] IPIV;
    delete[] WORK;
}

//basis funtions
//
/////////////////////////////////////////////////////////////////////////
//case 1: quadratic in space, static in time
double u_exact(double x, double y){
  // nonlocal limit example x^6+y^6
  //return x*x*x*x*x*x+y*y*y*y*y*y;

  // local limit example 
  return cos(x)*cos(y);
}

double Ffun(double x,double y, double delta){
  // analytical forcing term for nonlocal limit example x^6+y^6
  //return -8./pi/pow(delta,4)*pi*((5+2*x)*(5./32*pow(delta,8)+15./8*pow(delta,6)*(x*x+y*y)+15./4*pow(delta,4)*(x*x*x*x+y*y*y*y))+(15./32*pow(delta,8)+5./2*pow(delta,6)*x*x*x+3./2*pow(delta,4)*x*x*x*x*x));
  
  // analytical forcing term for local limit example cosx*cosy
  return 4*cos(x)*cos(y)+4*sin(x)*cos(x)*sin(y)*cos(y);
}

double diff_coef(double x, double y){
  // define local diffusion coefficient
  return 2+sin(x)*sin(y);
}

double nonlocal_diff_coef(double x1, double x2, double y1, double y2){
  // define nonlocal diffusion coefficient
  return 5+x1+y1;
}


////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////



vector<vector<double>>  Preprocess(double* x_0, double* y_0, int N,int k, double dhratio,vector<vector<int>> nei, int basedim)
{
  int cenx,cenx1;
  vvector<double> xi(2),zeta(2),gamma(2);
  double eta;
  vector<vector<double>> omega((N+1)*(N+1));  
    double h = 1./N;
    for(int i=0;i<N+1;i++){
        for(int j=0;j<N+1;j++){
            cenx = (i+k)*(N+1+2*k)+j+k;
            omega[i*(N+1)+j].resize(nei[i*(N+1)+j].size());
            // generate quadrature weights (double* omega)
            double* B = new double[basedim*(nei[i*(N+1)+j].size())];
            double* g = new double[basedim]; 
            for (int s = 0; s< nei[i*(N+1)+j].size(); s++){
              xi[0] = x_0[nei[i*(N+1)+j][s]] - x_0[cenx];
              xi[1] = y_0[nei[i*(N+1)+j][s]] - y_0[cenx];
              for(int m=0;m<basedim;m++){
                    //eta = phi(m,gamma) - phi(m,zeta);
                    eta = phi(m,xi);
                    B[m*nei[i*(N+1)+j].size()+s] = 8.0*eta/pi/pow(dhratio*h,4);
                }
                /*
                B[0*nei[i*(N+1)+j].size()+s] = 8/pi/pow(dhratio*h,4)*1; 
                B[1*nei[i*(N+1)+j].size()+s] = 8/pi/pow(dhratio*h,4)*(x_0[nei[i*(N+1)+j][s]]-x_0[cenx]); 
                B[2*nei[i*(N+1)+j].size()+s] = 8/pi/pow(dhratio*h,4)*(y_0[nei[i*(N+1)+j][s]]-y_0[cenx]); 
                B[3*nei[i*(N+1)+j].size()+s] = 8/pi/pow(dhratio*h,4)*pow(x_0[nei[i*(N+1)+j][s]]-x_0[cenx],2); 
                B[4*nei[i*(N+1)+j].size()+s] = 8/pi/pow(dhratio*h,4)*pow(y_0[nei[i*(N+1)+j][s]]-y_0[cenx],2); 
                B[5*nei[i*(N+1)+j].size()+s] = 8/pi/pow(dhratio*h,4)*(x_0[nei[i*(N+1)+j][s]]-x_0[cenx])*(y_0[nei[i*(N+1)+j][s]]-y_0[cenx]);
                */
            }

            for(int m=0;m<basedim;m++){
              g[m]=8/pi/pow(dhratio*h,4)*Iphi(m, dhratio*h);
            }
            /*
            g[0] = 8/pi/pow(dhratio*h,4)*pi*(dhratio*h)*(dhratio*h);
            g[1] = 0; 
            g[2] = 0; 
            g[3] = 8/pi/pow(dhratio*h,4)*1/4*pi*pow(dhratio*h, 4);
            g[4] = 8/pi/pow(dhratio*h,4)*1/4*pi*pow(dhratio*h, 4);
            g[5] = 0;
            */
             
            double* S = new double[basedim*basedim]; 
            for (int a = 0; a<basedim; a++){
              for (int b = 0; b<basedim; b++){
                S[a*basedim+b]=0;
                for (int ss = 0; ss<nei[i*(N+1)+j].size(); ss++){
                  S[a*basedim+b] += B[a*nei[i*(N+1)+j].size()+ss]*B[b*nei[i*(N+1)+j].size()+ss];
                }
              }
            }

            //pseudo_inverse(S, 6); 
            inverse(S,basedim);
            double* K = new double[nei[i*(N+1)+j].size()*basedim]; 
    
            for (int a = 0; a<nei[i*(N+1)+j].size(); a++){
              for (int b = 0; b<basedim; b++){
                K[a*basedim+b]=0;
                for (int ss = 0; ss<basedim; ss++){
                  K[a*basedim+b] += B[ss*nei[i*(N+1)+j].size()+a]*S[ss*basedim+b];
                }
              }
            }

            for (int a = 0; a<nei[i*(N+1)+j].size(); a++){
              omega[i*(N+1)+j][a]=0;
              for (int ss = 0; ss<basedim; ss++){
                  omega[i*(N+1)+j][a] += K[a*basedim+ss]*g[ss];  
                }
            } 
        }
    }
    return omega;

}





int main(int argc, char* argv[]){
    if(argc < 3){
        cerr<<"Error: Not enough input variables"<<" ";
        cerr<<" Usage: <N>  <dhratio>"<<endl;
     }


    int N = atoi(argv[1]);                        
    double dhratio = atof(argv[2]);
    int porder = atoi(argv[3]);
    int casee = atoi(argv[4]); //casee=0 corresponds to convergence to local limit, casee=1 corresponds to convergence to nonlocal limit, 
    if(casee != 0 && casee != 1){
      cout << 'Error: case can only be 0(convergence to local limit) or 1(convergence to nonlocal limit)' << endl; 
    }
    int basedim; 
    if(porder==2){
      basedim=6;
    }
    else if (porder==3){
      basedim=10; 
    }
    else if (porder==4){
      basedim=15;
    }
    else{
      basedim=21;
    }
    double h = 1./N;            // lattice constant, all the nodes are evenly distributed 
    int k = floor(dhratio);
    int dim = (N+1+2*k)*(N+1+2*k);
    //int basedim = 10;      //linear basis funtions
    double* x_0 = new double [dim];    //initial configuration 
    double* y_0 = new double [dim];
    double* u = new double [dim];   //displacement in x direction and y direction 
    double* u0 = new double [dim];
    double* rhs = new double [dim];
    double* stiffKT = new double[dim*dim];
    int* IPIV = new int[dim];
    int info;
    int nrhs = 1;
    char C='T';
    char D = 'A';
    double a_i, a_j, a_ij;

    vector< vector<int> > nei((N+1)*(N+1)); //neighborlist
  vvector<double> xi(2),zeta(2),gamma(2), eta(2);
    
    //Build the neighborhood list 
  //set initial configuration 
  for(int i=0;i<N+1+2*k;i++){
    for(int j=0;j<N+1+2*k;j++){
      x_0[i*(N+1+2*k)+j] = (j-k)*h;
      y_0[i*(N+1+2*k)+j] = (i-k)*h;
    }
  }

  //random perturbation 
    
  double random_coe = 0.1;
  for(int i=0;i<N+1+2*k;i++){
    for(int j=0;j<N+1+2*k;j++){
        x_0[i*(N+1+2*k)+j] += (2*rand()/double(RAND_MAX)-1)*random_coe*h;
        y_0[i*(N+1+2*k)+j] += (2*rand()/double(RAND_MAX)-1)*random_coe*h ;
    }
  }


  for(int i=0;i<N+1+2*k;i++){
    for(int j=0;j<N+1+2*k;j++){
      u0[i*(N+1+2*k)+j]=u_exact(x_0[i*(N+1+2*k)+j],y_0[i*(N+1+2*k)+j]);
    }
  }


  for(int i=0;i<dim*dim;i++){
    stiffKT[i] = 0;
  }

  //set neighborhood list for every particle 
  for(int i=0;i<N+1;i++){
    for(int j=0;j<N+1;j++){
      for(int s1=-k;s1<k+1;s1++){
        for(int s2=-k;s2<k+1;s2++){
          if((i+k+s1)*(N+1+2*k)+j+k+s2 < (N+1+2*k)*(N+1+2*k) && (i+k+s1)*(N+1+2*k)+j+k+s2 > -1){
            xi[0] = x_0[(i+k+s1)*(N+1+2*k)+j+k+s2] - x_0[(i+k)*(N+1+2*k)+j+k];
            xi[1] = y_0[(i+s1+k)*(N+1+2*k)+j+k+s2] - y_0[(i+k)*(N+1+2*k)+j+k];
            if(xi.Norm2() < dhratio*h){
              nei[i*(N+1)+j].push_back((i+k+s1)*(N+1+2*k)+j+k+s2);
            }
          }
        }
      }       
    }
  }
    
    //generate quadrature weights 
    vector<vector<double>> omega = Preprocess(x_0,y_0,N,k,dhratio,nei,basedim);

    

    //assemble matrix
    for(int i=0;i<N+1;i++){
        for(int j=0;j<N+1;j++){
            int cenx = (i+k)*(N+1+2*k)+j+k;
            for(int s=0;s<nei[i*(N+1)+j].size();s++){
              if (casee==1){
              a_ij=nonlocal_diff_coef(x_0[cenx],y_0[cenx],x_0[nei[i*(N+1)+j][s]],y_0[nei[i*(N+1)+j][s]]);
              }
              else{
                a_i=diff_coef(x_0[cenx],y_0[cenx]);
                a_j=diff_coef(x_0[nei[i*(N+1)+j][s]],y_0[nei[i*(N+1)+j][s]]);
                a_ij=2.0/(1.0/a_i+1.0/a_j);
              }
              stiffKT[cenx*dim+nei[i*(N+1)+j][s]] -= 8./pi/pow(dhratio*h,4)*omega[i*(N+1)+j][s]*a_ij;
              stiffKT[cenx*dim+cenx] += 8./pi/pow(dhratio*h,4)*omega[i*(N+1)+j][s]*a_ij;
            }
            rhs[cenx] = Ffun(x_0[cenx],y_0[cenx], dhratio*h);//+u0[cenx]/Dt;

        }
    }  
  
    //apply boundary condition 
    for(int i=0;i<N+1+2*k;i++){
    if(i < k || i > N+k){
      for(int j=0;j<N+1+2*k;j++){
        int cenx = i*(N+1+2*k)+j;
        stiffKT[cenx*dim+cenx] = 1;
        rhs[cenx] = u_exact(x_0[cenx],y_0[cenx]);
      }
    } 
    else{
      for(int j=0;j<k;j++){
        int cenx = i*(N+1+2*k)+j;
        int cenx1 = i*(N+1+2*k)+N+2*k-j;
        stiffKT[cenx*dim+cenx] = 1;
        stiffKT[cenx1*dim+cenx1] = 1;
        rhs[cenx] = u_exact(x_0[cenx],y_0[cenx]);
        rhs[cenx1] = u_exact(x_0[cenx1],y_0[cenx1]);
      }
    }
  }

  //compute truncation error

  double* ue = new double [dim];
  for(int i=0;i<N+1+2*k;i++){
    for(int j=0;j<N+1+2*k;j++){
      ue[i*(N+1+2*k)+j]=u_exact(x_0[i*(N+1+2*k)+j],y_0[i*(N+1+2*k)+j]);
    }
  }

  double* trun = new double [dim];
  for (int i1=0;i1<dim;i1++){
    trun[i1]=0.0;
    for (int i2=0;i2<dim;i2++){
      trun[i1] += stiffKT[i1*dim+i2]*ue[i2];
    }
  }

  //post process, calculate truncation error
    double trun_error = 0.;
    double trun_errorinf = 0.;
      for(int i=0;i<dim;i++){
        trun_error += pow(trun[i]-rhs[i],2);
        if (fabs(trun[i]-rhs[i])>trun_errorinf){
          //cout << fabs(trun[i]-rhs[i]) << endl;
          trun_errorinf = fabs(trun[i]-rhs[i]);
        }
      }


  dgetrf_(&dim,&dim,stiffKT,&dim,IPIV,&info);
  dgetrs_(&C,&dim,&nrhs,stiffKT,&dim, IPIV,rhs,&dim,&info);
  
  for(int i=0;i<dim;i++){
      u0[i] = rhs[i];
  }
    
    //post process, calculate error
    double error = 0.;
      for(int i=0;i<N+1+2*k;i++){
        for(int j=0;j<N+1+2*k;j++){
          error += pow(u0[i*(N+1+2*k)+j]-u_exact(x_0[i*(N+1+2*k)+j],y_0[i*(N+1+2*k)+j]),2);
        }
      }
    double errorinf = 0;
      for(int i=0;i<N+1+2*k;i++){
        for(int j=0;j<N+1+2*k;j++){
          if(fabs(u0[i*(N+1+2*k)+j]-u_exact(x_0[i*(N+1+2*k)+j],y_0[i*(N+1+2*k)+j])) > errorinf){
            errorinf = fabs(u0[i*(N+1+2*k)+j]-u_exact(x_0[i*(N+1+2*k)+j],y_0[i*(N+1+2*k)+j])); 
          }
        }
      }


      error /= (N+1)*(N+1);
    error = sqrt(error);
      cout<<" error L2: "<<error<<"  errorinf: "<<errorinf<<endl;

      cout<<" truncation error L2: "<< sqrt(trun_error)/(N+1) <<"  truncation errorinf: "<< trun_errorinf <<endl;

      //ofstream ufile,ufile1;
      //ufile1.open("vel_x");
      //for(int i=0;i<N+1;i++){
      //  int j=N;
      //  int cenx = (i+k)*(N+1+2*k)+j+k;
      //  ufile1<<du0[cenx]<<" ";
     // }

      //ufile1.close();
      //ufile.close();

      return 0;  
}
