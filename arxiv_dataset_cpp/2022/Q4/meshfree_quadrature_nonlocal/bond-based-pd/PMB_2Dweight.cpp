#include<iostream>
#include<vector>
#include<cmath>
#include<cstdlib>
#include"vvector.h"

extern "C" void dgetrf_(int*, int*, double*, int*, int*, int*);
extern "C" void dgetri_(int* , double* , int* , int* , double* , int* , int* );
extern "C" void dgetrs_(char*, int*, int*, double*, int*, int*, double*, int*, int*);
extern "C" void dgesvd_(char*, char*, int*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int*, int*);


using namespace std;
const double pi = 3.141592653;
const double nu = 0.25;

const double c1=nu/((1+nu)*(1-2*nu));
const double c2=1.0/(2.0*(1+nu));


//basis funtions
double phi(int m,vvector<double> X){
    double x=X[0];
    double y=X[1];
    if(m == 0){
        return 1;
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
    if (m == 3){
        return pi*delta;
    }
    else if (m == 5){
        return pi*delta;
    }
    else if (m == 10){
        return pi/4.0*pow(delta,3);
    }
    else if (m == 12){
        return pi/12.0*pow(delta,3);
    }
    else if (m == 14){
        return pi/4.0*pow(delta,3);
    }
    else {
        return 0.0;
    }
}


double u_exact(double x, double y){
    //return sin(x)*cos(y);
    return sin(x)*sin(y);
}
double v_exact(double x, double y){
    //return cos(x)*sin(y);
    return -cos(x)*cos(y);
}

double E(double x, double y){
    //return 1;
    return 2+sin(x)*sin(y);
}

double fx_exact(double x, double y){
    //return -6*c1*sin(x)*cos(y);
    return ((c1+c2)*(-4*sin(x)*sin(y)+2*cos(2*x)*sin(y)*sin(y))+c2*(-4*sin(x)*sin(y)+2*cos(2*y)*sin(x)*sin(x)));
}
double fy_exact(double x, double y){
    //return -6*c1*cos(x)*sin(y);
    return ((c1+2*c2)*(4*cos(x)*cos(y)+sin(2*x)*sin(2*y)));
}

vector<vector<double>> Preprocess(double* x, int N, double dhratio, vector<vector<int>> nei, int basedim){
    //solve for quadrature weights
    double* WORK = new double[1000];
    vector< vector<double> > weight((N+1)*(N+1)); //quadrature weights
    int k = floor(dhratio);
    double h=1./N;
    char C='T';
    char D='A';
    vvector<double> xi(2);
    double eta; 
    int s; 
    for(int i=0;i<N+1;i++){
        for(int j=0;j<N+1;j++){
            s = nei[i*(N+1)+j].size();
            weight[i*(N+1)+j].resize(s,0);
            double* stiffweight = new double[(s+basedim)*(s+basedim)];
            double* rhsweight = new double[s+basedim];
            double* U = new double[(s+basedim)*(s+basedim)];
            double* VT = new double[(s+basedim)*(s+basedim)];
            double* S = new double[s+basedim];
            int INFO;
            int LWORK = 1000;
            int dim1 = s+basedim;
            for(int t=0;t<(s+basedim)*(s+basedim);t++){
                stiffweight[t] = 0;
            }
            for(int t=0;t<s+basedim;t++){
                rhsweight[t] = 0;
            }
            for(int t=0;t<s;t++){
                xi[0] = x[nei[i*(N+1)+j][t]] - x[(i+k)*(N+1+2*k)+j+k]; 
                xi[1] = x[nei[i*(N+1)+j][t]+(N+1+2*k)*(N+1+2*k)] - x[(i+k)*(N+1+2*k)+j+k+(N+1+2*k)*(N+1+2*k)];
                //stiffweight[t*(s+basedim)+t] = 2;
                stiffweight[t*(s+basedim)+t] = 2.0/xi.Norm2();
                for(int m=0;m<basedim;m++){
                    eta = phi(m,xi);
                    stiffweight[t*(s+basedim)+s+m] = 3.0/pi/pow(dhratio*h,3)*eta/pow(xi.Norm2(),3);
                    stiffweight[(s+m)*(s+basedim)+t] = stiffweight[t*(s+basedim)+s+m];
                }
                
            }

            for(int m=0;m<basedim;m++){
                rhsweight[s+m] = 3.0/pi/pow(dhratio*h,3)*Iphi(m, dhratio*h);
            }
            
            dgesvd_(&D,&D,&dim1,&dim1,stiffweight,&dim1,S,U,&dim1,VT,&dim1,WORK,&LWORK,&INFO);
            for(int m=0;m<dim1;m++){
                if(S[m] < 1e-6) S[m] = 0;
            }
            for(int m=0;m<dim1;m++){
                if(S[m] != 0) S[m] = 1./S[m];
            }
            
            
            for(int t=0;t<s;t++){
                for(int l=0;l<dim1;l++){
                    for(int m=0;m<dim1;m++){
                            weight[i*(N+1)+j][t] += VT[t*dim1+l]*S[l]*U[l*dim1+m]*rhsweight[m];
                    }
                }
            }

            delete[] stiffweight;
            delete[] rhsweight;
            delete[] U;
            delete[] S;
            delete[] VT;
        }
    }
    delete[] WORK;
    return weight; 
}


int main(int argc, char* argv[]){

    int N = atoi(argv[1]);                        
    double dhratio = atof(argv[2]);
    double h = 1./N;            // lattice constant, all the nodes are evenly distributed 
    int k = floor(dhratio);
    int dim = 2*(N+1+2*k)*(N+1+2*k);
    int porder = atoi(argv[3]);
    int basedim; //linear basis funtions
    if(porder==2){
      basedim=15; //since C(x,y)=(x-y)(x-y)/|x-y|^2 in pd eqn already has 2nd order poly in numerator, resulting in 2+2=4th order in the basis selection
    }
    else if (porder==3){
      basedim=21; //similar as above, resulting in 5th order in basis selection
    }
    double* x = new double [dim];    //initial configuration 
    double* u = new double [dim];   //displacement in x direction and y direction 
    double* rhs = new double [dim];
    double* stiffK = new double [dim*dim];
    vector< vector<int> > nei((N+1)*(N+1)); //neighborlist
    vector< vector<double> > weight((N+1)*(N+1)); //quadrature weights
    vvector<double> xi(2);
    double eta, mu_i, mu_j, mu_ij;

    double diff = 100;
    double error = 0;
    double load = 100;
    double Bulk = 4e4; 

    double c = 72*Bulk/5/pi/pow(dhratio*h,3);
    double slope = 5*load/3/Bulk;
    int cenx,ceny,cenx1,ceny1;
    double dpi = 0;
    int nstep = 0;
    int s = 0;

    int* IPIV = new int[dim];
    int info;
    int nrhs = 1;
    char C='T';
    char D = 'A';

    //set initial configuration 
    for(int i=0;i<N+1+2*k;i++){
        for(int j=0;j<N+1+2*k;j++){
            x[i*(N+1+2*k)+j] = (j-k)*h;
            x[i*(N+1+2*k)+j+(N+1+2*k)*(N+1+2*k)] = (i-k)*h;
        }
    }

    
    //random perturbation 
    
    double random_coe = atof(argv[4]);
    srand(5);
    for(int i=0;i<N+1+2*k;i++){
        for(int j=0;j<N+1+2*k;j++){
            x[i*(N+1+2*k)+j] += (2*rand()/double(RAND_MAX)-1)*random_coe*h;
            x[i*(N+1+2*k)+j+(N+1+2*k)*(N+1+2*k)] += (2*rand()/double(RAND_MAX)-1)*random_coe*h ;
        }
    }

    //set neighborhood list for every particle 
    for(int i=0;i<N+1;i++){
        for(int j=0;j<N+1;j++){
            for(int s1=-k;s1<k+1;s1++){
                for(int s2=-k;s2<k+1;s2++){
                    if((i+k+s1)*(N+1+2*k)+j+k+s2 < (N+1+2*k)*(N+1+2*k) && (i+k+s1)*(N+1+2*k)+j+k+s2 > -1){
                        xi[0] = x[(i+s1+k)*(N+1+2*k)+j+k+s2] - x[(i+k)*(N+1+2*k)+j+k];
                        xi[1] = x[(i+s1+k+N+1+2*k)*(N+1+2*k)+j+k+s2] - x[(i+k+N+1+2*k)*(N+1+2*k)+j+k];
                        if(xi.Norm2() < dhratio*h && !(s1 == 0 && s2 == 0)){
                            nei[i*(N+1)+j].push_back((i+k+s1)*(N+1+2*k)+j+k+s2);
                        }
                    }
                }
            }       
        }
    }

    weight = Preprocess(x, N, dhratio, nei, basedim);
        
        for(int i=0;i<dim;i++){
            for(int j=0;j<dim;j++){
                stiffK[i*dim+j] = 0;
            }
            rhs[i] = 0; 
        }
    

        for(int i=0;i<N+1;i++){
            for(int j=0;j<N+1;j++){
                cenx = (i+k)*(N+1+2*k)+j+k;
                ceny = (i+k+N+1+2*k)*(N+1+2*k)+j+k;
                for(int s=0;s<nei[i*(N+1)+j].size();s++){

                    xi[0] = x[nei[i*(N+1)+j][s]] - x[cenx];
                    xi[1] = x[nei[i*(N+1)+j][s]+(N+1+2*k)*(N+1+2*k)] - x[ceny];

                    mu_i=0.4*E(x[cenx],x[ceny]);
                    mu_j=0.4*E(x[nei[i*(N+1)+j][s]],x[nei[i*(N+1)+j][s]+(N+1+2*k)*(N+1+2*k)]);
                    mu_ij=2.0/(1.0/mu_i+1.0/mu_j);

                    c=8*mu_ij*3.0/pi/pow(dhratio*h,3);

                    stiffK[cenx*dim+cenx] += -c*xi[0]*xi[0]*weight[i*(N+1)+j][s]/pow(xi.Norm2(),3);
                    stiffK[cenx*dim+nei[i*(N+1)+j][s]] += c*xi[0]*xi[0]*weight[i*(N+1)+j][s]/pow(xi.Norm2(),3);
                    stiffK[cenx*dim+ceny] += -c*xi[0]*xi[1]*weight[i*(N+1)+j][s]/pow(xi.Norm2(),3);
                    stiffK[cenx*dim+nei[i*(N+1)+j][s]+dim/2] += c*xi[0]*xi[1]*weight[i*(N+1)+j][s]/pow(xi.Norm2(),3);

                    stiffK[ceny*dim+ceny] += -c*xi[1]*xi[1]*weight[i*(N+1)+j][s]/pow(xi.Norm2(),3);
                    stiffK[ceny*dim+nei[i*(N+1)+j][s]+dim/2] += c*xi[1]*xi[1]*weight[i*(N+1)+j][s]/pow(xi.Norm2(),3);
                    stiffK[ceny*dim+cenx] += -c*xi[0]*xi[1]*weight[i*(N+1)+j][s]/pow(xi.Norm2(),3);
                    stiffK[ceny*dim+nei[i*(N+1)+j][s]] += c*xi[0]*xi[1]*weight[i*(N+1)+j][s]/pow(xi.Norm2(),3);
                }
                //rhs[cenx] = fx_exact(x[cenx],x[ceny],Bulk);
                //rhs[ceny] = fy_exact(x[cenx],x[ceny],Bulk);//+9*Bulk*pow(dhratio*h,2)/25;
                rhs[cenx] = fx_exact(x[cenx],x[ceny]);
                rhs[ceny] = fy_exact(x[cenx],x[ceny]);//+9*Bulk*pow(dhratio*h,2)/25;
            }
        }

        
         //apply boundary condition
        for(int i=0;i<N+1+2*k;i++){

            if(i < k || i > N+k){
                for(int j=0;j<N+1+2*k;j++){
                    cenx = i*(N+1+2*k)+j;
                    ceny = (i+N+1+2*k)*(N+1+2*k)+j; 
                    rhs[cenx] = u_exact(x[cenx],x[ceny]);
                    rhs[ceny] = v_exact(x[cenx],x[ceny]);
                }
            }
            else{
                for(int j=0;j<k;j++){
                    cenx = i*(N+1+2*k)+j;
                    cenx1 = i*(N+1+2*k)+N+2*k-j;
                    ceny = (i+N+1+2*k)*(N+1+2*k)+j;
                    ceny1 = (i+N+1+2*k)*(N+1+2*k)+N+2*k-j;
                    rhs[cenx] = u_exact(x[cenx],x[ceny]);
                    rhs[cenx1] = u_exact(x[cenx1],x[ceny1]);
                    rhs[ceny] = v_exact(x[cenx],x[ceny]);
                    rhs[ceny1] = v_exact(x[cenx1],x[ceny1]);
                }
            }
        }
       

//        for(int i=0;i<N+1+2*k;i++){
//            for(int j=0;j<N+1+2*k;j++){
//                cout<<rhs[i*(N+1+2*k)+j]<<" ";
//            }
//            cout<<endl;
//        }


        //Modify the stiff matrix, apply Dirichlet BC
        for(int i=0;i<N+1+2*k;i++){
            if(i < k || i > N+k){
                for(int j=0;j<N+1+2*k;j++){
                    cenx = i*(N+1+2*k)+j;
                    ceny = (i+N+1+2*k)*(N+1+2*k)+j;
                    stiffK[cenx*dim+cenx] = 1;
                    stiffK[ceny*dim+ceny] = 1;
                }
            } 
            else{
                for(int j=0;j<k;j++){
                    cenx = i*(N+1+2*k)+j;
                    cenx1 = i*(N+1+2*k)+N+2*k-j;
                    ceny = (i+N+1+2*k)*(N+1+2*k)+j;
                    ceny1 = (i+N+1+2*k)*(N+1+2*k)+N+2*k-j;
                    stiffK[cenx*dim+cenx] = 1;
                    stiffK[cenx1*dim+cenx1] = 1;
                    stiffK[ceny*dim+ceny] = 1;
                    stiffK[ceny1*dim+ceny1] = 1;
                }
            }
        }

  //      for(int i=0;i<dim;i++){
  //          for(int j=0;j<dim;j++){
  //              u[i] += stiffK[i*dim+j]*rhs[j];
  //          }
  //      }

  //      for(int i=0;i<N+1+2*k;i++){
  //          for(int j=0;j<N+1+2*k;j++){
  //              cout<<u[i*(N+1+2*k)+j]<<" ";
  //          }
  //          cout<<endl;
  //      }


  //compute truncation error

  double* ue = new double [dim];
  for(int i=0;i<N+1+2*k;i++){
    for(int j=0;j<N+1+2*k;j++){
      ue[i*(N+1+2*k)+j]=u_exact(x[i*(N+1+2*k)+j],x[i*(N+1+2*k)+j+(N+1+2*k)*(N+1+2*k)]);
      ue[dim/2+i*(N+1+2*k)+j]=v_exact(x[i*(N+1+2*k)+j],x[i*(N+1+2*k)+j+(N+1+2*k)*(N+1+2*k)]);
    }
  }

  double* trun = new double [dim];
  for (int i1=0;i1<dim;i1++){
    trun[i1]=0.0;
    for (int i2=0;i2<dim;i2++){
      trun[i1] += stiffK[i1*dim+i2]*ue[i2];
    }
  }

  //post process, calculate truncation error
    double trun_error = 0.;
    double trun_errorinf = 0.;
      for(int i=0;i<dim/2;i++){
        trun_error += pow(trun[i]-rhs[i],2)+pow(trun[i+dim/2]-rhs[i+dim/2],2);
        //if (fabs(trun[i]-rhs[i])>trun_errorinf){
        //  trun_errorinf = fabs(trun[i]-rhs[i]);
        //}
      }

        dgetrf_(&dim,&dim,stiffK,&dim,IPIV,&info);
        dgetrs_(&C,&dim,&nrhs,stiffK,&dim, IPIV,rhs,&dim,&info);
        
        diff = 0;
       



        for(int i=0;i<dim;i++){
            u[i] = rhs[i];
        }    
        

    for(int i=0;i<N+1+2*k;i++){
        for(int j=0;j<N+1+2*k;j++){
            cenx = i*(N+1+2*k)+j;
            ceny = i*(N+1+2*k)+j+(N+1+2*k)*(N+1+2*k);
            error += pow( u[cenx] - u_exact(x[cenx],x[ceny]),2) + pow( u[ceny] - v_exact(x[cenx],x[ceny]),2); 
        }
    }

    
    error /= (N+1)*(N+1);
    error = sqrt(error);

    cout<<" error l2: "<<error<<endl;
    cout<<" truncation error L2: "<< sqrt(trun_error)/(N+1) <<endl;

    ofstream ufile,ufile1;
    ufile.open("displacement_y.csv");
    ufile1.open("displacement_x.csv");
    ufile<<"x y v"<<endl;
    ufile1<<"x y u"<<endl;
    for(int i=0;i<N+1+2*k;i++){
        for(int j=0;j<N+1+2*k;j++){
            cenx = i*(N+1+2*k)+j;
            ceny = i*(N+1+2*k)+j+(N+1+2*k)*(N+1+2*k);
            ufile1<<x[cenx]<<" "<<x[ceny]<<" "<<u[cenx]<<" "<<endl;
            ufile<<x[cenx]<<" "<<x[ceny]<<" "<<u[ceny]<<" "<<endl;
        }
        ufile1<<endl;
        ufile<<endl;
    }
    
    delete[] x;
    delete[] u;
    delete[] rhs;
    delete[] stiffK;
    delete[] IPIV;
    return 0;  
}
