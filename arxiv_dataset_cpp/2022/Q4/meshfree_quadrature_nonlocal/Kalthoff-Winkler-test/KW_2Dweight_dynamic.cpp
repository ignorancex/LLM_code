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


//basis funtions, use 5th-order basis here
double phi(int m,vvector<double> X){
    double x=X[0];
    double y=X[1];
    if(m == 0){
        return 0;
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


vector<int> BoundaryID(double* x, double* y, int N, double dhratio, double h, double centerx, double centery){
    vector<int> ID(N*(2*N-1));
    for(int i=0;i<2*N-1;i++){
        for(int j=0;j<N;j++){
            ID[i*N+j] = 0;
            if(abs(x[i*N+j]-centerx) > 10-dhratio*h) ID[i*N+j] = 1;
            if(abs(y[i*N+j]-centery) > 5-dhratio*h) ID[i*N+j] = 1;
            if(ID[i*N+j] == 1 && abs(x[i*N+j]) <= 2.5+0.725 && y[i*N+j] > -dhratio*h-1e-10) ID[i*N+j] = 2;
        }
    }
    return ID;
}

double Bulk_f(){
    return 191e1;
}

double den_f(){
    return 8e-3;
}

double smax_f(double delta){
    return 0.0099/sqrt(delta);
}

double u_bc(double x, double y, double t, double id){
    double u_temp;
    if (id==0){
        u_temp=0.0;
    }
    else if (id == 1){
        u_temp=0.0;
    }
    else if (id==2){
        u_temp=0.0;
    }
    return u_temp;
}

double v_bc(double x, double y, double t, double id){
    double v_temp;
    if (id==0){
        v_temp=0.0;
    }
    else if (id == 1){
        v_temp=0.0;
    }
    else if (id==2){
        v_temp=-3.2*t;
    }
    return v_temp;
}

double fx(double x, double y){
    return 0.0;
}

double fy(double x, double y){
    return 0.0;
}

vector<vector<double>> Preprocess(double* x, double* y, int N, double dhratio, double h, int basedim, vector<vector<int>> nei, vector<int> ID){
    //solve for quadrature weights
    double* WORK = new double[1000];
    vector< vector<double> > weight(N*(2*N-1)); //quadrature weights
    int k = floor(dhratio);
    char C='T';
    char D='A';
    vvector<double> xi(2);
    double eta; 
    int s; 
    for(int i=0;i<2*N-1;i++){
        for(int j=0;j<N;j++){
            if(ID[i*N+j] == 0){
            s = nei[i*N+j].size();
            weight[i*N+j].resize(s,0);
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
                xi[0] = x[nei[i*N+j][t]] - x[i*N+j]; 
                xi[1] = y[nei[i*N+j][t]] - y[i*N+j];
                stiffweight[t*(s+basedim)+t] = 2.0/xi.Norm2();
                for(int m=0;m<basedim;m++){
                    eta = phi(m,xi);
                    stiffweight[t*(s+basedim)+s+m] = eta/pow(xi.Norm2(),3);
                    stiffweight[(s+m)*(s+basedim)+t] = stiffweight[t*(s+basedim)+s+m];
                }
                
            }
            
            for(int m=0;m<basedim;m++){
                rhsweight[s+m] = Iphi(m, dhratio*h);
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
                            weight[i*N+j][t] += VT[t*dim1+l]*S[l]*U[l*dim1+m]*rhsweight[m];
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
    }
    delete[] WORK;
    return weight; 
}

int main(int argc, char* argv[]){
    
    int N = atoi(argv[1]);                        
    double dhratio = atof(argv[2]);
    
    double Mx = 20.0;
    double My = 10.0;               
    double h = My/(N-1);            // lattice constant, all the nodes are evenly distributed 
    int k = floor(dhratio);
    int dim = 2*N*(2*N-1);
    cout<<"total number of points: "<<dim/2<<endl;
    int porder = atoi(argv[3]);
    int basedim; //linear basis funtions
    if(porder==2){
      basedim=15; //since C(x,y)=(x-y)(x-y)/|x-y|^2 in pd eqn already has 2nd order poly in numerator, resulting in 2+2=4th order in the basis selection
    }
    else if (porder==3){
      basedim=21; //similar as above, resulting in 5th order in basis selection
    }
    double* x = new double [dim/2];    //initial configuration 
    double* y = new double [dim/2];    //initial configuration 
    double* damage = new double [dim/2];     
    double* u = new double [dim];   //displacement in x direction and y direction 
    double* u_old = new double [dim];   //displacement in x direction and y direction 
    double* u_oldold = new double [dim];   //displacement in x direction and y direction 
    double* rhs = new double [dim];
    double* stiffK = new double [dim*dim];
    vector< vector<int> > nei(N*(2*N-1)); //neighborlist
    vector< vector<int> > Bondbroke(N*(2*N-1)); //bondbroke
    vector< vector<double> > weight(N*(2*N-1)); //quadrature weights
    vvector<double> xi(2),zeta(2),gamma(2);
    double eta;

    double diff = 100;
    double error = 0;
    double load = 100;
    //double Bulk = 191e1;
    double Bulk=Bulk_f();
    //double smax = 0.0099/sqrt(dhratio*h);
    double smax=smax_f(dhratio*h);
    //double dt = 2e-4;
    double dt=atof(argv[4]);
    //int Nsteps = 1000;
    int Nsteps = atoi(argv[5]);
    //double den = 8e-3;
    double den=den_f();
    cout << "Bulk" << " " << "smax" << " " << "dt" << " " << "Nsteps" << " " << "den" << endl;
    cout << Bulk << " " << smax << " " << dt << " " << Nsteps << " " << den << endl;
    // constant in 2d PMB model 
    double c = 72*Bulk/5/pi/pow(dhratio*h,3);
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
    cout<<"set initial configuration"<<endl;
    double movex = -10.0;
    double movey = -10.0;
    double centerx = 0.0;
    double centery = -5.0;
    vector<int> ID(N*(2*N-1));
    for(int i=0;i<2*N-1;i++){
        for(int j=0;j<N;j++){
            x[i*N+j] = h*i+movex;
            y[i*N+j] = h*j+movey;
        }
    }

    ID = BoundaryID(x, y, N, dhratio, h, centerx, centery);

    //set neighborhood list for every particle 
    cout<<"build neighborhood list"<<endl;
    for(int i=0;i<2*N-1;i++){
        for(int j=0;j<N;j++){
            if(ID[i*N+j] == 0){
                for(int s1=-k;s1<k+1;s1++){
                    for(int s2=-k;s2<k+1;s2++){
                        xi[0] = x[(i+s1)*N+j+s2] - x[i*N+j];
                        xi[1] =  y[(i+s1)*N+j+s2] - y[i*N+j];
                        if(xi.Norm2() < dhratio*h && !( s1==0 && s2==0)){
                            nei[i*N+j].push_back((i+s1)*(N)+j+s2);
                            Bondbroke[i*N+j].push_back(false);
                        }
                    }
                }
            }       
        }
    }

    cout<<"break bonds that pass through boundary or crack"<<endl;
    for(int i=0;i<2*N-1;i++){
        for(int j=0;j<N;j++){
            if(ID[i*N+j] == 0){
                for(int s=0;s<nei[i*N+j].size();s++){
                    //break the left crack
                    double xc1 = -2.5-0.725;
                    if((x[i*N+j]-xc1)*(x[nei[i*N+j][s]]-xc1) <= 0 ){
                        double slope = (xc1-x[i*N+j])/(x[nei[i*N+j][s]]-x[i*N+j]);
                        double yc = y[i*N+j] + slope*(y[nei[i*N+j][s]]-y[i*N+j]);
                        if(yc >= -5.0) Bondbroke[i*N+j][s] = true;
                    }
                    //right crack    
                    double xc2 = 2.5+0.725;
                    if((x[i*N+j]-xc2)*(x[nei[i*N+j][s]]-xc2) <= 0 ){
                        double slope = (xc2-x[i*N+j])/(x[nei[i*N+j][s]]-x[i*N+j]);
                        double yc = y[i*N+j] + slope*(y[nei[i*N+j][s]]-y[i*N+j]);
                        if(yc >= -5.0) Bondbroke[i*N+j][s] = true;
                    }
                    //break other boundarys as free surface
                    if((y[i*N+j]<-2*dhratio*h) && (ID[nei[i*N+j][s]] == 1)){
                        Bondbroke[i*N+j][s] = true;
                    }
                }
            }
        }
    }

    string folderpath = "./result/";
    string command; 
    command = "mkdir " + folderpath;
    system(command.c_str());
    cout <<"create folder to save results" << endl;

    cout<<"generate quadrature weights"<<endl;
    weight=Preprocess(x, y, N, dhratio, h, basedim, nei, ID);

    ofstream ufile,ufile1,dfile;
            
        for(int step = 1;step <=Nsteps;step++ ){    
            if(step == 1){
                //print out initial damage file 
                //postprocess damage 
                for(int i=0;i<2*N-1;i++){
                    for(int j=0;j<N;j++){
                        int nneigh = 0;
                        double sum = 0;
                        if(ID[i*N+j] == 0){
                            for(int s=0;s<nei[i*N+j].size();s++){
                                if(Bondbroke[i*N+j][s]) sum++;
                                nneigh++;
                            }
                            damage[i*N+j] = sum/double(nneigh);
                        }
                    }
                }
                dfile.open("./result/initial_damage.csv");
                dfile<<"x y damage"<<endl;
                for(int i=0;i<2*N-1;i++){
                    for(int j=0;j<N;j++){
                        if(ID[i*N+j] == 0){
                            dfile<<x[i*N+j]<<" "<<y[i*N+j]<<" "<<damage[i*N+j]<<endl;
                        }
                    }
                }
                dfile.close();
            }    
            double t = dt*step;
           
            
            
            
            for(int i=0;i<dim;i++){
                for(int j=0;j<dim;j++){
                    stiffK[i*dim+j] = 0;
                }
                rhs[i] = 0; 
            }
        
            cout<<"assemble matrix"<<endl; 
            for(int i=0;i<2*N-1;i++){
                for(int j=0;j<N;j++){
                    cenx = i*N+j;
                    ceny = i*N+j+(2*N-1)*N;
                    //cout<<x[cenx]<<" "<<y[cenx]<<endl;
                if(ID[cenx] == 0){
                    //cout<<nei[cenx].size()<<endl;
                    for(int s=0;s<nei[cenx].size();s++){
                        //cout<<weight[cenx][s]<<" ";
                        xi[0] = x[nei[cenx][s]] - x[cenx];
                        xi[1] = y[nei[cenx][s]] - y[cenx];
                        double broke = double(!Bondbroke[i*N+j][s]);
                        stiffK[cenx*dim+cenx] += c*xi[0]*xi[0]*weight[cenx][s]/pow(xi.Norm2(),3)*broke;
                        stiffK[cenx*dim+nei[cenx][s]] += -c*xi[0]*xi[0]*weight[cenx][s]/pow(xi.Norm2(),3)*broke;
                        stiffK[cenx*dim+ceny] += c*xi[0]*xi[1]*weight[cenx][s]/pow(xi.Norm2(),3)*broke;
                        stiffK[cenx*dim+nei[cenx][s]+dim/2] += -c*xi[0]*xi[1]*weight[cenx][s]/pow(xi.Norm2(),3)*broke;

                        stiffK[ceny*dim+ceny] += c*xi[1]*xi[1]*weight[cenx][s]/pow(xi.Norm2(),3)*broke;
                        stiffK[ceny*dim+nei[cenx][s]+dim/2] += -c*xi[1]*xi[1]*weight[cenx][s]/pow(xi.Norm2(),3)*broke;
                        stiffK[ceny*dim+cenx] += c*xi[0]*xi[1]*weight[cenx][s]/pow(xi.Norm2(),3)*broke;
                        stiffK[ceny*dim+nei[cenx][s]] += -c*xi[0]*xi[1]*weight[cenx][s]/pow(xi.Norm2(),3)*broke;
                    }
                    //cout<<endl;
                    rhs[cenx] = fx(x[cenx], y[cenx]) + (2*u_old[cenx]-u_oldold[cenx])/dt/dt*den;
                    rhs[ceny] = fy(x[cenx], y[cenx]) + (2*u_old[ceny]-u_oldold[ceny])/dt/dt*den;
                    stiffK[cenx*dim+cenx] += den/dt/dt;
                    stiffK[ceny*dim+ceny] += den/dt/dt; 
                }
                else{
                    //rhs[cenx] = 0.0;
                    //rhs[ceny] = -double(ID[cenx] == 2)*3.2*t;
                    rhs[cenx] = u_bc(x[cenx], y[ceny], t, ID[cenx]);
                    rhs[ceny] = v_bc(x[cenx], y[ceny], t, ID[cenx]);
                    //cout << "difference x=" << u_bc(x[cenx], y[ceny], t, ID[cenx])-0.0 << "difference y=" << v_bc(x[cenx], y[ceny], t, ID[cenx])+double(ID[cenx] == 2)*3.2*t << endl;
                    stiffK[cenx*dim+cenx] = 1;
                    stiffK[ceny*dim+ceny] = 1; 
                }
                   
            }
        }
        cout<<"solve for displacement"<<endl;
        dgetrf_(&dim,&dim,stiffK,&dim,IPIV,&info);
        dgetrs_(&C,&dim,&nrhs,stiffK,&dim, IPIV,rhs,&dim,&info);
        
        diff = 0;


        for(int i=0;i<dim;i++){
            u[i] = rhs[i];
        }    
        //update broken bonds
        int Nbroke = 0;
        for(int i=0;i<2*N-1;i++){
            for(int j=0;j<N;j++){
                if(ID[i*N+j] == 0){
                    for(int s=0;s<nei[i*N+j].size();s++){
                        xi[0] = x[nei[i*N+j][s]] + u[nei[i*N+j][s]]-x[i*N+j] -u[i*N+j];
                        xi[1] = y[nei[i*N+j][s]] + u[nei[i*N+j][s]+dim/2]-y[i*N+j] - u[i*N+j+dim/2];
                        gamma[0] = x[nei[i*N+j][s]] - x[i*N+j];
                        gamma[1] = y[nei[i*N+j][s]] - y[i*N+j];
                        double stretch = xi.Norm2()/gamma.Norm2()-1.0;
                            if(!Bondbroke[i*N+j][s] && stretch > smax && ID[nei[i*N+j][s]] == 0){
                                Bondbroke[i*N+j][s] = true;
                                Nbroke += 1;
                            }
                    }
                }
            }
        }
    cout<<"bonds broken at step "<<step<<": "<<Nbroke<<endl;
    //update uold and uoldold 
    for(int i=0;i<dim;i++){
        u_oldold[i] = u_old[i];
        u_old[i] = u[i];
    }

    //output displacement and damage field
    if(step % 2 == 0){
    ufile.open("./result/displacement_y"+std::to_string(step)+".csv");
    ufile1.open("./result/displacement_x"+std::to_string(step)+".csv");
    ufile<<"x y v"<<endl;
    ufile1<<"x y u"<<endl;
        for(int i=0;i<2*N-1;i++){
            for(int j=0;j<N;j++){
                if(ID[i*N+j] == 0){
                cenx = i*N+j;
                ceny = i*N+j+(2*N-1)*N;
                ufile1<<x[cenx]<<" "<<y[cenx]<<" "<<u[cenx]<<" "<<endl;
                ufile<<x[cenx]<<" "<<y[cenx]<<" "<<u[ceny]<<" "<<endl;
                }
            }
        //ufile1<<endl;
        //ufile<<endl;
        }
        ufile.close();
        ufile1.close();
        for(int i=0;i<2*N-1;i++){
            for(int j=0;j<N;j++){
                int nneigh = 0;
                double sum = 0;
                if(ID[i*N+j] == 0){
                for(int s=0;s<nei[i*N+j].size();s++){
                    if(Bondbroke[i*N+j][s]) sum++;
                        nneigh++;
                        }
                        damage[i*N+j] = sum/double(nneigh);
                    }
                }
            }
            ofstream dfile;
            dfile.open("./result/damage_"+std::to_string(step)+".csv");
            dfile<<"x y damage"<<endl;
            for(int i=0;i<2*N-1;i++){
                for(int j=0;j<N;j++){
                    if(ID[i*N+j] == 0){
                        dfile<<x[i*N+j]<<" "<<y[i*N+j]<<" "<<damage[i*N+j]<<endl;
                    }
                }
            }
            dfile.close();
    
    }
    }
    delete[] x;
    delete[] u;
    delete[] rhs;
    delete[] stiffK;
    delete[] IPIV;
    return 0;
}
