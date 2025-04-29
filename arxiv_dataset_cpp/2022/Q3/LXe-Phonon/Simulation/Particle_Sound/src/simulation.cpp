#include <simulation.h>
#include <iostream>
#include <iomanip>
#include <ctime>

/////////////////////////////////////////////////////
// Simulation Class /////////////////////////////////
/////////////////////////////////////////////////////


/////////////////////////////////////////////////////
// Constructors /////////////////////////////////////

// Default constructor
Simulation::Simulation(){
    dt = 0;
    dx = 0;
    dz = 0;

    Nx = 0;
    Nz = 0;

    R_max = 0;
    Z_max = 0;

    P_prev = nullptr;
    P_curr = nullptr;
    P_next = nullptr;

    A = nullptr;
    B = nullptr;
    D = nullptr;
    lhs = nullptr;

    fluid = nullptr;
}


Simulation::Simulation(double dt, double dx, double dz, double R_max, double Z_max, Fluid* fluid, double (*E)(double,double)){
    this->dt = dt;
    this->dx = dx;
    this->dz = dz;

    this->R_max = R_max;
    this->Z_max = Z_max;

    this->Nx = R_max/dx;
    this->Nz = Z_max/dz;

    this->fluid = fluid;

    this->E = E;

    // Setup the matrices
    setup();

    // Create the solving tridiagonla matrix
    create_tridiag(A,B,D);
}

Simulation::Simulation(double dt, double dx, double R_max, double Z_max, Fluid* fluid, double (*E)(double,double)){
    this->dt = dt;
    this->dx = dx;
    this->dz = dx;

    this->R_max = R_max;
    this->Z_max = Z_max;

    this->Nx = R_max/dx;
    this->Nz = Z_max/dz;

    this->fluid = fluid;

    this->E = E;

    // Setup the matrices
    setup();

    // Create the solving tridiagonla matrix
    create_tridiag(A,B,D);
}

Simulation::Simulation(double dt, double dx, double dz, int Nx, int Nz, Fluid* fluid, double (*E)(double,double)){
    this->dt = dt;
    this->dx = dx;
    this->dz = dz;

    this->R_max = dx*Nx;
    this->Z_max = dz*Nz;

    this->Nx = Nx;
    this->Nz = Nz;

    this->fluid = fluid;

    this->E = E;

    // Setup the matrices
    setup();

    // Create the solving tridiagonla matrix
    create_tridiag(A,B,D);
}

Simulation::Simulation(double dt, double dx, int Nx, int Nz, Fluid* fluid, double (*E)(double,double)){
    this->dt = dt;
    this->dx = dx;
    this->dz = dx;

    this->R_max = dx*Nx;
    this->Z_max = dz*Nz;

    this->Nx = Nx;
    this->Nz = Nz;

    this->fluid = fluid;

    this->E = E;

    // Setup the matrices
    setup();

    // Create the solving tridiagonal matrix
    create_tridiag(A,B,D);
}

/////////////////////////////////////////////////////
// Class Functions //////////////////////////////////

// The function that performs a single simulation step
void Simulation::step(){
    // First obrain lhs
    // Timing
    clock_t c_start = std::clock();
    std::cout << "\n\t Creating LHS  ...  ";
    create_lhs(lhs);
    std::cout << "\t Completed after: " << std::fixed << std::setprecision(2) << 1000. * (std::clock() - c_start) / CLOCKS_PER_SEC << " ms" << std::endl; 

    // Then get the next pressure vector
    block_tridiag(A,Nx-1,B,Nx-1,D,Nx,lhs,Nx, P_next);

    // Do whatever you want with the pressure Vectors


    // Swap them (i.e. P_prev = P_curr, P_curr = P_next)
    P_prev = P_curr;
    P_curr = P_next;
}

// Runs the simulation for a specific number of iterations N
void Simulation::run(int N){
    clock_t sim_start = std::clock();

    for (int t=0; t<N; t++){
        clock_t c_start = std::clock();
        std::cout << "Step: " << t << " started  ...  ";
        step();
        std::cout << "Completed after: " << std::fixed << std::setprecision(2) << 1000. * (std::clock() - c_start) / CLOCKS_PER_SEC << " ms\n" << std::endl; 
    }

    std::cout << "Simulation completed after: " << std::fixed << std::setprecision(2) << 1000. * (std::clock() - sim_start) / CLOCKS_PER_SEC << " ms" << std::endl; 
}

// Runs the simulation for a specific amount of time in seconds
void Simulation::run(double T){
    int N = T/dt;
    run(N);
}

// Set's up the simulatio, after this is called, the sim is ready to run
// Called in constructor by default.
void Simulation::setup(){
    // Create the appropriate matrix arrays and initialize them to zero

    // For the tridiagonal matrix
    // Center Diagonal
    D = new Matrix[Nx];
    for (int n=0; n<Nx; n++) D[n].init(Nz,Nz);

    // Upper Diagonal
    A = new Matrix[Nx-1];
    for (int n=0; n<Nx-1; n++) A[n].init(Nz,Nz);

    // Lower Diagonal
    B = new Matrix[Nx-1];
    for (int n=0; n<Nx-1; n++) B[n].init(Nz,Nz);

    // Now we initialize the left hand side array and the pressure arrays
    lhs = new Array[Nx];
    P_prev = new Array[Nx];
    P_curr = new Array[Nx];
    P_next = new Array[Nx];

    for (int i=0; i < Nx; i++) {
        lhs[i].init(Nz);
        P_prev[i].init(Nz);
        P_curr[i].init(Nz);
        P_next[i].init(Nz);
    }
}



// Takes as input a linear vector Nx x Nz and decomposes it to array vectors for tridiag solving.
void Simulation::convert_to_Array(Array* from, Array* to){
    for (int i=0; i<Nx; i++){
        for(int j=0; j<Nz; j++){
            to[i][j] = (*from)[i*Nx+j];
        }
    }
}


// Set intial conditions for P by passing P_prev and P_curr
void Simulation::set_initial(Array& P_prev,Array& P_curr){
    convert_to_Array(&P_prev,this->P_prev);
    convert_to_Array(&P_curr,this->P_curr);
}

// Creates the tridiagonal matrix to be solved in each iteration.
// Assumes that the arrays A, B, D, and b exist and are initialised
// the partition happens every Nz cells for Nx repetitions
void Simulation::create_tridiag(Matrix* A, Matrix* B, Matrix* D){
    
    // For all the submatrices
    for(int n=0; n<Nx; n++){

        // Initialize the D matrix
        for (int i=0; i<Nz; i++){
            D[n][i][i]   = -4*fluid->mu()/(fluid->K()*dx*dx*dt) + fluid->r0()/(fluid->K()*dt*dt);
            if (i+1 < Nz) D[n][i][i+1] = fluid->mu()/(fluid->K()*dx*dx*dt);
            if (i-1 > 0 ) D[n][i][i-1] = fluid->mu()/(fluid->K()*dx*dx*dt);
        }

        if (n<Nx-1){
            // Initialize the A and B matrices
            for (int i=0; i<Nz-1; i++){
                double r = n*dx;
                A[n][i][i]  = fluid->mu()/(fluid->K()*dx*dx*dt) + fluid->mu()/(2*r*fluid->K()*dx*dt);
                
                r = (n+1)*dx;
                B[n][i][i]  = fluid->mu()/(fluid->K()*dx*dx*dt) - fluid->mu()/(2*r*fluid->K()*dx*dt);
            }
        }
    }

}


// Creates an array for the left hand side to be used for tridiagonal solving
// Assumes that the array b exists and has already the correct dimensions
void Simulation::create_lhs(Array* b){
    for(int n=0; n<Nx; n++){
        for(int i=0; i<Nz; i++){
            double r = n*dx;
            // For the cells that are not in the boundaries
            if ((n+1 < Nx && n-1 >= 0) && (i+1<Nz && i-1>=0)){
                b[n][i] = 
                (1 + fluid->mu()/(fluid->K() * dt))/(dx*dx)*(
                    P_curr[n][i+1] - 2*P_curr[n][i] + P_curr[n][i-1] + 
                    P_curr[n+1][i] - 2*P_curr[n][i] + P_curr[n-1][i]
                ) + 1/(2*r*dx)* (P_curr[n+1][i] - P_curr[n-1][i]) +
                fluid->r0()/(fluid->K()*dt*dt)* (2*P_curr[n][i] - P_prev[n][i]) - 
                fluid->beta()/fluid->Cp() * E(r,i*dx);
            }


            // Now we are going to implement a box because it is easy
            // But it actually may be better to change this in the future
            // Specifically we are implementing fully absorbent boundary conditions
            else if(n+1 >= Nx){
                if (i+1 >= Nz){

                    b[n][i] = 
                    (1 + fluid->mu()/(fluid->K() * dt))/(dx*dx)*(
                        - 2*P_curr[n][i] + P_curr[n][i-1] + 
                        - 2*P_curr[n][i] + P_curr[n-1][i]
                    ) + 1/(2*r*dx)* (- P_curr[n-1][i]) +
                    fluid->r0()/(fluid->K()*dt*dt)* (2*P_curr[n][i] - P_prev[n][i]) - 
                    fluid->beta()/fluid->Cp() * E(r,i*dx);

                }

                else if (i-1<0){

                    b[n][i] = 
                    (1 + fluid->mu()/(fluid->K() * dt))/(dx*dx)*(
                        P_curr[n][i+1] - 2*P_curr[n][i] + 
                        - 2*P_curr[n][i] + P_curr[n-1][i]
                    ) + 1/(2*r*dx)* (- P_curr[n-1][i]) +
                    fluid->r0()/(fluid->K()*dt*dt)* (2*P_curr[n][i] - P_prev[n][i]) - 
                    fluid->beta()/fluid->Cp() * E(r,i*dx);

                }
            }

            else if(n-1 < 0){
                if(i+1 >= Nz){
                    b[n][i] = 
                    (1 + fluid->mu()/(fluid->K() * dt))/(dx*dx)*(
                        - 2*P_curr[n][i] + P_curr[n][i-1] + 
                        P_curr[n+1][i] - 2*P_curr[n][i]
                    ) + 1/(2*r*dx)* (P_curr[n+1][i]) +
                    fluid->r0()/(fluid->K()*dt*dt)* (2*P_curr[n][i] - P_prev[n][i]) - 
                    fluid->beta()/fluid->Cp() * E(r,i*dx);
                }

                else if(i-1 < 0){
                    b[n][i] = 
                    (1 + fluid->mu()/(fluid->K() * dt))/(dx*dx)*(
                        P_curr[n][i+1] - 2*P_curr[n][i] + 
                        P_curr[n+1][i] - 2*P_curr[n][i]
                    ) + 1/(2*r*dx)* (P_curr[n+1][i]) +
                    fluid->r0()/(fluid->K()*dt*dt)* (2*P_curr[n][i] - P_prev[n][i]) - 
                    fluid->beta()/fluid->Cp() * E(r,i*dx);
                }
            }
        }
    }
}



/////////////////////////////////////////////////////
// Mutators /////////////////////////////////////////

// Getters
double Simulation::get_dt(){ return dt;}
double Simulation::get_dx(){ return dx;}
double Simulation::get_dz(){ return dz;}

int Simulation::get_Nx(){ return Nx;}
int Simulation::get_Nz(){ return Nz;}

double Simulation::get_R_max(){ return R_max;}
double Simulation::get_Z_max(){ return Z_max;}

Array* Simulation::get_P_prev(){ return P_prev;}
Array* Simulation::get_P_curr(){ return P_curr;}
Array* Simulation::get_P_next(){ return P_next;}

Fluid* Simulation::get_fluid(){ return fluid;}


// Setters
void Simulation::set_dt(double dt){ this->dt = dt;}
void Simulation::set_dx(double dx){ this->dx = dx;}
void Simulation::set_dz(double dz){ this->dz = dz;}

void Simulation::set_Nx(int Nx){ this->Nx = Nx;}
void Simulation::set_Nz(int Nz){ this->Nz = Nz;}

void Simulation::set_R_max(double R_max){ this->R_max = R_max;}
void Simulation::set_Z_max(double Z_maz){ this->Z_max = Z_max;}

void Simulation::set_fluid(Fluid* fluid){ this->fluid = fluid;}



/////////////////////////////////////////////////////
// Printing Functions ///////////////////////////////

// Prints everything into a stream object
void Simulation::print(std::ostream& stream){
    stream << "Particle Through Fluid Simulation object \n" << std::endl;
    
    stream << "\t dt:\t" << get_dt() << std::endl;
    stream << "\t dx:\t" << get_dx() << std::endl;
    stream << "\t dz:\t" << get_dz() << std::endl;

    stream << "\t Nx:\t" << get_Nx() << std::endl;
    stream << "\t Nz:\t" << get_Nz() << std::endl;
    
    stream << "\t R_max:\t" << get_R_max() << std::endl;
    stream << "\t Z_max:\t" << get_Z_max() << std::endl;

    stream << "\t fluid:\n" << std::endl;
    get_fluid()->print(stream);
}

void Simulation::print(){
    print(std::cout);
}