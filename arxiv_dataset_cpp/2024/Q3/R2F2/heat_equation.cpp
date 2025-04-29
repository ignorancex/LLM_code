#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <assert.h>

#include "dcl.h"

using namespace std;


#define NX 100  // Number of spatial grid points
#define NT 5000 // Number of time steps
#define L 1.0   // Length of the domain
#define T 5.0   // Total simulation time

#define DX (L / (NX - 1))     // Spatial step size
#define DT (T / NT)   // Time step size
#define ALPHA 0.01              // Thermal diffusivity

/// Courant–Friedrichs–Lewy (CFL) condition:
/// (DT * ALPHA) / (DX * DX) <= 1/2

float u_prev[NX]; 
float u_curr[NX];
ofstream fp;


// Function to initialize the temperature profile
void initialize() {
    for (int i = 0; i < NX; i++) {
        float x = i * DX;
        u_prev[i] = 2000*sin(M_PI * x);
    }
}

// Function to solve the heat equation using explicit finite difference method
void solveHeatEquation() {
#pragma HLS INTERFACE s_axilite port=return

    for (int n = 0; n < NT; n++) {

        if(n == 0)
            fp.open("temperature.txt");
        else fp.open("temperature.txt", ios::app);
        
        fp << "Time step " << n << "\n";
        for (int i = 0; i < NX; i++) {
            double x = i * DX;
            fp << std::fixed << std::setprecision(16) << x << " " << u_prev[i] << "\n"; 
        }
        fp.close();

        bool overflow = false;
        float tmp1, tmp2, tmp3, tmp4, tmp5;

        // Apply explicit finite difference scheme
        for (int i = 1; i < NX - 1; i++) {
            
            // original equation:
            // u_curr[i] = u_prev[i] + ALPHA * DT / (DX * DX) * (u_prev[i - 1] - 2 * u_prev[i] + u_prev[i + 1]);
            tmp1 = DX * DX;
            tmp2 = (u_prev[i - 1] - 2 * u_prev[i] + u_prev[i + 1]);
            tmp3 = ALPHA * DT;
            tmp4 = tmp3 / tmp1;

            // this is where the flexible multiplier is used
            // removed the real code for simplicity
            //tmp5 = flexible_float_multiply_retry(tmp4, tmp2, true);
            tmp5 = tmp4 * tmp2;
            u_curr[i] = u_prev[i] + tmp5;
        }
        
        // Approximate du/dx at boundaries using finite differences
        // Neumann boundary condition: du/dx at x = 0
        // u_curr[0] = u_prev[0] + ALPHA * DT / (DX * DX) * (u_prev[1] - u_prev[0]);
        tmp1 = DX * DX;
        tmp2 = (u_prev[1] - u_prev[0]);
        tmp3 = ALPHA * DT;
        tmp4 = tmp3 / tmp1;
        
        // this is where the flexible multiplier is used
        // removed the real code for simplicity
        // tmp5 = flexible_float_multiply_retry(tmp4, tmp2, true);
        tmp5 = tmp4 * tmp2;
        u_curr[0] = u_prev[0] + tmp5;
        
        // Neumann boundary condition: du/dx at x = L
        // u_curr[NX - 1] = u_prev[NX - 1] + ALPHA * DT / (DX * DX) * (u_prev[NX - 2] - u_prev[NX - 1]);
        tmp1 = DX * DX;
        tmp2 = (u_prev[NX - 2] - u_prev[NX - 1]);
        tmp3 = ALPHA * DT;
        tmp4 = tmp3 / tmp1;

        // this is where the flexible multiplier is used
        // removed the real code for simplicity
        // tmp5 = flexible_float_multiply_retry(tmp4, tmp2, true);
        tmp5 = tmp4 * tmp2;
        u_curr[NX - 1] = u_prev[NX - 1] + tmp5;

        // Update the solution for the next time step
        for (int i = 0; i < NX; i++) {
            u_prev[i] = u_curr[i];
        }
    }
}


// Function to print the temperature profile
void printTemperature() {

    // write results to file
    fp.open("temperature.txt", ios::app);
    
    for (int i = 0; i < NX; i++) {
        float x = i * DX;
        printf("%f %f\n", x, u_prev[i]);
        float ff = u_prev[i];
        //fp << std::fixed << std::setprecision(16) << x << " " << ff << "\n";
        fp << x << " " << ff << "\n";
    }

    fp.close();
}


int main() {

    // Check CFL condition:
    // (DT * ALPHA) / (DX * DX) <= 1/2
    printf("(DT * ALPHA) / (DX * DX): %f\n", (DT * ALPHA) / (DX * DX));
    printf("(%f * %f) / (%f * %f)\n", DT, ALPHA, DX, DX);

    // Initialize temperature profile
    initialize();

    // Solve the heat equation
    solveHeatEquation();

    // Print the final temperature profile
    printTemperature();

    return 0;
}
