
// This CPP implementation is manually translated from shallow_water_equation_python.ipynb
// Original code author of shallow_water_equation_python.ipynb is Dr. Kezhou (Melody) Lu from Georgia Tech


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <cfloat>
#include "dcl.h"


typedef vector<double> VEC;
typedef vector<VEC> VEC2;
typedef vector<VEC2> VEC3;


int glob_EB[5] = {5, 3, 3, 3, 3};
int glob_MB[5] = {10, 12, 12, 12, 12};

void lax_wendroff(double dx, double dy, double dt, 
            int ny, int nx, 
            double up[180][167], double vp[180][167], double hp[180][167], 
            double g, double f[180],
            double u_new[178][165], double v_new[178][165], double h_new[178][165])
{
    VEC2 q1(180, VEC(167, 0.0));
    VEC2 q2(180, VEC(167, 0.0));
    VEC2 q3(180, VEC(167, 0.0));
    VEC2 Ux(180, VEC(167, 0.0));
    VEC2 Uy(180, VEC(167, 0.0));
    VEC2 Vx(180, VEC(167, 0.0));
    VEC2 Vy(180, VEC(167, 0.0));
    VEC2 Qu(180, VEC(167, 0.0));
    VEC2 Qv(180, VEC(167, 0.0));

    for(int i = 0; i < ny; i++) {
        for(int j = 0; j < nx; j++) {
            q1[i][j] = up[i][j] * hp[i][j];
            q2[i][j] = vp[i][j] * hp[i][j];
            q3[i][j] = hp[i][j];

            Ux[i][j] = up[i][j] * q1[i][j] + g * hp[i][j] * hp[i][j] * 0.5;
            Uy[i][j] = q1[i][j] * vp[i][j];

            Vx[i][j] = Uy[i][j];
            Vy[i][j] = vp[i][j] * q2[i][j] + g * hp[i][j] * hp[i][j] * 0.5;

            Qu[i][j] = f[i] * vp[i][j];
            Qv[i][j] = -f[i] * up[i][j];
        }
    }

    // Get the mid-point values in time and space
    // X-momentum equation
    // q1_mx = 0.5*(q1[:,0:-1]+q1[:,1:]) - (0.5*dt/dx)*(Ux[:,1:]-Ux[:,0:-1])
    double q1_mx[180][166];
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx - 1; ++j) {
            q1_mx[i][j] = 0.5 * (q1[i][j] + q1[i][j + 1]) - (0.5 * dt / dx) * (Ux[i][j + 1] - Ux[i][j]);
        }
    }

    // q1_my = 0.5*(q1[0:-1,:]+q1[1:,:]) - (0.5*dt/dy)*(Uy[1:,:]-Uy[0:-1,:])
    double q1_my[179][167];
    for (int i = 0; i < ny - 1; ++i) {
        for (int j = 0; j < nx; ++j) {
            q1_my[i][j] = 0.5 * (q1[i][j] + q1[i + 1][j]) - (0.5 * dt / dy) * (Uy[i + 1][j] - Uy[i][j]);
        }
    }

    // Y-momentum equation
    // q2_mx = 0.5*(q2[:,0:-1]+q2[:,1:]) - (0.5*dt/dx)*(Vx[:,1:]-Vx[:,0:-1])
    double q2_mx[180][166];
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx - 1; ++j) {
            q2_mx[i][j] = 0.5 * (q2[i][j] + q2[i][j + 1]) - (0.5 * dt / dx) * (Vx[i][j + 1] - Vx[i][j]);
        }
    }

    // q2_my = 0.5*(q2[0:-1,:]+q2[1:,:]) - (0.5*dt/dy)*(Vy[1:,:]-Vy[0:-1,:])
    double q2_my[179][167];
    for (int i = 0; i < ny - 1; ++i) {
        for (int j = 0; j < nx; ++j) {
            q2_my[i][j] = 0.5 * (q2[i][j] + q2[i + 1][j]) - (0.5 * dt / dy) * (Vy[i + 1][j] - Vy[i][j]);
        }
    }

    // Mass-conservation equation
    // q3_mx = 0.5*(q3[:,0:-1]+q3[:,1:]) - (0.5*dt/dx)*(q1[:,1:]-q1[:,0:-1])
    double q3_mx[180][166];
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx - 1; ++j) {
            q3_mx[i][j] = 0.5 * (q3[i][j] + q3[i][j + 1]) - (0.5 * dt / dx) * (q1[i][j + 1] - q1[i][j]);
        }
    }

    // q3_my = 0.5*(q3[0:-1,:]+q3[1:,:]) - (0.5*dt/dy)*(q2[1:,:]-q2[0:-1,:])
    double q3_my[179][167];
    for (int i = 0; i < ny - 1; ++i) {
        for (int j = 0; j < nx; ++j) {
            q3_my[i][j] = 0.5 * (q3[i][j] + q3[i + 1][j]) - (0.5 * dt / dy) * (q2[i + 1][j] - q2[i][j]);
        }
    }


    // Update the mid-point values of Ux, Uy, Vy, and Vx
    // Ux_mx = q1_mx*q1_mx/q3_mx + 0.5*g*q3_mx**2
    // Vx_mx = q1_mx*q2_mx/q3_mx
    double Ux_mx[180][166];
    double Vx_mx[180][166];
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx - 1; ++j) {
            
            // Ux_mx[i][j] = q1_mx[i][j] * q1_mx[i][j] / q3_mx[i][j]
            //             + 0.5 * g * q3_mx[i][j] * q3_mx[i][j];


            // Vx_mx[i][j] = q1_mx[i][j] * q2_mx[i][j] / q3_mx[i][j];

            double t1 = q1_mx[i][j];
            double t2 = q3_mx[i][j];
            double t3 = t1 * t1;
            double t4 = t3 / t2;
            double t5 = t2 * t2;
            double t6 = t4 + 0.5 * g * t5;
            Ux_mx[i][j] = t6;

            double t7 = q2_mx[i][j];

            double t8;
            // this is where flexible multiplication is used
            // t8 = flexible_float_multiply_retry(t1, t7, false);
            t8 = t1 * t7;
            double t9 = t8 / t2;
            Vx_mx[i][j] = t9;

            
        }
    }

    // Uy_my = q1_my*q2_my/q3_my
    // Vy_my = q2_my*q2_my/q3_my + 0.5*g*q3_my**2
    double Uy_my[179][167];
    double Vy_my[179][167];
    for (int i = 0; i < ny - 1; ++i) {
        for (int j = 0; j < nx; ++j) {
            Uy_my[i][j] = q1_my[i][j] * q2_my[i][j] / q3_my[i][j];
            Vy_my[i][j] = q2_my[i][j] * q2_my[i][j] / q3_my[i][j]
                        + 0.5 * g * q3_my[i][j] * q3_my[i][j];
        }
    }
    
    // Apply the midpoint value to predict the value at next time step
    
    // Mass-conservation equation
    // q3_new = hp[1:-1,1:-1] - (dt/dx)*(q1_mx[1:-1,1:]-q1_mx[1:-1,0:-1])\
    //                        - (dt/dy)*(q2_my[1:,1:-1]-q2_my[0:-1,1:-1])
    // X-momentum equation
    // q1_new = q1[1:-1,1:-1] - (dt/dx)*(Ux_mx[1:-1,1:]-Ux_mx[1:-1,0:-1])\
    //                        - (dt/dy)*(Uy_my[1:,1:-1]-Uy_my[0:-1,1:-1])\
    //                        + dt*Qu[1:-1,1:-1]*0.5*(hp[1:-1,1:-1]+q3_new)

    double q1_new[178][165];
    double q2_new[178][165];
    double q3_new[178][165];
    for (int i = 1; i < ny - 1; ++i) {
        for (int j = 1; j < nx - 1; ++j) {
            q3_new[i - 1][j - 1] = hp[i][j] 
                                 - (dt / dx) * (q1_mx[i][j] - q1_mx[i][j - 1])
                                 - (dt / dy) * (q2_my[i][j] - q2_my[i - 1][j]);

            q1_new[i - 1][j - 1] = q1[i][j]
                                 - (dt / dx) * (Ux_mx[i][j] - Ux_mx[i][j - 1])
                                 - (dt / dy) * (Uy_my[i][j] - Uy_my[i - 1][j])
                                 + dt * Qu[i][j] * 0.5 * (hp[i][j] + q3_new[i - 1][j - 1]);

            q2_new[i - 1][j - 1] = q2[i][j]
                                - (dt / dx) * (Vx_mx[i][j] - Vx_mx[i][j - 1])
                                - (dt / dy) * (Vy_my[i][j] - Vy_my[i - 1][j])
                                + dt * Qv[i][j] * 0.5 * (hp[i][j] + q3_new[i - 1][j - 1]);
        }
    }
    
    for (int i = 0; i < ny - 2; ++i) {
        for (int j = 0; j < nx - 2; ++j) {
            u_new[i][j] = q1_new[i][j] / q3_new[i][j];
            v_new[i][j] = q2_new[i][j] / q3_new[i][j];
            h_new[i][j] = q3_new[i][j];
        }
    }
}



int main()
{
    
    // Define the variables and the empty matrix to store the value
    double star_lon = 118;
    double cut_off_lon = 285;
    double omega = 7.2921 * 0.00001; // Earth angular velocity
    double g = 9.81; // Gravitational acceleration

    int ny = 180;
    int nx = 167;

    double lat[180]; // latitude
    double lon[167]; // longtitude
    double y[180];
    double x[167];
    double f[180];

    for(int i = 0; i < ny; i++) {
        lat[i] = -90 + 1.0 * i * ny / (ny - 1);
        y[i] = lat[i] * 111111;
        f[i] = y[i] * 5 * 0.0000000001;
    }
    for(int i = 0; i < nx; i++) {
        lon[i] = 118 + 1.0 * i * nx / (nx - 1);
        x[i] = lon[i] * 111320.0; // 111320.0 * star_lon + i * 111320.0 * (cut_off_lon - star_lon) / (nx - 1);
    
    }

    double dy = abs(y[1] - y[0]); // unit:m
    double dx = abs(x[1] - x[0]); // unit:m
    double dt = 60; // unit:s
    double nt = (60 * 12) + 1; // Run for 12 hours


    int isave = 30; // Save data every half hr
    int nsave = int(nt / isave) + 1; // The first point is the initial condition

    VEC3 h_eq(25, VEC2(180, VEC(167)));
    VEC3 u_eq(25, VEC2(180, VEC(167)));
    VEC3 v_eq(25, VEC2(180, VEC(167)));


    double x_cen = star_lon*111320 + (180-118)*111320;
    double y_cen = 0;

    double std_x = 111320 * 22 * 0.25;
    double std_y = 111111 * 6 * 0.25;

    double X[180][167];
    double Y[180][167];

    for(int i = 0; i < ny; i++) {
        for(int j = 0; j < nx; j++) {
            X[i][j] = x[j];
            Y[i][j] = y[i];
        }
    }
    
    double height[180][167];
    double ee[180][167];
    double h_min = 9999;
    for(int i = 0; i < ny; i++) {
        for(int j = 0; j < nx; j++) {
            ee[i][j] = (pow(X[i][j]-x_cen, 2) + pow(Y[i][j]-y_cen, 2))/(2*std_y*std_x);
            height[i][j] = 9750 - 1000 * exp(-ee[i][j]);
        }
    }
    
    double h_ub[167];
    double h_lb[167];
    double hp[180][167];
    double up[180][167];
    double vp[180][167];

    for(int i = 0; i < ny; i++) {
        for(int j = 0; j < nx; j++) {
            h_eq[0][i][j] = height[i][j];
            hp[i][j] = height[i][j];
        }
    }

    for(int j = 0; j < nx; j++) {
        h_ub[j] = height[ny-1][j];
        h_lb[j] = height[0][j];
    }

    double u_new[178][165];
    double v_new[178][165];
    double h_new[178][165];

    int count = 1;
    for(int n = 1; n <= nt; n++) {
        
        lax_wendroff(dx, dy, dt, ny, nx, up, vp, hp, g, f, 
                    u_new, v_new, h_new);

        // Boundary Condition
        double uc[180][167];
        double vc[180][167];
        double hc[180][167];

        for (int i = 1; i < ny - 1; ++i) {
            for (int j = 1; j < nx - 1; ++j) {
                uc[i][j] = u_new[i-1][j-1];
                vc[i][j] = v_new[i-1][j-1];
                hc[i][j] = h_new[i-1][j-1];
            }
        }

        for(int j = 0; j < nx; j++) {
            uc[ny-1][j] = uc[ny-2][j];
            uc[0][j] = uc[1][j];

            vc[ny-1][j] = 0;
            vc[0][j] = 0;

            hc[ny-1][j] = h_ub[j];
            hc[0][j] = h_lb[j];
        }

        for(int i = 0; i < ny; i++) {
            uc[i][nx-1] = uc[i][nx-2];
            uc[i][0] = uc[i][1];
                
            vc[i][nx-1] = vc[i][nx-2];
            vc[i][0] = vc[i][1];

            hc[i][nx-1] = hc[i][nx-2];
            hc[i][0] = hc[i][1];

        }

        for(int i = 0; i < ny; i++) {
            for(int j = 0; j < nx; j++) {
                up[i][j] = uc[i][j];
                vp[i][j] = vc[i][j];
                hp[i][j] = hc[i][j];
            }
        }

        if( n % isave == 0) {
            for(int i = 0; i < ny; i++) {
                for(int j = 0; j < nx; j++) {
                    u_eq[count][i][j] = uc[i][j];
                    v_eq[count][i][j] = vc[i][j];
                    h_eq[count][i][j] = hc[i][j];
                }
            }
            count++;
        }
    }

    // write zonal velocity to binary for visualization
    ofstream outfile("zonal_velocity.bin", ios::binary);
    for(int cnt = 0; cnt < count; cnt++) {
        for (int i = 0; i < ny; ++i) {
            for (int j = 0; j < nx; ++j) {
                outfile.write(reinterpret_cast<char*>(&u_eq[cnt][i][j]), sizeof(double));
            }
        }
    }
    outfile.close();

    return 0;
}