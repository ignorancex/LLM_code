#include <iostream>
#include <math.h>
#include <time.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include <assert.h>
#include <algorithm>
#define PI 3.14159265359

int main(int argc, char** argv){

        //---------------------------------------------//
        int Lx_, Ly_;
        int N_orb, N_spin;

        Lx_=atoi(argv[1]);
        Ly_=atoi(argv[2]);

        N_orb=2; N_spin=2;

        double A_val, B_val, M_val;
        A_val=atof(argv[3]);
        B_val=atof(argv[4]);

        bool Periodic_Y=true;

        string connection_out = argv[5];
        ofstream connec_out(connection_out.c_str());

        complex<double> Zero_(0,0),One_(1,0),Iota_(0,1);
        //---------------------------------------------//


        //-----------------------------------------------------------//
        //All Connections:->
        //Intra-orbital Hoppings:->
        complex<double> taa_x_up(B_val,0),tbb_x_up(-B_val,0);
        complex<double> taa_x_dn(B_val,0),tbb_x_dn(-B_val,0);
        complex<double> taa_y_up(B_val,0),tbb_y_up(-B_val,0);
        complex<double> taa_y_dn(B_val,0),tbb_y_dn(-B_val,0);

        //Inter-orbital Mixings:->
        complex<double> tab_x_up(0,0.5*A_val),tba_x_up(0,0.5*A_val);
        complex<double> tab_x_dn(0,-0.5*A_val),tba_x_dn(0,-0.5*A_val);
        complex<double> tab_y_up(0.5*A_val,0),tba_y_up(-0.5*A_val,0);
        complex<double> tab_y_dn(0.5*A_val,0),tba_y_dn(-0.5*A_val,0);
        //-----------------------------------------------------------//

        int MS_=N_orb*N_spin*Lx_*Ly_;

        //Complex matrix C_mat[i][j] = C_mat[row][col] (=H[i][j]):--->
        Mat_2_Complex_doub C_mat;
        C_mat.resize(MS_);
        for(int i=0;i<MS_;i++){
                C_mat[i].resize(MS_);
                for(int j=0;j<MS_;j++){
                        C_mat[i][j]=Zero_;
                }
        }

        //--------------------------------------------------//

        int ra_u,rb_u,rax_u,rbx_u,ray_u,rby_u;
        int ra_d,rb_d,rax_d,rbx_d,ray_d,rby_d;

        int orb_a=0,orb_b=1;
        int spin_u=0,spin_d=1;

        int r_pos,r_pos_px,r_pos_py;
        //--------------------------------------------------//
        //
        for(int rx=0;rx<Lx_;rx++){
                for(int ry=0;ry<Ly_;ry++){

                        r_pos    = ry + Ly_*rx;
                        r_pos_px = ry + Ly_*(rx+1);
                        r_pos_py = ry+1 + Ly_*rx;

                        ra_u = orb_a + N_orb*spin_u + N_orb*N_spin*r_pos;
                        rb_u = orb_b + N_orb*spin_u + N_orb*N_spin*r_pos;
                        ra_d = orb_a + N_orb*spin_d + N_orb*N_spin*r_pos;
                        rb_d = orb_b + N_orb*spin_d + N_orb*N_spin*r_pos;

                        rax_u = orb_a + N_orb*spin_u + N_orb*N_spin*r_pos_px;
                        rbx_u = orb_b + N_orb*spin_u + N_orb*N_spin*r_pos_px;
                        rax_d = orb_a + N_orb*spin_d + N_orb*N_spin*r_pos_px;
                        rbx_d = orb_b + N_orb*spin_d + N_orb*N_spin*r_pos_px;

                        ray_u = orb_a + N_orb*spin_u + N_orb*N_spin*r_pos_py;
                        rby_u = orb_b + N_orb*spin_u + N_orb*N_spin*r_pos_py;
                        ray_d = orb_a + N_orb*spin_d + N_orb*N_spin*r_pos_py;
                        rby_d = orb_b + N_orb*spin_d + N_orb*N_spin*r_pos_py;

                        if(rx!=Lx_-1){
                                C_mat[ra_u][rax_u] = 1.0*conj(taa_x_up);
                                C_mat[rb_u][rbx_u] = 1.0*conj(tbb_x_up);
                                C_mat[ra_d][rax_d] = 1.0*conj(taa_x_dn);
                                C_mat[rb_d][rbx_d] = 1.0*conj(tbb_x_dn);

                                C_mat[ra_u][rbx_u] = 1.0*conj(tab_x_up);
                                C_mat[rb_u][rax_u] = 1.0*conj(tba_x_up);
                                C_mat[ra_d][rbx_d] = 1.0*conj(tab_x_dn);
                                C_mat[rb_d][rax_d] = 1.0*conj(tba_x_dn);
                        }

                        if(ry!=Ly_-1){
                                C_mat[ra_u][ray_u] = 1.0*conj(taa_y_up);
                                C_mat[rb_u][rby_u] = 1.0*conj(tbb_y_up);
                                C_mat[ra_d][ray_d] = 1.0*conj(taa_y_dn);
                                C_mat[rb_d][rby_d] = 1.0*conj(tbb_y_dn);

                                C_mat[ra_u][rby_u] = 1.0*conj(tab_y_up);
                                C_mat[rb_u][ray_u] = 1.0*conj(tba_y_up);
                                C_mat[ra_d][rby_d] = 1.0*conj(tab_y_dn);
                                C_mat[rb_d][ray_d] = 1.0*conj(tba_y_dn);
                        }

                        if(Periodic_Y==true){
                                r_pos_py = 0 + Ly_*rx;

                                ray_u = orb_a + N_orb*spin_u + N_orb*N_spin*r_pos_py;
                                rby_u = orb_b + N_orb*spin_u + N_orb*N_spin*r_pos_py;
                                ray_d = orb_a + N_orb*spin_d + N_orb*N_spin*r_pos_py;
                                rby_d = orb_b + N_orb*spin_d + N_orb*N_spin*r_pos_py;

                                if(ry==Ly_-1){
                                        C_mat[ray_u][ra_u] = 1.0*taa_y_up;
                                        C_mat[rby_u][rb_u] = 1.0*tbb_y_up;
                                        C_mat[ray_d][ra_d] = 1.0*taa_y_dn;
                                        C_mat[rby_d][rb_d] = 1.0*tbb_y_dn;

                                        C_mat[rby_u][ra_u] = 1.0*tab_y_up;
                                        C_mat[ray_u][rb_u] = 1.0*tba_y_up;
                                        C_mat[rby_d][ra_d] = 1.0*tab_y_dn;
                                        C_mat[ray_d][rb_d] = 1.0*tba_y_dn;
                                }
                        }
                }
        }

/*
for(int i=0;i<MS_;i++){
    for(int j=0;j<MS_;j++){
                    connec_out<<C_mat[i][j]<<" ";
    }
    connec_out<<endl;
}
*/

for(int i=0;i<MS_;i++){
        connec_out<<"[";
        for(int j=0;j<MS_;j++){
                if(j==MS_-1){
                        connec_out<<C_mat[i][j].real()<<"+i"<<C_mat[i][j].imag();
                }
                else{
                        connec_out<<C_mat[i][j].real()<<"+i"<<C_mat[i][j].imag()<<", ";
                }

        }
        if(i==MS_-1){
                connec_out<<"]"<<endl;
        }
        else{
                connec_out<<"],"<<endl;
        }
}


return 0;
}
