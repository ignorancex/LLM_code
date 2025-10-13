#include "globes/globes.h"
#include <stdio.h>
#include <complex.h>
#include "Running_and_matching.h"
#include <stdbool.h>

double f(double T3, double Q, const struct SMEFT S_EW) {
    double g_L = 0.6516;  // PDG 2024
    double g_Y = 0.3575;  // PDG 2024
    double g_L2_minus_g_Y2 = g_L * g_L - g_Y * g_Y;

    double term1 = -Q * (g_L * g_Y) / g_L2_minus_g_Y2 * S_EW.c_HWB;
    double term2 = (
        0.25 * S_EW.c_ll[0][1][1][0] - 0.5 * S_EW.c3_phi_l[0][0] - 0.5 * S_EW.c3_phi_l[1][1] - 0.25 * S_EW.c_HD
    ) * (T3 + Q * (g_Y * g_Y) / g_L2_minus_g_Y2);

    return term1 + term2;
}

void Running_and_matching(double running_smeft[3][3],double running_weft_mb[5][5],double running_weft_low_scale[5][5],struct SMEFT High_scale,struct WEFT *Low_scale){
	
  int n_l = 3;
  int n_q = 3;
	
  //Intermediate EW scale operators
	
  struct SMEFT S_EW;
  struct WEFT W_EW;
	

  //definitions
  
  //CKM matrix PDG 2022
  double V_CKM[3][3];
		
  V_CKM[0][0] = 0.97435;
  V_CKM[0][1] = 0.225;
  V_CKM[0][2] = 0.00369;
  V_CKM[1][0] = 0.22486;
  V_CKM[1][1] = 0.97349;
  V_CKM[1][2] = 0.04182;
  V_CKM[2][0] = 0.00857;
  V_CKM[2][1] = 0.04110;
  V_CKM[2][2] = 0.999118;
  
  double delta[3][3];
  
  delta[0][0] = 1.0;
  delta[1][1] = 1.0;
  delta[2][2] = 1.0;
  delta[0][1] = 0.0;
  delta[0][2] = 0.0;
  delta[1][2] = 0.0;
  delta[2][1] = 0.0;
  delta[2][0] = 0.0;
  delta[1][0] = 0.0;
		
  // Gfermi (GeV-2) PDG
		
  double GF =1.166378e-5;

  //Running from 1 TeV to the EW scale 
	
		
// Running and mixing of operators
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        for (int l1 = 0; l1 < 3; l1++) {
            for (int l2 = 0; l2 < 3; l2++) {
                S_EW.c_ll[i][j][l1][l2] = High_scale.c_ll[i][j][l1][l2];
                S_EW.c3_lq[i][j][l1][l2] = High_scale.c3_lq[i][j][l1][l2];
                S_EW.c_ledq[i][j][l1][l2] = running_smeft[0][0] * High_scale.c_ledq[i][j][l1][l2];
                S_EW.c_lequ[i][j][l1][l2] = running_smeft[1][1] * High_scale.c_lequ[i][j][l1][l2] + running_smeft[1][2] * High_scale.c3_lequ[i][j][l1][l2];
                S_EW.c3_lequ[i][j][l1][l2] = running_smeft[2][1] * High_scale.c_lequ[i][j][l1][l2] + running_smeft[2][2] * High_scale.c3_lequ[i][j][l1][l2];
                S_EW.c3_ledq[i][j][l1][l2] = High_scale.c3_ledq[i][j][l1][l2];
                S_EW.c_lq[i][j][l1][l2] = High_scale.c_lq[i][j][l1][l2];
                S_EW.c_ld[i][j][l1][l2] = High_scale.c_ld[i][j][l1][l2];
                S_EW.c_lu[i][j][l1][l2] = High_scale.c_lu[i][j][l1][l2];
                S_EW.c_le[i][j][l1][l2] = High_scale.c_le[i][j][l1][l2];
            }
        }

        S_EW.c3_phi_l[i][j] = High_scale.c3_phi_l[i][j];
        S_EW.c3_phi_e[i][j] = High_scale.c3_phi_e[i][j];
        S_EW.c3_phi_q[i][j] = High_scale.c3_phi_q[i][j];
        S_EW.c3_phi_le[i][j] = High_scale.c3_phi_le[i][j];
        S_EW.c_phi_ud[i][j] = High_scale.c_phi_ud[i][j];
        S_EW.c_phi_u[i][j] = High_scale.c_phi_u[i][j];
        S_EW.c_phi_d[i][j] = High_scale.c_phi_d[i][j];
        S_EW.c_phi_e[i][j] = High_scale.c_phi_e[i][j];
        S_EW.c_phi_q[i][j] = High_scale.c_phi_q[i][j];
        S_EW.c_phi_l[i][j] = High_scale.c_phi_l[i][j];
    }
}




S_EW.c_HD = High_scale.c_HD;
S_EW.c_HWB = High_scale.c_HWB ;
     
   


// Matching at EW scale CC

for (int alpha = 0; alpha < 3; alpha++){
    for (int beta = 0; beta < 3; beta++){
        for (int up = 0; up < 2; up++){
            for (int down = 0; down < 3; down++){
                double V_jd_sum = 0.0;
                double V_jd_c3_lq_sum = 0.0;

                for (int i = 0; i < 3; i++) {
                    V_jd_sum += V_CKM[i][down] * S_EW.c3_phi_q[up][i];
                    V_jd_c3_lq_sum += V_CKM[i][down] * S_EW.c3_lq[alpha][beta][up][i];
                }

                // epsilon_L
                W_EW.epsilon_CC[0][up][down][alpha][beta] = (1.0 / V_CKM[up][down]) * (
                    V_CKM[up][down] * S_EW.c3_phi_l[alpha][beta] + V_jd_sum * delta[alpha][beta] - V_jd_c3_lq_sum
                );

                // epsilon_R
                W_EW.epsilon_CC[1][up][down][alpha][beta] = (1.0 / (2 * V_CKM[up][down])) * S_EW.c_phi_ud[up][down] * delta[alpha][beta];
                
                // epsilon_S
                double complex lequ_sum = 0.0;
                double complex ledq = conj(S_EW.c_ledq[beta][alpha][down][up]);

                for (int i = 0; i < 3; i++) {
                    lequ_sum += V_CKM[i][down] * conj(S_EW.c_lequ[beta][alpha][i][up]);
                }

                W_EW.epsilon_CC[2][up][down][alpha][beta] = -(1.0 / (2 * V_CKM[up][down])) * (lequ_sum + ledq);

                // epsilon_P
                W_EW.epsilon_CC[3][up][down][alpha][beta] = -(1.0 / (2 * V_CKM[up][down])) * (lequ_sum - ledq);

                // epsilon_T
                double lequ_t_sum = 0.0;
                for (int i = 0; i < 3; i++) {
                    lequ_t_sum += V_CKM[i][down] * conj(S_EW.c3_lequ[beta][alpha][i][up]);
                }

                W_EW.epsilon_CC[4][up][down][alpha][beta] = -(2.0 / V_CKM[up][down]) * lequ_t_sum;
            }
        }
    }
}

double stheta2 = 0.23865; // MSbar scheme

// Function to calculate f(T3, Q) as per the Warsaw basis


// Vertex corrections as matrices
double delta_g_L_Ze[3][3];
double delta_g_R_Ze[3][3];
double delta_g_L_We[3][3];
double delta_g_L_Zu[3][3];
double delta_g_R_Zu[3][3];
double delta_g_L_Zd[3][3];
double delta_g_R_Zd[3][3];
double delta_g_L_Znu[3][3];

// Precompute vertex corrections for all flavors
for (int alpha = 0; alpha < 3; alpha++) {
    for (int beta = 0; beta < 3; beta++) {
        delta_g_L_Ze[alpha][beta] = -0.5 * S_EW.c3_phi_l[alpha][beta] - 0.5 * S_EW.c_phi_l[alpha][beta] + delta[alpha][beta] * f(-0.5, -1.0, S_EW);
        delta_g_R_Ze[alpha][beta] = -0.5 * S_EW.c_phi_e[alpha][beta] + delta[alpha][beta] * f(0.0, -1.0, S_EW);
        delta_g_L_We[alpha][beta] = S_EW.c3_phi_l[alpha][beta] + delta[alpha][beta] * (f(0.5, 0.0, S_EW) - f(-0.5, -1.0, S_EW));
        delta_g_L_Zu[alpha][beta] = 0.5 * S_EW.c3_phi_q[alpha][beta] - 0.5 * S_EW.c_phi_q[alpha][beta] + delta[alpha][beta] * f(0.5, 2.0 / 3.0, S_EW);
        delta_g_R_Zu[alpha][beta] = -0.5 * S_EW.c_phi_u[alpha][beta] + delta[alpha][beta] * f(0.0, 2.0 / 3.0, S_EW);
        delta_g_L_Zd[alpha][beta] = -0.5 * S_EW.c3_phi_q[alpha][beta] - 0.5 * S_EW.c_phi_q[alpha][beta] + delta[alpha][beta] * f(-0.5, -1.0 / 3.0, S_EW);
        delta_g_R_Zd[alpha][beta] = -0.5 * S_EW.c_phi_d[alpha][beta] + delta[alpha][beta] * f(0.0, -1.0 / 3.0, S_EW);
        delta_g_L_Znu[alpha][beta] = delta_g_L_We[alpha][beta]+delta_g_L_Ze[alpha][beta];
    }
}

// Main loop to calculate epsilon parameters
for (int alpha = 0; alpha < 3; alpha++) {
    for (int beta = 0; beta < 3; beta++) {
        // epsilon^{e,L}
        W_EW.epsilon_NC[0][0][0][alpha][beta] = delta[0][beta] * delta_g_L_We[alpha][0] +
            delta[alpha][beta] * delta_g_L_Ze[0][0] +
            (-1.0 + 2.0 * stheta2) * (delta_g_L_We[alpha][beta] + delta_g_L_Ze[alpha][beta]) -
            0.5 * S_EW.c_ll[0][0][alpha][beta] +
            delta[alpha][0] * (delta_g_L_We[0][beta] +
            0.5 * delta[0][beta] * (-2.0 * (delta_g_L_We[0][0] + delta_g_L_We[1][1]) + S_EW.c_ll[0][1][1][0])) -
            0.5 * S_EW.c_ll[alpha][beta][0][0];

        // epsilon^{e,R}
        W_EW.epsilon_NC[1][0][0][alpha][beta] = 2.0 * stheta2 * (delta_g_L_We[alpha][beta] + delta_g_L_Ze[alpha][beta]) +
            delta[alpha][beta] * delta_g_R_Ze[0][0] -
            0.5 * S_EW.c_le[alpha][beta][0][0];

        // epsilon^{u,L}
        W_EW.epsilon_NC[0][1][1][alpha][beta] = delta[alpha][beta] * delta_g_L_Zu[0][0] +
            (1.0 - 4.0 / 3.0 * stheta2) * delta_g_L_Znu[alpha][beta] -
            0.5 * (S_EW.c3_lq[alpha][beta][0][0] + S_EW.c_lq[alpha][beta][0][0]);

        // epsilon^{u,R}
        W_EW.epsilon_NC[1][1][1][alpha][beta] = -4.0 / 3.0 * stheta2 * delta_g_L_Znu[alpha][beta] +
            delta[alpha][beta] * delta_g_R_Zu[0][0] -
            0.5 * S_EW.c_lu[alpha][beta][0][0];

        // epsilon^{d,L}
        W_EW.epsilon_NC[0][2][2][alpha][beta] = delta[alpha][beta] * delta_g_L_Zd[0][0] +
            (1.0 / 3.0) * (-3.0 + 2.0 * stheta2) * delta_g_L_Znu[alpha][beta] +
            0.5 * (S_EW.c3_lq[alpha][beta][0][0] - S_EW.c_lq[alpha][beta][0][0]);

        // epsilon^{d,R}
        W_EW.epsilon_NC[1][2][2][alpha][beta] = (2.0 / 3.0) * stheta2 * delta_g_L_Znu[alpha][beta] +
            delta[alpha][beta] * delta_g_R_Zd[0][0] -
            0.5 * S_EW.c_ld[alpha][beta][0][0];
    }
}

		
    // Running below the EW
		
			
	//Light quarks 
			
			
	for(int i=0 ; i < 2; i++)
	{
	  for( int j = 0; j<2; j++)
	  {
	    for( int l1=0; l1<3; l1++)
	    {
	      for( int l2=0; l2<3; l2++)
	      {
		 Low_scale->epsilon_CC[0][i][j][l1][l2] = running_weft_low_scale[0][0]*W_EW.epsilon_CC[0][i][j][l1][l2];
		 
		 Low_scale->epsilon_CC[1][i][j][l1][l2] = running_weft_low_scale[1][1]*W_EW.epsilon_CC[1][i][j][l1][l2];
		 
		 Low_scale->epsilon_CC[2][i][j][l1][l2] = running_weft_low_scale[2][2]*W_EW.epsilon_CC[2][i][j][l1][l2]+running_weft_low_scale[2][3]*W_EW.epsilon_CC[3][i][j][l1][l2]+ running_weft_low_scale[2][4]*W_EW.epsilon_CC[4][i][j][l1][l2];
					
		 Low_scale->epsilon_CC[3][i][j][l1][l2] = running_weft_low_scale[3][2]*W_EW.epsilon_CC[2][i][j][l1][l2]+running_weft_low_scale[3][3]*W_EW.epsilon_CC[3][i][j][l1][l2]+ running_weft_low_scale[3][4]*W_EW.epsilon_CC[4][i][j][l1][l2];
					
		 Low_scale->epsilon_CC[4][i][j][l1][l2] = running_weft_low_scale[4][2]*W_EW.epsilon_CC[2][i][j][l1][l2]+running_weft_low_scale[4][3]*W_EW.epsilon_CC[3][i][j][l1][l2] +running_weft_low_scale[4][4]*W_EW.epsilon_CC[4][i][j][l1][l2];
							
               }
	     }			
	   }
        }
			
	//Bottom quark 
			
	for(int i=0 ; i < 2; i++)
	{
	  for( int j = 2; j<3; j++)
	  {
	    for( int l1=0; l1<3; l1++)
	    {
	      for( int l2=0; l2<3; l2++)
	      {
		 Low_scale->epsilon_CC[0][i][j][l1][l2] = running_weft_mb[0][0]*W_EW.epsilon_CC[0][i][j][l1][l2];
		 
		 Low_scale->epsilon_CC[1][i][j][l1][l2] = running_weft_mb[1][1]*W_EW.epsilon_CC[1][i][j][l1][l2];
		 
		 Low_scale->epsilon_CC[2][i][j][l1][l2] = running_weft_mb[2][2]*W_EW.epsilon_CC[2][i][j][l1][l2]+running_weft_mb[2][3]*W_EW.epsilon_CC[3][i][j][l1][l2]+ running_weft_mb[2][4]*W_EW.epsilon_CC[4][i][j][l1][l2];
					
		 Low_scale->epsilon_CC[3][i][j][l1][l2] = running_weft_mb[3][2]*W_EW.epsilon_CC[2][i][j][l1][l2]+running_weft_mb[3][3]*W_EW.epsilon_CC[3][i][j][l1][l2]+ running_weft_mb[3][4]*W_EW.epsilon_CC[4][i][j][l1][l2];
					
		 Low_scale->epsilon_CC[4][i][j][l1][l2] = running_weft_mb[4][2]*W_EW.epsilon_CC[2][i][j][l1][l2]+running_weft_mb[4][3]*W_EW.epsilon_CC[3][i][j][l1][l2] +running_weft_mb[4][4]*W_EW.epsilon_CC[4][i][j][l1][l2];
							
               }
	     }			
	   }
        }
			
	//NC Running, check if they have significative running belowe EW scale
	
	
	for(int i=0 ; i < 3; i++)
	{
	    for( int l1=0; l1<3; l1++)
	    {
	      for( int l2=0; l2<3; l2++)
	      {
	      
		   Low_scale->epsilon_NC[0][i][i][l1][l2] = W_EW.epsilon_NC[0][i][i][l1][l2];
		 
		   Low_scale->epsilon_NC[1][i][i][l1][l2] = W_EW.epsilon_NC[1][i][i][l1][l2];
						
               }
	     }			
	   
        }
		     
			
		
			
	
		
	
}








void running_filling(double running_smeft[3][3],double running_weft_mb[5][5],double running_weft_low_scale[5][5] ) {
  
 
 //Default mode High scale 1TeV, low scale 2GeV 


      //Filling the smeft running matrix
        running_smeft[0][0] = 1.196;
        running_smeft[0][1] = 0.0;
        running_smeft[0][2] = 0.0;
    
        running_smeft[1][0] = 0.0;
        running_smeft[2][0] = 0.0;
    
        running_smeft[1][1] = 1.20;
        running_smeft[1][2] = -0.19;
        running_smeft[2][1] = -0.0038;
        running_smeft[2][2] = 0.958;


      //Filling the WEFT running matrix mb 

        running_weft_mb[0][0] = 1.009;
        running_weft_mb[1][1] = 1.005;
        running_weft_mb[2][2] = 1.456;
 
        running_weft_mb[3][3] = 1.456;
        running_weft_mb[3][4] =-0.018;
        running_weft_mb[4][3] = -0.00034;
        running_weft_mb[4][4] = 0.878;


      //Filling the WEFT running matrix low scale 

        running_weft_low_scale[0][0] = 1.008;
        running_weft_low_scale[1][1] = 1.004;
        running_weft_low_scale[2][2] = 1.722;
 
        running_weft_low_scale[3][3] = 1.723;
        running_weft_low_scale[3][4] = -0.024;
        running_weft_low_scale[4][3] = -0.00043;
        running_weft_low_scale[4][4] = 0.824;

   
  
  
}







