/* GLoBES -- General LOng Baseline Experiment Simulator
 * (C) 2002 - 2007,  The GLoBES Team
 *
 * GLoBES is mainly intended for academic purposes. Proper
 * credit must be given if you use GLoBES or parts of it. Please
 * read the section 'Credit' in the README file.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */
 
/* GLoBES-EFT -- Simple Example Program
 * 
 * This example demonstrates the basic GLoBES-EFT features:
 * - Loading and checking production/detection coefficients
 * - Computing probabilities with EFT effects
 * - Setting WEFT parameters and computing chi-squared
 * - Switching to SMEFT mode and setting SMEFT parameters
 * 
 * Compile with: make example-smeft
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#include <globes/globes.h>   /* GLoBES library */
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include "smeft.h"


#define MAX_PARAMS 2405       // Adjust as needed
#define MAX_NAME_LENGTH 64

char smeft_param_strings[MAX_PARAMS][MAX_NAME_LENGTH];
int number_param = MAX_PARAMS;   
static char **smeftparamnames = NULL;
// Function to get index of a parameter
int get_index(char smeft_param_strings[][MAX_NAME_LENGTH], int size, char *str) {
    for (int i = 0; i < size; i++) {
        if (strcmp(smeft_param_strings[i], str) == 0)
            return i;
    }
    return -1;
}

int copyNames() {
    int nP = glbGetNumOfOscParams();
    smeftparamnames = calloc(nP, sizeof(char*));
    if (!smeftparamnames) { perror("calloc"); return -1; }

    for (int k = 0; k < nP; k++) {
        
        if (smeft_param_strings[k][0] != '\0') {
            smeftparamnames[k] = malloc(MAX_NAME_LENGTH);
            if (!smeftparamnames[k]) { perror("malloc"); return -1; }
       
            strncpy(smeftparamnames[k], smeft_param_strings[k], MAX_NAME_LENGTH - 1);
            smeftparamnames[k][MAX_NAME_LENGTH - 1] = '\0';
        } else {
            smeftparamnames[k] = NULL;  
        }
    }
    return 0;
}



// Function to set oscillation parameter by name
int smeft_glbSetOscParamByName(glb_params p, double value, char *name) {
    int index = get_index(smeft_param_strings, number_param, name);
    if (index == -1) {
        fprintf(stderr, "Error: Parameter '%s' not found.\n", name);
        return -1;
    }
    glbSetOscParams(p, value, index);
    return 0;
}

int smeft_remove(glb_params p){
    
    for(int i = 6; i < number_param; i++){
       
            glbSetOscParams(p,0.0,i);
            
        
    }
    return 0;
}

int main(int argc, char *argv[])
{
    printf("=== GLoBES-EFT Simple Example ===\n\n");
    
    /* Initialize GLoBES and load experiment */
    glbInit(argv[0]);
   
    
    /* Initialize GLoBES-EFT probability engine */
    smeft_init_probability_engine_3();
    glbRegisterProbabilityEngine(number_param, &smeft_probability_matrix,
                                &smeft_set_oscillation_parameters,
                                &smeft_get_oscillation_parameters, NULL);
                                
    glbInitExperiment("NF-smeft.glb", &glb_experiment_list[0], &glb_num_of_exps);
    
    

    /* Set up standard oscillation parameters */
    double theta12 = asin(sqrt(0.307))/2;
    double theta13 = asin(sqrt(0.022))/2;
    double theta23 = asin(sqrt(0.563))/2;
    double deltacp = 1.36;
    double sdm = 7.42e-5;
    double ldm = 2.51e-3;
    
    glb_params true_values = glbAllocParams();
    glb_params test_values = glbAllocParams();
    
    glbDefineParams(true_values, theta12, theta13, theta23, deltacp, sdm, ldm);
    glbSetDensityParams(true_values, 1.0, GLB_ALL);
    glbDefineParams(test_values, theta12, theta13, theta23, deltacp, sdm, ldm);
    glbSetDensityParams(test_values, 1.0, GLB_ALL);
    
    /*Set to 0 all smeft and weft operators*/
    smeft_remove(true_values);
    smeft_remove(test_values);
    
    /*Set the true values*/ 
    glbSetOscillationParameters(true_values);
    glbSetRates();
    copyNames();
    glbSetParamNames(smeftparamnames);
    /* ========================================= */
    /* 1. Check production and detection coefficients */
    /* ========================================= */
    printf("1. Production and Detection Coefficients\n");
    printf("=========================================\n");
    
    double E = 2.0; /* GeV */
    printf("At energy E = %.1f GeV:\n\n", E);
    
    /* Production coefficients */
    printf("Production coefficients (flux #0):\n");
    printf("  LL (electron): %12.6e\n", glbEFTFluxCoeff(0, 0, GLB_EFT_L, GLB_EFT_L, 0, E));
    printf("  LR (electron): %12.6e\n", glbEFTFluxCoeff(0, 0, GLB_EFT_L, GLB_EFT_R, 0, E));
    printf("  LS (electron): %12.6e\n", glbEFTFluxCoeff(0, 0, GLB_EFT_L, GLB_EFT_S, 0, E));
    printf("  LP (electron): %12.6e\n", glbEFTFluxCoeff(0, 0, GLB_EFT_L, GLB_EFT_P, 0, E));
    printf("  LT (electron): %12.6e\n", glbEFTFluxCoeff(0, 0, GLB_EFT_L, GLB_EFT_T, 0, E));
    
    /* Detection coefficients */
    printf("\nDetection coefficients (xsec #0):\n");
    printf("  LL (electron): %12.6e\n", glbEFTXSecCoeff(0, 0, GLB_EFT_L, GLB_EFT_L, 0, +1, E));
    printf("  LR (electron): %12.6e\n", glbEFTXSecCoeff(0, 0, GLB_EFT_L, GLB_EFT_R, 0, +1, E));
    printf("  LS (electron): %12.6e\n", glbEFTXSecCoeff(0, 0, GLB_EFT_L, GLB_EFT_S, 0, +1, E));
    printf("  LP (electron): %12.6e\n", glbEFTXSecCoeff(0, 0, GLB_EFT_L, GLB_EFT_P, 0, +1, E));
    printf("  LT (electron): %12.6e\n", glbEFTXSecCoeff(0, 0, GLB_EFT_L, GLB_EFT_T, 0, +1, E));
    
    
    /* ========================================= */
    /* 2. Compute EFT probabilities */
    /* ========================================= */
    printf("2. EFT Probabilities\n");
    printf("====================\n");
    
    double baseline = 1300; /* km */
    int flux_id = 0, cross_id = 0;
    
    printf("Energy = %.1f GeV, Baseline = %.0f km\n", E, baseline);
    printf("Using flux_id = %d, cross_id = %d\n\n", flux_id, cross_id);
    
    /* Calculate some probabilities, the main difference with standard globes is the need to specify the cross section and flux  */
    double P_mue = smeft_glbVacuumProbability(2, 1, +1, E, baseline, flux_id, cross_id);
    double P_mumu = smeft_glbVacuumProbability(2, 2, +1, E, baseline, flux_id, cross_id);
    double P_mutau = smeft_glbVacuumProbability(2, 3, +1, E, baseline, flux_id, cross_id);
    
    printf("Neutrino probabilities:\n");
    printf("  P(νμ → νe) = %.6f\n", P_mue);
    printf("  P(νμ → νμ) = %.6f\n", P_mumu);
    printf("  P(νμ → ντ) = %.6f\n", P_mutau);
    
    /* ========================================= */
    /* 3. WEFT mode - modify parameter and compute chi2 */
    /* ========================================= */
    printf("3. WEFT Mode\n");
    printf("============\n");
    printf("Default mode is WEFT (Wilson coefficients at 2 GeV)\n");
    
    
    
    /* Set a WEFT parameter */
    printf("Setting WEFT parameter: |ε_R^{ud}|_{ee} = 1e-3\n");
    glbSetOscParamByName(test_values, 0.001, "ABS_EPS_CC_R_ud_EE");
    glbSetOscParamByName(test_values, 0.0, "ARG_EPS_CC_L_ud_EE");
    
    /* Compute chi-squared with WEFT operators */
    double chi2_weft = glbChiSys(test_values, GLB_ALL, GLB_ALL);
    printf("Δχ² = %.3f\n\n", chi2_weft);
    
    
    /* Reset WEFT parameter and set SMEFT parameter instead */
    smeft_remove(test_values);
    
    /* ========================================= */
    /* 4. SMEFT mode - switch and modify parameter */
    /* ========================================= */
    printf("\n4. SMEFT Mode\n");
    printf("=============\n");
    
    /* Switch to SMEFT mode */
    printf("Switching to SMEFT mode (Wilson coefficients at 1 TeV)\n");
    printf("Will run RGE: 1 TeV → EW scale → WEFT matching → 2 GeV\n");
    
    glbSetOscParamByName(test_values, 1, "SMEFT_FLAG");
    
    
    
     /* Set a SMEFT parameter - four-fermion operator */
    printf("Setting SMEFT parameter: |C_ll|_{eeee} = 1e-3\n");
    glbSetOscParamByName(test_values, 1e-3, "ABS_C_LL_EE_EE");
    glbSetOscParamByName(test_values, 0.0, "ARG_C_LL_EE_EE");
    
    
    
    /* Compute chi-squared with SMEFT effects */
    double chi2_smeft = glbChiSys(test_values, GLB_ALL, GLB_ALL);
    printf("Δχ² = %.3f\n", chi2_smeft);
    
    
    
    /* Cleanup */
    glbFreeParams(true_values);
    glbFreeParams(test_values);
    smeft_free_probability_engine();
    
    return 0;
}
