#include "dcl.h"


// This is the testbench for verifying the correctness of
// the templated arbitrary precision of FP (TFP)
// and the flexible run-time reconfigurable FP (FFP)

// TFP: Templated Floating Point --> arbitrary precision of FP but fixed and pre-defined
// FFP: Flexible Floating Point --> run-time reconfigurable FP

// Variables:
// myEB: specified exponent bitwidth during runtime
// myMB: specified mantissa bitwidth during runtime
// TOTAL_FLEX: total number of bits including sign bit; specified in dcl.h

// err_TFP: error percentage between golden 32-bit floating point v.s. TFP
// err_FFP: error percentage between golden 32-bit floating point v.s. FFP
// err_TFP_FFP: error percentage between TFP and FFP (this is expected to be very small)

// max_err_TFP: max error percentage between golden 32-bit floating point v.s. TFP
// max_err_FFP: max error percentage between golden 32-bit floating point v.s. FFP
// max_err_TFP_FFP: max error percentage between TFP and FFP (this is expected to be very small)
// alpha_TFP: error threshold for max_err_TFP
// alpha_TFP_FFP: error threshold for max_err_TFP_FFP

#define alpha_TFP 0.5           // in percentage
#define alpha_TFP_FFP 0.000001  // in percentage

int main(void)
{
    int myEB, myMB;
    
    for(myEB = 3; myEB <= 6; myEB++) {
        myMB = TOTAL_FLEX - 1 - myEB;
    
        printf("################ Testing precision %d-bit: <EB: %d, MB: %d> ################\n", TOTAL_FLEX, myEB, myMB);
        printf("...\n");

        int total_test_TFP = 0;
        int total_pass_TFP = 0;
        
        int total_test_TFP_FFP = 0;
        int total_pass_TFP_FFP = 0;

        float err_TFP, err_FFP, err_TFP_FFP;
        float max_err_TFP = -999;
        float max_err_FFP = -999;
        float max_err_TFP_FFP = -999;
        
        for(int i = 0; i < DATA_PAIRS * 2; i+= 2) {

            float a, b, c, c_tfp, c_ffp;
            bool overflow_tfp = false;
            bool overflow_ffp = false;
            
            // floating point operands range
            // setting it to be small so 16-bit multiplications do not overflow too often
            a = randomFloatRange(-100, 100);
            b = randomFloatRange(-100, 100);

            c = a * b;

            c_tfp = templated_float_multiply(a, b, &overflow_tfp, myEB, myMB);
            c_ffp = flexible_float_multiply(a, b, &overflow_ffp, myEB, myMB, false);

            err_TFP = (c - c_tfp) / c * 100;
            err_FFP = (c - c_ffp) / c * 100;
            err_TFP_FFP = (c_ffp - c_tfp) / c_tfp * 100;
            
            // error between templated FP and golden 32-bit
            total_test_TFP++;
            if( err_TFP < alpha_TFP ) {
                total_pass_TFP++;
                max_err_TFP = max(max_err_TFP, err_TFP);
            }
            else if(overflow_tfp == true) {
                // large error due to overflow; ignore
                total_test_TFP--;
            }
            else {
                printf("[test %d] Error between TFP and golden 32-bit: %f\%\n", i, err_TFP);
                printFloatBits(a);
                printFloatBits(b);
                printFloatBits(c);
                printFloatBits(c_tfp);
                printf("\n\n");
                max_err_TFP = max(max_err_TFP, err_TFP);
            }

            // error between templated FP and flexible FP
            total_test_TFP_FFP++;
            if( err_TFP_FFP < alpha_TFP_FFP) {
                total_pass_TFP_FFP++;
                max_err_TFP_FFP = max(max_err_TFP_FFP, err_TFP_FFP);
            }
            else if(overflow_tfp == true && overflow_ffp == true) {
                // large error due to overflow; ignore
                total_test_TFP_FFP--;
                }
            else {
                printf("[test %d] Error between TFP and FFP: %f\%\n", i, err_TFP_FFP);
                printFloatBits(a);
                printFloatBits(b);
                printFloatBits(c);
                printFloatBits(c_tfp);
                printFloatBits(c_ffp);
                printf("\n\n");
                max_err_TFP_FFP = max(max_err_TFP_FFP, err_TFP_FFP);
            }
        }

        printf("\n################ Test Results ################\n");
        
        printf("Templated arbitrary precision floating point (error threshold %f)\n", alpha_TFP);
        printf("TOTAL PASSED %.2f\% (%d out of %d)\n\n", 
                total_pass_TFP*1.0/(total_test_TFP)*100.0, total_pass_TFP, total_test_TFP);

        printf("Flexible run-time reconfigurable floating point (error threshold %f)\n", alpha_TFP_FFP);
        printf("TOTAL PASSED %.2f\% (%d out of %d)\n\n", 
                total_pass_TFP_FFP*1.0/(total_test_TFP_FFP)*100.0, total_pass_TFP_FFP, total_test_TFP_FFP);
        
        printf("Max error between FFP and standard 32-bit: %f\%\n", max_err_TFP);
        printf("Max error between FFP and TFP: %f\%\n\n\n", max_err_TFP_FFP);

        max_err_TFP = -999;
        max_err_FFP = -999;
        max_err_TFP_FFP = -999;
    }

    return 0;
}