#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include ".././tsc_x86.h"
#include "../../clib/multivector_activation/multivector_act.c"

// Baseline
#define multivector_act_forward baseline_forward
#include "../../clib/multivector_activation/multivector_act.c"
#undef multivector_act_forward

// Opt1
#define multivector_act_forward opt1_forward
#include "../../clib/multivector_activation/multivector_act_opt1.c"
#undef multivector_act_forward

// Opt2
#define multivector_act_forward opt2_forward
#include "../../clib/multivector_activation/multivector_act_opt2.c"
#undef multivector_act_forward

#define K 4
#define REPEAT 1000
#define FREQUENCY 1.8e9

// Max sizes (for static arrays)
#define MAX_B 512
#define MAX_C 512
#define MAX_I 8

float input[MAX_B * MAX_C * MAX_I];
float output[MAX_B * MAX_C * MAX_I];
float weights[MAX_C * K];
float bias[MAX_C];
int kernel_indices[K] = {0, 1, 2, 3};

void init_data_linear(int B, int C, int I) {
    for (int i = 0; i < B * C * I; i++) {
        input[i] = (float)rand() / RAND_MAX;
        output[i] = 0.0f;
    }
    for (int i = 0; i < C * K; i++) {
        weights[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < C; i++) {
        bias[i] = (float)rand() / RAND_MAX;
    }
}

void init_data_sum_mean(int B, int C, int I) {
    for (int i = 0; i < B * C * I; i++) {
        input[i] = (float)rand() / RAND_MAX;
        output[i] = 0.0f;
    }
}

typedef void (*mvaf_fn)(const float*, const float*, const float*, int, int, int, int, const int*, int, float*);

void run_benchmark(const char* version, mvaf_fn fn, int mode, const char* mode_name, int B, int C, int I) {
    if (mode == 0) init_data_linear(B, C, I);
    else           init_data_sum_mean(B, C, I);

    fn(input, mode == 0 ? weights : NULL, mode == 0 ? bias : NULL, B, C, I, K, kernel_indices, mode, output);

    uint64_t start = start_tsc();
    for (int rep = 0; rep < REPEAT; rep++) {
        fn(input, mode == 0 ? weights : NULL, mode == 0 ? bias : NULL, B, C, I, K, kernel_indices, mode, output);
    }
    uint64_t cycles = stop_tsc(start);

    double total_pairs = (double)B * C;
    double elapsed_sec = (double)cycles / FREQUENCY;
    double avg_time = elapsed_sec / REPEAT;
    double cycles_per_call = (double)cycles / REPEAT;

    int blades = I, kb = K;
    double flops_per_pair = (mode==0 ? (2.*kb + 1 + 32 + blades)
                            : (mode==1 ? (1.*kb + 32 + blades)
                                       : (1.*kb + 1 + 32 + blades)));
    double flops_per_call = flops_per_pair * total_pairs;
    double flops_per_cycle = flops_per_call / cycles_per_call;
    double gflops = flops_per_call / avg_time / 1e9;

    printf("%-6s,%s,%4d,%4d,%2d,%8.2f,%.0f,%10.2f,%.2f\n",
           version, mode_name, B, C, I,
           avg_time * 1e6, cycles_per_call,
           flops_per_cycle, gflops);
}

int main() {
    srand((unsigned)time(NULL));
    const char* versions[3]   = {"Baseline", "Opt1", "Opt2"};
    mvaf_fn fns[3]            = {baseline_forward, opt1_forward, opt2_forward};
    const char* mode_names[3] = {"Linear", "Sum", "Mean"};

    printf("version,mode,B,C,I,time_us,cycles,FLOP/cycle,GFLOPS\n");

    //TODO: vary batch sizes fix everything else 

    for (int B = 8; B <= 256; B *= 2) {
        for (int C = 16; C <= 512; C *= 2) {
            int I = 8; // Keep blades fixed for now; can loop over this too
            for (int mode = 0; mode < 3; mode++) {
                //for (int v = 0; v < 3; v++) {
                int v = 2; // Only run the baseline for now
                run_benchmark(versions[v], fns[v], mode, mode_names[mode], B, C, I);
                //}
            }
        }
    }

    return 0;
}
