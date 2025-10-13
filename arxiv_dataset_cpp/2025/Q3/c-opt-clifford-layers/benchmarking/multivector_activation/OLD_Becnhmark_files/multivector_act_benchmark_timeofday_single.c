// benchmark.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <sys/time.h>

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


#define B 256
#define C 256
#define I 16
#define K 4
#define REPEAT 1000
#define FREQUENCY 3.268e9

float input[B * C * I];
float output[B * C * I];
float weights[C * K];
float bias[C];
int kernel_indices[K] = {0,1,2,3};

void init_data_linear() {
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

void init_data_sum_mean() {
    for (int i = 0; i < B * C * I; i++) {
        input[i] = (float)rand() / RAND_MAX;
        output[i] = 0.0f;
    }
}

double get_timeofday_seconds() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

// function‐pointer type
typedef void (*mvaf_fn)(
    const float*, const float*, const float*,
    int, int, int, int, const int*, int, float*
);

void run_benchmark(const char* version,
                   mvaf_fn fn,
                   int mode,
                   const char* mode_name)
{
    // data init
    if (mode == 0) init_data_linear();
    else           init_data_sum_mean();

    // warm-up
    fn(input,
       mode==0 ? weights : NULL,
       mode==0 ? bias    : NULL,
       B, C, I, K, kernel_indices, mode, output);

    double start = get_timeofday_seconds();
    for (int rep = 0; rep < REPEAT; rep++) {
        fn(input,
           mode==0 ? weights : NULL,
           mode==0 ? bias    : NULL,
           B, C, I, K, kernel_indices, mode, output);
    }
    double end = get_timeofday_seconds();

    double avg_time   = (end - start) / REPEAT;
    double avg_cycles = avg_time * FREQUENCY;

    // flop count per (B,C) pair
    int blades = I, kb = K;
    double flops_per_pair = (mode==0
      ? (2.*kb + 1 + 32 + blades)    // linear
      : (mode==1
         ? (1.*(kb-1) + 32 + blades)     // sum
         : (1.*(kb-1) + 1 + 32 + blades) // mean
        )
    );
    double total_pairs    = (double)B * C;
    double flops_per_call = flops_per_pair * total_pairs;
    double flops_per_cycle = flops_per_call / avg_cycles;

    printf("%-8s | %-6s | %8.2f µs | %8.0f cycles | %6.2f FLOP/cycle\n",
           mode_name, version,
           avg_time*1e6,
           avg_cycles,
           flops_per_cycle);
}

int main() {
    srand((unsigned)time(NULL));
    const char* versions[4]   = {"Baseline","Opt1","Opt2"};
    mvaf_fn     fns[4]        = {baseline_forward,
                                opt1_forward,
                                opt2_forward};
    const char* mode_names[3] = {"Linear","Sum","Mean"};

    printf("multivector_act_forward benchmarking\n");
    printf(" Version  | Mode   |    Time   |   Cycles   | FLOP/Cycle\n");
    printf("--------------------------------------------------------\n");

    for (int mode = 0; mode < 3; mode++) {
        for (int v = 0; v < 3; v++) {
            run_benchmark(versions[v], fns[v], mode, mode_names[mode]);
        }
    }

    return 0;
}
