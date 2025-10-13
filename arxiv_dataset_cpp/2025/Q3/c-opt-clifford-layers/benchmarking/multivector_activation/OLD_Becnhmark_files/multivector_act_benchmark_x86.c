#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include ".././tsc_x86.h"
#include "../../clib/multivector_activation/multivector_act.c"

#define B 32        // Batch size
#define C 64        // Number of channels
#define I 8         // Number of blades in the embedded tensor
#define K 4         // Number of kernel blades (length of kernel_blades array)
#define REPEAT 10000

// Global arrays for data (using fixed sizes)
float input[B * C * I];
float output[B * C * I];
float weights[C * K];
float bias[C];
int kernel_indices[K] = {0, 1, 2, 3};


// Initializes data for "linear" mode (weights and bias are used)
void init_data_linear() {
    for (int i = 0; i < B * C * I; i++) {
        input[i] = (float)rand() / (float)RAND_MAX;
        output[i] = 0.0f;
    }
    for (int i = 0; i < C * K; i++) {
        weights[i] = (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < C; i++) {
        bias[i] = (float)rand() / (float)RAND_MAX;
    }
}

// Initializes data for "sum" or "mean" mode (weights and bias are not used)
void init_data_sum_mean() {
    for (int i = 0; i < B * C * I; i++) {
        input[i] = (float)rand() / (float)RAND_MAX;
        output[i] = 0.0f;
    }
}

// Benchmarks one mode (agg_mode: 0=linear, 1=sum, 2=mean) and prints the average cycles.
void benchmark_mode(int mode, const char* mode_name) {
    uint64_t total_cycles = 0;
    int i;

    // Initialize data according to the mode
    if (mode == 0) { 
        init_data_linear();
    } else {
        init_data_sum_mean();
    }

    // Warmup call (to minimize first-run overhead)
    multivector_act_forward(input,
                            (mode == 0 ? weights : NULL),
                            (mode == 0 ? bias : NULL),
                            B, C, I, K, kernel_indices, mode, output);

    // Benchmark loop
    for (i = 0; i < REPEAT; i++) {
        uint64_t start = start_tsc();
        multivector_act_forward(input,
                                (mode == 0 ? weights : NULL),
                                (mode == 0 ? bias : NULL),
                                B, C, I, K, kernel_indices, mode, output);
        uint64_t end = stop_tsc(start);
        total_cycles += (end - start);
    }

    printf("[%s] Average cycles: %llu\n", mode_name, total_cycles / REPEAT);

    const int blades = I;
    const int kernel_blades = K;
    const int pairs = B * C;
    
    double flops_per_pair;
    if (mode == 0) {
        // Linear: K muls + K adds + 1 bias add + sigmoid + broadcast muls
        flops_per_pair = 2 * kernel_blades + 1 + 32 + blades;
    } else if (mode == 1) {
        // Sum: K adds + sigmoid + broadcast muls
        flops_per_pair = kernel_blades + 32 + blades;
    } else {
        // Mean: K adds + 1 division + sigmoid + broadcast muls
        flops_per_pair = kernel_blades + 1 + 32 + blades;
    }

    double flops_per_call = flops_per_pair * pairs;
    double avg_cycles = (double)total_cycles / REPEAT;
    double flops_per_cycle = flops_per_call / avg_cycles;

    printf("[%s] Estimated: %.3f FLOP/cycle\n", mode_name, flops_per_cycle);

}

int main() {
    srand((unsigned int)time(NULL));

    printf("Benchmarking multivector_act_forward:\n");
    benchmark_mode(0, "Linear");
    benchmark_mode(1, "Sum");
    benchmark_mode(2, "Mean");

    return 0;
}
