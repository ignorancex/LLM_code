#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include ".././tsc_x86.h"
#include "../../clib/multivector_activation/multivector_act.c" // Ideally should be header, but OK

#define DEFAULT_REPEAT 1000
#define CPU_GHZ 1.8 // Adjust to your actual CPU speed

float* input = NULL;
float* output = NULL;
float* weights = NULL;
float* bias = NULL;
int kernel_indices[4] = {0, 1, 2, 3};

void random_init(float* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)rand() / (float)RAND_MAX;
    }
}

void init_data(int B, int C, int I, int mode) {
    int input_size = B * C * I;
    int weight_size = C * 4;
    int bias_size = C;

    input = malloc(sizeof(float) * input_size);
    output = malloc(sizeof(float) * input_size);
    if (mode == 0) {
        weights = malloc(sizeof(float) * weight_size);
        bias = malloc(sizeof(float) * bias_size);
    } else {
        weights = NULL;
        bias = NULL;
    }

    random_init(input, input_size);
    if (weights) random_init(weights, weight_size);
    if (bias) random_init(bias, bias_size);

    for (int i = 0; i < input_size; i++) {
        output[i] = 0.0f;
    }
}

void free_data() {
    if (input) free(input);
    if (output) free(output);
    if (weights) free(weights);
    if (bias) free(bias);
}

void benchmark_mode(int mode, const char* mode_name, int B, int C, int I, int repeat) {
    uint64_t total_cycles = 0;

    init_data(B, C, I, mode);

    multivector_act_forward(input,
                            (mode == 0 ? weights : NULL),
                            (mode == 0 ? bias : NULL),
                            B, C, I, 4, kernel_indices, mode, output);

    for (int i = 0; i < repeat; i++) {
        uint64_t start = start_tsc();
        multivector_act_forward(input,
                                (mode == 0 ? weights : NULL),
                                (mode == 0 ? bias : NULL),
                                B, C, I, 4, kernel_indices, mode, output);
        uint64_t cycles = stop_tsc(start);
        total_cycles += cycles;
    }

    const int blades = I;
    const int kernel_blades = 4;
    const int pairs = B * C;

    double flops_per_pair;
    if (mode == 0) {
        flops_per_pair = 2 * kernel_blades + 1 + 32 + blades;
    } else if (mode == 1) {
        flops_per_pair = kernel_blades + 32 + blades;
    } else {
        flops_per_pair = kernel_blades + 1 + 32 + blades;
    }

    double flops_per_call = flops_per_pair * pairs;
    double avg_cycles = (double)total_cycles / repeat;
    double input_size = (double)(B * C * I); // number of floats

    double seconds = avg_cycles / (CPU_GHZ * 1e9); // seconds
    double gflops_per_sec = (flops_per_call / seconds) / 1e9; // GigaFLOP/s

    printf("%s,%d,%d,%d,%.0f,%" PRIu64 ",%.0f,%.3f,%.3f\n", 
           mode_name, B, C, I,
           input_size,
           (uint64_t)(avg_cycles),
           (uint64_t)flops_per_call,
           flops_per_call / avg_cycles,    // FLOP/cycle
           gflops_per_sec                  // GFLOP/sec
    );

    free_data();
}

int main(int argc, char** argv) {
    int repeat = DEFAULT_REPEAT;

    if (argc > 1) {
        repeat = atoi(argv[1]);
        if (repeat <= 0) {
            fprintf(stderr, "Invalid repeat count, using default: %d\n", DEFAULT_REPEAT);
            repeat = DEFAULT_REPEAT;
        }
    }

    srand((unsigned int)time(NULL));

    // CSV Header
    printf("mode,B,C,I,input_size,avg_cycles,flops,FLOP_per_cycle,GFLOP_per_sec\n");

    for (int B = 8; B <= 128; B *= 2) {
        for (int C = 16; C <= 128; C *= 2) {
            int I = 8; // keep blades fixed
            benchmark_mode(0, "Linear", B, C, I, repeat);
            //benchmark_mode(1, "Sum",    B, C, I, repeat);
            //benchmark_mode(2, "Mean",   B, C, I, repeat);
        }
    }

    return 0;
}
