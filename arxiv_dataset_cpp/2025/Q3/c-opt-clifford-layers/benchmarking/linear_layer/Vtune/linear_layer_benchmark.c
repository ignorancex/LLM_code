#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include "affine_forward.h"

// Assuming CPU frequency (you can adjust!)
#define CPU_GHZ 3.6
#define REPEAT 20

uint64_t time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

void random_init(float* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)rand() / (float)RAND_MAX;
    }
}

void random_g_init(float* g, int n) {
    int nrnull = 0;
    float values[] = {-1., 0., 1.};
    float values2[] = {1, -1};
    while(1) {
        for (int i = 0; i < n; i++) {
            g[i] = (float) values[rand() % 3]; // Random values in {-1, 0, 1}
        }
        int flag = 0;
        for (int i = 0; i < n; i++) {
            if(g[i] != 0.) {
                flag = 1;
                break;
            }
        }
        if(flag == 1) {
            break;
        }
    }
}

double estimate_flops_base(int dim, int batches, int n_blades, int in_channels, int out_channels) {
    // FLOPs for affine_forward_base
    double batches_flops = ((double)batches) * ((double)n_blades) * ((double)out_channels);
    double affine = 2.*((double)n_blades) * ((double)in_channels) * batches_flops + batches_flops;
    double inchanneloutchannel = ((double)in_channels) * ((double)out_channels);
    // FLOPs for matrix multiplication
    if (dim == 1) {
        return inchanneloutchannel + affine;
    } else if (dim == 2 ){
        return 12.*inchanneloutchannel + affine;
    } else if (dim == 3) {
        return 72.*inchanneloutchannel + affine;
    } else {
        fprintf(stderr, "Unsupported dimension: %d\n", dim);
        exit(EXIT_FAILURE);
    }
    
}

// Estimate total floating-point operations for affine_forward_base
double estimate_flops_opt1(int dim, int batches, int n_blades, int in_channels, int out_channels) {
    // FLOPs for affine_forward_base
    double bias = ((double)batches) * ((double)n_blades) * ((double)out_channels);
    double batch_inchanneloutchannel = ((double)batches)*((double)in_channels) * ((double)out_channels);
    // FLOPs for matrix multiplication
    if (dim == 1) {
        return 2.*4.*batch_inchanneloutchannel + bias;
    } else if (dim == 2 ){
        return (2.*16. + 8.)*batch_inchanneloutchannel + bias;
    } else if (dim == 3) {
        return (2.*64.+ 48.)*batch_inchanneloutchannel + bias;
    } else {
        fprintf(stderr, "Unsupported dimension: %d\n", dim);
        exit(EXIT_FAILURE);
    }
}

double estimate_flops_opt2(float* g, int dim, int batches, int n_blades, int in_channels, int out_channels) {
    if (dim == 1) {
        if (g[0] == -1.)
            return 2.*4 * in_channels * out_channels * batches;
        else if (g[0] == 0.)
            return 2.*3 * in_channels * out_channels * batches;
        else if (g[0] == 1.)
            return 2.*4 * in_channels * out_channels * batches;
    }
    if (dim == 2) {
        if (g[0] == -1. && g[1] == -1.)
            return 2.*16 * in_channels * out_channels * batches;
        else if (g[0] == 0. && g[1] == -1.)
            return 2.*12 * in_channels * out_channels * batches;
        else if (g[0] == 1. && g[1] == -1.)
            return 2.*16 * in_channels * out_channels * batches;
        else if (g[0] == -1. && g[1] == 0.)
            return 2.*12 * in_channels * out_channels * batches;
        else if (g[0] == 0. && g[1] == 0.)
            return 2.*9 * in_channels * out_channels * batches;
        else if (g[0] == 1. && g[1] == 0.)
            return 2.*12 * in_channels * out_channels * batches;
        else if (g[0] == -1. && g[1] == 1.)
            return 2.*16 * in_channels * out_channels * batches;
        else if (g[0] == 0. && g[1] == 1.)
            return 2.*12 * in_channels * out_channels * batches;
        else if (g[0] == 1. && g[1] == 1.)
            return 2.*16 * in_channels * out_channels * batches;
    }
    if (dim == 3) {
        if (g[0] == -1. && g[1] == -1. && g[2] == -1.)
            return 2.*64 * in_channels * out_channels * batches;
        else if (g[0] == 0. && g[1] == -1. && g[2] == -1.)
            return 2.*48 * in_channels * out_channels * batches;
        else if (g[0] == 1. && g[1] == -1. && g[2] == -1.)
            return 2.*64 * in_channels * out_channels * batches;
        else if (g[0] == -1. && g[1] == 0. && g[2] == -1.)
            return 2.*48 * in_channels * out_channels * batches;
        else if (g[0] == 0. && g[1] == 0. && g[2] == -1.)
            return 2.*36 * in_channels * out_channels * batches;
        else if (g[0] == 1. && g[1] == 0. && g[2] == -1.)
            return 2.*48 * in_channels * out_channels * batches;
        else if (g[0] == -1. && g[1] == 1. && g[2] == -1.)
            return 2.*64 * in_channels * out_channels * batches;
        else if (g[0] == 0. && g[1] == 1. && g[2] == -1.)
            return 2.*48 * in_channels * out_channels * batches;
        else if (g[0] == 1. && g[1] == 1. && g[2] == -1.)
            return 2.*64 * in_channels * out_channels * batches;
        else if (g[0] == -1. && g[1] == -1. && g[2] == 0.)
            return 2.*48 * in_channels * out_channels * batches;
        else if (g[0] == 0. && g[1] == -1. && g[2] == 0.)
            return 2.*36 * in_channels * out_channels * batches;
        else if (g[0] == 1. && g[1] == -1. && g[2] == 0.)
            return 2.*48 * in_channels * out_channels * batches;
        else if (g[0] == -1. && g[1] == 0. && g[2] == 0.)
            return 2.*36 * in_channels * out_channels * batches;
        else if (g[0] == 0. && g[1] == 0. && g[2] == 0.)
            return 2.*27 * in_channels * out_channels * batches;
        else if (g[0] == 1. && g[1] == 0. && g[2] == 0.)
            return 2.*36 * in_channels * out_channels * batches;
        else if (g[0] == -1. && g[1] == 1. && g[2] == 0.)
            return 2.*48 * in_channels * out_channels * batches;
        else if (g[0] == 0. && g[1] == 1. && g[2] == 0.)
            return 2.*36 * in_channels * out_channels * batches;
        else if (g[0] == 1. && g[1] == 1. && g[2] == 0.)
            return 2.*48 * in_channels * out_channels * batches;
        else if (g[0] == -1. && g[1] == -1. && g[2] == 1.)
            return 2.*64 * in_channels * out_channels * batches;
        else if (g[0] == 0. && g[1] == -1. && g[2] == 1.)
            return 2.*48 * in_channels * out_channels * batches;
        else if (g[0] == 1. && g[1] == -1. && g[2] == 1.)
            return 2.*64 * in_channels * out_channels * batches;
        else if (g[0] == -1. && g[1] == 0. && g[2] == 1.)
            return 2.*48 * in_channels * out_channels * batches;
        else if (g[0] == 0. && g[1] == 0. && g[2] == 1.)
            return 2.*36 * in_channels * out_channels * batches;
        else if (g[0] == 1. && g[1] == 0. && g[2] == 1.)
            return 2.*48 * in_channels * out_channels * batches;
        else if (g[0] == -1. && g[1] == 1. && g[2] == 1.)
            return 2.*64 * in_channels * out_channels * batches;
        else if (g[0] == 0. && g[1] == 1. && g[2] == 1.)
            return 2.*48 * in_channels * out_channels * batches;
        else if (g[0] == 1. && g[1] == 1. && g[2] == 1.)
            return 2.*64 * in_channels * out_channels * batches;
    }
    fprintf(stderr, "Unsupported dimension: %d\n", dim);
    exit(EXIT_FAILURE);
}

void benchmark(int dim, int batches, int n_blades, int in_channels, int out_channels) {
    int input_size = batches * in_channels * n_blades;
    int output_size = batches * out_channels * n_blades;
    int weight_size = n_blades * out_channels * in_channels;
    int bias_size = n_blades*out_channels;

    float* input = malloc(sizeof(float) * input_size);
    if (!input) { perror("malloc input"); exit(EXIT_FAILURE); }
    float* output = malloc(sizeof(float) * output_size);
    if (!output) { perror("malloc input"); exit(EXIT_FAILURE); }
    float* weights = malloc(sizeof(float) * weight_size);
    if (!weights) { perror("malloc input"); exit(EXIT_FAILURE); }
    float* bias = malloc(sizeof(float) * bias_size);
    if (!bias) { perror("malloc input"); exit(EXIT_FAILURE); }
    float* g = malloc(sizeof(float) * dim);
    if (!g) { perror("malloc input"); exit(EXIT_FAILURE); }

    random_init(input, input_size);
    random_init(weights, weight_size);
    random_init(bias, bias_size);
    random_g_init(g, dim);

    // Warmup
    for (int i = 0; i < 5; i++) {
        affine_forward(g, dim, n_blades, in_channels, out_channels, weights, bias, input, batches, output);
    }
    

    double start = time_ns();
    for (int r = 0; r < REPEAT; r++) {
        affine_forward(g, dim, n_blades, in_channels, out_channels, weights, bias, input, batches, output);
    }
    double end = time_ns();

    double elapsed_ns = (end - start) / REPEAT;
    double elapsed_sec = elapsed_ns / 1e9;
    double elapsed_cycles = elapsed_sec * CPU_GHZ * 1e9; // GHz * seconds -> cycles

    double flops = estimate_flops_base(g, dim, batches, n_blades, in_channels, out_channels);
    double flops_per_cycle = flops / elapsed_cycles;

    printf("%d,%d,%d,%d,%d,%d,%.3f,%.0f,%.0f,%.3f\n",
           dim, batches, n_blades, in_channels, out_channels, input_size,
           elapsed_ns,
           elapsed_cycles,
           flops,
           flops_per_cycle);

    free(input);
    free(output);
    free(weights);
    free(bias);
    free(g);
}

int main() {
    srand((unsigned int)time(NULL));

    // CSV header
    printf("dim,batches,n_blades,in_channels,out_channels,input_size, elapsed_ns,avg_cycles,flops,FLOP_per_cycle\n");

    for (int dim = 1; dim <= 3; dim++) {
        for (int batches = 1; batches <= 1024; batches *= 2) {
            double channels = 32;
            //for (int channels = 2; channels <= 1000; channels *= 2) {
                benchmark(dim, batches, pow(2, dim), channels, channels);
            //}
        }
    }
    return 0;
}
