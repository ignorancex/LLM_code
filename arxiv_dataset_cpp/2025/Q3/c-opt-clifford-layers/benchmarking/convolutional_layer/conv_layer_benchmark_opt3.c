#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include "../tsc_x86.h"
#include "../../clib/convolutional/conv_opt3.c"


// CPU frequency in GHz (adjust as needed)
#define CPU_GHZ 3.6 // Max's
#define REPEAT 10

void random_init(float* arr, int n) {
    for (int i = 0; i < n; i++)
        arr[i] = (float)rand() / RAND_MAX;
}

void random_g_init(float* g, int n) {
    float values[] = {-1., 0., 1.};
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

void random_g_init_general(float* g, int n) {
    float values[] = {-1., 1.};
    for (int i = 0; i < n; i++) {
        g[i] = (float) values[rand() % 2];
    }
}

double estimate_flops(int dim, float* g, int batches, int n_blades, int in_channels, int out_channels, int d1, int d2, int d3, int filter_size) {
    if (dim == 1) {
        if (g[0] == -1.)
            return 2.*4* out_channels * in_channels * (d1-filter_size+1)   * batches * filter_size ;
        else if (g[0] == 0.)
            return 2.*3* out_channels * in_channels * (d1-filter_size+1)   * batches * filter_size ;
        else if (g[0] == 1.)
            return 2.*4* out_channels * in_channels * (d1-filter_size+1)   * batches * filter_size ;
    }
    if (dim == 2) {
        if (g[0] == -1. && g[1] == -1.)
            return 2.*16* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1)  * batches * filter_size * filter_size;
        else if (g[0] == 0. && g[1] == -1.)
            return 2.*12* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1)  * batches * filter_size * filter_size;
        else if (g[0] == 1. && g[1] == -1.)
            return 2.*16* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1)  * batches * filter_size * filter_size;
        else if (g[0] == -1. && g[1] == 0.)
            return 2.*12* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1)  * batches * filter_size * filter_size;
        else if (g[0] == 0. && g[1] == 0.)
            return 2.*9* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1)  * batches * filter_size * filter_size;
        else if (g[0] == 1. && g[1] == 0.)
            return 2.*12* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1)  * batches * filter_size * filter_size;
        else if (g[0] == -1. && g[1] == 1.)
            return 2.*16* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1)  * batches * filter_size * filter_size;
        else if (g[0] == 0. && g[1] == 1.)
            return 2.*12* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1)  * batches * filter_size * filter_size;
        else if (g[0] == 1. && g[1] == 1.)
            return 2.*16* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1)  * batches * filter_size * filter_size;
    }
    if (dim == 3) {
        if (g[0] == -1. && g[1] == -1. && g[2] == -1.)
            return 2.*64* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == 0. && g[1] == -1. && g[2] == -1.)
            return 2.*48* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == 1. && g[1] == -1. && g[2] == -1.)
            return 2.*64* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == -1. && g[1] == 0. && g[2] == -1.)
            return 2.*48* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == 0. && g[1] == 0. && g[2] == -1.)
            return 2.*36* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == 1. && g[1] == 0. && g[2] == -1.)
            return 2.*48* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == -1. && g[1] == 1. && g[2] == -1.)
            return 2.*64* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == 0. && g[1] == 1. && g[2] == -1.)
            return 2.*48* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == 1. && g[1] == 1. && g[2] == -1.)
            return 2.*64* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == -1. && g[1] == -1. && g[2] == 0.)
            return 2.*48* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == 0. && g[1] == -1. && g[2] == 0.)
            return 2.*36* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == 1. && g[1] == -1. && g[2] == 0.)
            return 2.*48* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == -1. && g[1] == 0. && g[2] == 0.)
            return 2.*36* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == 0. && g[1] == 0. && g[2] == 0.)
            return 2.*27* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == 1. && g[1] == 0. && g[2] == 0.)
            return 2.*36* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == -1. && g[1] == 1. && g[2] == 0.)
            return 2.*48* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == 0. && g[1] == 1. && g[2] == 0.)
            return 2.*36* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == 1. && g[1] == 1. && g[2] == 0.)
            return 2.*48* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == -1. && g[1] == -1. && g[2] == 1.)
            return 2.*64* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == 0. && g[1] == -1. && g[2] == 1.)
            return 2.*48* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == 1. && g[1] == -1. && g[2] == 1.)
            return 2.*64* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == -1. && g[1] == 0. && g[2] == 1.)
            return 2.*48* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == 0. && g[1] == 0. && g[2] == 1.)
            return 2.*36* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == 1. && g[1] == 0. && g[2] == 1.)
            return 2.*48* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == -1. && g[1] == 1. && g[2] == 1.)
            return 2.*64* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == 0. && g[1] == 1. && g[2] == 1.)
            return 2.*48* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
        else if (g[0] == 1. && g[1] == 1. && g[2] == 1.)
            return 2.*64* out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * (d3-filter_size+1) * batches * filter_size * filter_size* filter_size;
    }
    fprintf(stderr, "Unsupported dimension: %d\n", dim);
    exit(EXIT_FAILURE);
}

void benchmark(int dim, int batches, int n_blades, int in_channels, int out_channels, int d1, int d2, int d3, int filter_size) {
    int input_elems = batches * n_blades * in_channels * d1 * ((d2>0)?d2:1) * ((d3>0)?d3:1);
    int out1  = d1 - filter_size + 1;
    int out2  = (d2>0) ? d2 - filter_size + 1 : 1;
    int out3  = (d3>0) ? d3 - filter_size + 1 : 1;
    int out_elems = batches * n_blades * out_channels * out1 * out2 * out3;
    int filter_mem = 1;
    for(int i = 0; i < dim; i++) filter_mem *= filter_size;
    int weight_elems = n_blades * out_channels * in_channels * filter_mem;
    int bias_elems   = out_channels;

    float *input  = malloc(sizeof(float) * input_elems);
    float *output = malloc(sizeof(float) * out_elems);
    float *weight = malloc(sizeof(float) * weight_elems);
    float *bias   = malloc(sizeof(float) * bias_elems);
    float *g      = malloc(sizeof(float) * dim);
    assert(input);
    assert(output);
    assert(weight);
    assert(bias);
    assert(g);

    random_init(input, input_elems);
    random_init(weight, weight_elems);
    random_init(bias, bias_elems);
    random_g_init_general(g, dim);

    for(int i=0;i<3;++i) {
        if(dim == 1) {
            conv1d(g, batches, in_channels, d1, out_channels, filter_size, weight, bias, input, output);
        } else if(dim == 2) {
            conv2d(g, batches, in_channels, d1, d2, out_channels, filter_size, weight, bias, input, output);
        } else if(dim == 3) {
            conv3d(g, batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
        }
    }

    double start = start_tsc();
    if(dim == 1) {
        for (int i = 0; i < REPEAT; i++) {
            conv1d(g, batches, in_channels, d1, out_channels, filter_size, weight, bias, input, output);
        }
    } else if(dim == 2) {
        for (int i = 0; i < REPEAT; i++) {
            conv2d(g, batches, in_channels, d1, d2, out_channels, filter_size, weight, bias, input, output);
        }
    } else if(dim == 3) {
        for (int i = 0; i < REPEAT; i++) {
            conv3d(g, batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
        }
    }
    double cycles = stop_tsc(start);

    double avg_cycles   = (double) cycles/REPEAT;      
    double avg_time = avg_cycles / CPU_GHZ;

    // estimate_flops(int dim, int batches, int n_blades, int in_channels, int out_channels, int d1, int d2, int d3, int filter_size) {
    double flops  = estimate_flops(dim, g, batches, n_blades, in_channels, out_channels, d1, d2, d3, filter_size);
    double flops_per_cycle = flops / avg_cycles;

    printf(
        "%d,%d,%d,%d,%d,%d,%d,%d,%d,%.0f,%.3f,%.3f\n",
        dim, batches, n_blades,
        in_channels, out_channels,
        d1, d2, d3, filter_size,
        flops, avg_time, flops_per_cycle
    );

    fflush(stdout); 

    free(input);
    free(output);
    free(weight);
    free(bias);
    free(g);
}

int main() {
    srand((unsigned)time(NULL));

    // CSV header
    printf("dim,batches,n_blades,in_channels,out_channels,"
           "d1,d2,d3,filter_size,"
           "flops,avg_ns,flops_per_cycle\n");
    int dim = DIM;
    int n_blades = 1 << dim;
    int fsz = 17;
    int ch_step = 7;
    int ch_stop = 60;
    int d1 = 60;
    int d2 = -1;
    int d3 = -1;
    int batch_size = 8*UNROLL;
    for(int ch=ch_step; ch <= ch_stop; ch += ch_step) {
        int in_ch = ch, out_ch = ch;
        benchmark(dim, batch_size, n_blades, in_ch, out_ch, d1, d2, d3, fsz);
    }
    return 0;
}
