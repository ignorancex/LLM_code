#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include "../tsc_x86.h"
#ifdef BASE
#include "../../clib/convolutional/conv_base.c"
#elif OPT1
#include "../../clib/convolutional/conv2d_opt1.c"
#endif


// CPU frequency in GHz (adjust as needed)
// #define CPU_GHZ 1.8
#define CPU_GHZ 3.6 // Max's
#define REPEAT 3

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

// Estimate FLOPs: 2*mults per MAC, plus 1 add for bias per output.
// Ignores transposes.
double conv_flop_count_base(double  dim,
                       double batches,
                       double n_blades,
                       double in_channels,
                       double out_channels,
                       double d1,
                        double d2,
                       double d3,
                       double filter_size)
{
    if (dim == 1){
        return  out_channels *  in_channels * filter_size + 
               2 *  n_blades * out_channels * in_channels *n_blades* filter_size *(d1 - filter_size + 1)*batches;
    }else if (dim == 2){
        return 12 * out_channels * in_channels * filter_size * filter_size +
               2 * n_blades * out_channels * in_channels * n_blades * filter_size * filter_size * (d1 - filter_size + 1) * (d2 - filter_size + 1)*batches;
    }else{
        return 72* out_channels * in_channels * filter_size * filter_size * filter_size +
               2 * n_blades * out_channels * in_channels * n_blades * filter_size * filter_size * filter_size * (d1 - filter_size + 1) * (d2 - filter_size + 1) * (d3 - filter_size + 1)*batches;
    }
}

double conv_flop_count_opt1(double dim,
                       double batches,
                       double n_blades,
                       double in_channels,
                       double out_channels,
                       double d1,
                       double d2,
                       double d3,
                       double filter_size)
{
   if (dim ==2) {
    return 40*out_channels * in_channels * (d1-filter_size+1) * (d2-filter_size+1) * batches * filter_size * filter_size;
   } else {
    return 0;
   }
}

void benchmark(int dim, int batches, int n_blades,
               int in_channels, int out_channels,
               int d1, int d2, int d3,
               int filter_size)
{
    int x_elems = batches * n_blades * in_channels * d1 * ((d2>0)?d2:1) * ((d3>0)?d3:1);
    int out1  = d1 - filter_size + 1;
    int out2  = (d2>0) ? d2 - filter_size + 1 : 1;
    int out3  = (d3>0) ? d3 - filter_size + 1 : 1;
    int out_elems = batches * n_blades * out_channels * out1 * out2 * out3;
    // conv_base expects 2 blocks for dim=1, 4 for dim=2, 8 for dim=3
    int n_blocks = (dim == 1 ? 2 : (dim == 2 ? 4 : 8));
    int filter_mem = 1;
    for(int i = 0; i < dim; i++) filter_mem *= filter_size;
    int weight_elems = n_blocks
                      * out_channels
                      * in_channels
                      * filter_mem;
    int bias_elems   = out_channels;

    float *x      = malloc(sizeof(float) * x_elems);
    float *output = malloc(sizeof(float) * out_elems);
    float *weight = malloc(sizeof(float) * weight_elems);
    float *bias   = malloc(sizeof(float) * bias_elems);
    float *g      = malloc(sizeof(float) * dim);
    assert(x && output && weight && bias && g);

    random_init(x, x_elems);
    random_init(weight, weight_elems);
    random_init(bias, bias_elems);
    random_g_init_general(g, dim);

    // warm-up
#ifdef BASE
    conv_base(g, batches, dim, n_blades,
              in_channels, d1, d2, d3,
              out_channels, filter_size,
              weight, bias, x, output);
#elif OPT1
    conv2d(g, batches, in_channels, d1, d2, out_channels, filter_size, weight, bias, x, output);
#endif
    

    // time REPEAT runs
    double start = start_tsc();
    for (int i = 0; i < REPEAT; i++) {
#ifdef BASE
        conv_base(g, batches, dim, n_blades,
                  in_channels, d1, d2, d3,
                  out_channels, filter_size,
                  weight, bias, x, output);
#elif OPT1
        conv2d(g, batches, in_channels, d1, d2, out_channels, filter_size, weight, bias, x, output);
#endif
    }
    double cycles = stop_tsc(start);

    double avg_cycles   = (double) cycles/REPEAT;      
    double avg_time = avg_cycles / CPU_GHZ;

#ifdef OPT1
    double flops  = conv_flop_count_opt1(dim, batches, n_blades,
                                   in_channels, out_channels,
                                   d1, d2, d3, filter_size);
#elif BASE
    double flops  = conv_flop_count_base(dim, batches, n_blades,
                                   in_channels, out_channels,
                                   d1, d2, d3, filter_size);
#endif
    double flops_per_cycle = flops / avg_cycles;

    printf(
        "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.0f,%.3f,%.3f\n",
        dim, batches, n_blades,
        in_channels, out_channels,
        d1, d2, d3, filter_size,
        x_elems,
        flops, avg_time, flops_per_cycle
    );

    fflush(stdout); 

    free(x);
    free(output);
    free(weight);
    free(bias);
    free(g);
}

int main() {
    srand((unsigned)time(NULL));

    // CSV header
    printf("dim,batches,n_blades,in_channels,out_channels,"
           "d1,d2,d3,filter_size,input_elems,"
           "flops,avg_ns,flops_per_cycle\n");

#ifdef BASE
    for (int dim = 1; dim <= 3; ++dim) {
        int n_blades = 1 << dim;
        int fsz = dim<=2?17:11;
        int ch_step = dim<=2?7:3;
        int ch_stop = dim<=2?30:9;
        int d1 = dim<=2?60:27;
        int d2 = (dim>=2)?d1:-1;
        int d3 = (dim>=3)?d1:-1;
        int batch_size = 8;
        for(int ch=ch_step; ch <= ch_stop; ch += ch_step) {
            int in_ch = ch, out_ch = ch;
            benchmark(dim, batch_size, n_blades, in_ch, out_ch, d1, d2, d3, fsz);
        }
    }
#elif OPT1
    for (int dim = 2; dim <= 2; ++dim) {
        int n_blades = 1 << dim;
        int fsz = 17;
        int ch_step = 7;
        int ch_stop = 40;
        int d1 = 60;
        int d2 = d1;
        int d3 = -1;
        int batch_size = 8;
        for(int ch=ch_step; ch <= ch_stop; ch += ch_step) {
            int in_ch = ch, out_ch = ch;
            benchmark(dim, batch_size, n_blades, in_ch, out_ch, d1, d2, d3, fsz);
        }
    }
#endif
    return 0;
}