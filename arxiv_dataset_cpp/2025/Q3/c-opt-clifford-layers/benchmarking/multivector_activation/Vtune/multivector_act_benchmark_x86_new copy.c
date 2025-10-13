/* benchmark_matrix.csv.c  – sweeps B, C, K; dumps results to CSV & console */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <sys/time.h>
#include "tsc_x86.h"

#ifndef K_VAL
#define K_VAL 8
#endif

#ifndef MODE
#define MODE 0
#endif


#if defined(VERSION_BASELINE)
    #define multivector_act_forward benchmark_forward
    #include "../../clib/multivector_activation/multivector_act.c"
#elif defined(VERSION_OPT1)
    #define multivector_act_forward benchmark_forward
    #include "../../clib/multivector_activation/multivector_act_opt1.c"
#elif defined(VERSION_OPT2)
    #define multivector_act_forward benchmark_forward
    #include "../../clib/multivector_activation/multivector_act_opt2.c"
#endif



#define I 16 // fixxed for now
#define REPEAT 500
#define FREQUENCY 1.8e9

typedef void (*mvaf_fn)(
        const float*, const float*, const float*,
        int,int,int,int,const int*,int,float*);

static double now_sec(void) {
    struct timeval t; gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec*1e-6;
}

static void fill_random(float *a, size_t n) {
    for (size_t i=0;i<n;++i) a[i] = (float)rand()/RAND_MAX;
}

static void bench_one(FILE *csv,
                      const char *version, mvaf_fn fn,
                      const char *mode_name, int mode,
                      int B,int C,int K)
{
    const size_t NI   = (size_t)B*C*I;
    const size_t NW   = (size_t)C*K;
    const size_t NBIA = (size_t)C;

    float *input   = (float*)malloc(NI   *sizeof(float));
    float *output  = (float*)malloc(NI   *sizeof(float));
    float *weights = (mode==0)? (float*)malloc(NW*sizeof(float)) : NULL;
    float *bias    = (mode==0)? (float*)malloc(NBIA*sizeof(float)) : NULL;
    int   *kidx    = (int*)malloc(K*sizeof(int));


    for(int k=0;k<K;++k) kidx[k]=k;
    fill_random(input,NI);
    if(weights) fill_random(weights,NW);
    if(bias)    fill_random(bias,NBIA);

    /* warm-up */
    fn(input, weights, bias, B,C,I,K, kidx, mode, output);

    uint64_t start = start_tsc();
    for(int r=0;r<REPEAT;++r) {
        fn(input, weights, bias, B,C,I,K, kidx, mode, output);
    }
    uint64_t cycles = stop_tsc(start);

    double avg_cycles   = (double) cycles/REPEAT;      
    double avg_time = avg_cycles / FREQUENCY;


    double blades = I;
    double flops_pair =
        (mode==0) ? (2.0*K + 1 + 32 + blades) :
        (mode==1) ? (1.0*(K-1) + 32 + blades) :
                    (1.0*(K-1) + 1 + 32 + blades);
    double flops_call   = flops_pair * (double)B * C;
    double f_per_cycle  = flops_call / avg_cycles;

    printf("%-8s | %-6s | B=%-3d C=%-3d K=%-2d | %8.2f µs | %9.0f cyc | %7.2f F/cyc\n",
           version, mode_name, B,C,K,
           avg_time*1e6, avg_cycles, f_per_cycle);

    fprintf(csv,"%s,%s,%d,%d,%d,%g,%g,%g\n",
            version,mode_name,B,C,K,avg_time*1e6,avg_cycles,f_per_cycle);

    free(kidx); free(input); free(output);
    if(weights) free(weights);
    if(bias)    free(bias);
}

int main(void)
{
    srand((unsigned)time(NULL));

    const char *version;
    #if defined(VERSION_BASELINE)
        version = "Baseline";
    #elif defined(VERSION_OPT1)
        version = "Opt1";
    #elif defined(VERSION_OPT2)
        version = "Opt2";
    #endif

    const char *mode_name;
    #if MODE == 0
    mode_name = "Linear";
    #elif MODE == 1
        mode_name = "Sum";
    #elif MODE == 2
        mode_name = "Mean";
    #endif

    const int B_set[] = {32,64,128,256};
    const int C_set[] = {32,64,128,256};

    FILE *csv = fopen("bench_result.csv", "w");
    fprintf(csv,"Version,Mode,B,C,K,Time_us,Cycles,FLOP_per_cycle\n");

    printf("Benchmark: %s | %s | K=%d\n", version, mode_name, K_VAL);
    printf("---------------------------------------------------------------\n");

    for (size_t bi = 0; bi < sizeof(B_set)/sizeof(B_set[0]); ++bi)
    for (size_t ci = 0; ci < sizeof(C_set)/sizeof(C_set[0]); ++ci) {
        int B = B_set[bi], C = C_set[ci];
        bench_one(csv, version, multivector_act_forward, mode_name, MODE, B, C, K_VAL);
    }

    fclose(csv);
    printf("Results written to bench_result.csv\n");
    return 0;
}