/* benchmark_matrix.csv.c  – sweeps B, C, K; dumps results to CSV & console */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <sys/time.h>

#define multivector_act_forward baseline_forward
#include "../../clib/multivector_activation/multivector_act.c"
#undef  multivector_act_forward

#define multivector_act_forward opt1_forward
#include "../../clib/multivector_activation/multivector_act_opt1.c"
#undef  multivector_act_forward

#define multivector_act_forward opt2_forward
#include "../../clib/multivector_activation/multivector_act_opt2.c"
#undef  multivector_act_forward


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

    double t0 = now_sec();
    for(int r=0;r<REPEAT;++r)
        fn(input, weights, bias, B,C,I,K, kidx, mode, output);
    double t1 = now_sec();

    double avg_time   = (t1-t0)/REPEAT;       /* seconds   */
    double avg_cycles = avg_time * FREQUENCY; /* cycles    */

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

    const char *versions[]   = {"Baseline","Opt1","Opt2"};
    mvaf_fn      funcs[]     = {baseline_forward,opt1_forward,opt2_forward};
    const char *mode_names[] = {"Linear","Sum","Mean"};

    const int B_set[] = {32,64,128,256};
    const int C_set[] = {32,64,128,256};
    const int K_set[] = {4,8};

    FILE *csv = fopen("multivector_bench.csv","w");
    fprintf(csv,"Version,Mode,B,C,K,Time_us,Cycles,FLOP_per_cycle\n");

    printf("multivector_act_forward sweep benchmark\n"
           "---------------------------------------------------------------\n");

    for(size_t bi=0; bi<sizeof(B_set)/sizeof(B_set[0]); ++bi)
    for(size_t ci=0; ci<sizeof(C_set)/sizeof(C_set[0]); ++ci)
    for(size_t ki=0; ki<sizeof(K_set)/sizeof(K_set[0]); ++ki)
    {
        int B=B_set[bi], C=C_set[ci], K=K_set[ki];

        for(int mode=0; mode<3; ++mode)
            for(int v=0; v<3; ++v)
                bench_one(csv, versions[v], funcs[v],
                          mode_names[mode], mode,
                          B,C,K);
        printf("---------------------------------------------------------------\n");
    }

    fclose(csv);
    printf("Results written to multivector_bench.csv\n");
    return 0;
}
