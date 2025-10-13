#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <sys/time.h>
#include <string.h>
#include "tsc_x86.h"

#define REPEAT 100
#define FREQUENCY 3.6e9

extern void multivector_act_forward_base(
    const float* input_ptr, const float* weights_ptr, const float* bias_ptr,
    int B, int C, int I, int K, const int* KIDX, int mode, float* output_ptr);

extern void multivector_act_forward_opt1(
    const float* input_ptr, const float* weights_ptr, const float* bias_ptr,
    int B, int C, int I, int K, const int* KIDX, int mode, float* output_ptr);

extern void multivector_act_forward_opt2(
    const float* input_ptr, const float* packed_input_ptr, const float* weights_ptr, const float* bias_ptr,
    int B, int C, int I, int K, int mode, float* output_ptr);

extern void multivector_act_forward_opt3(
    const float* input_ptr, const float* packed_input_ptr, const float* weights_ptr, const float* bias_ptr,
    int B, int C, int I, int K, int mode, float* output_ptr);

extern void multivector_act_forward_opt4(
    const float* input_ptr, const float* packed_input_ptr, const float* weights_ptr, const float* bias_ptr,
    int B, int C, int I, int K, int mode, float* output_ptr);

typedef void (*mvaf_fn)(
    const float*, const float*, const float*,
    int,int,int,int,const int*,int,float*);

typedef void (*mvaf_packed_fn)(
    const float*, const float*, const float*, const float*,
    int,int,int,int,int,float*);

static double now_sec(void) {
    struct timeval t; gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec*1e-6;
}

static void fill_random(float *a, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] = (float)rand()/RAND_MAX;
}

static void repack_blades(const float *v, float *v_pack,
                          int B, int C, int I, int K, const int *kidx)
{
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const float *src = v + (size_t)b*C*I + c*I;
            float       *dst = v_pack + (size_t)b*C*K + c*K;
            for (int k_loop = 0; k_loop < K; ++k_loop)
                dst[k_loop] = src[kidx[k_loop]];
        }
    }
}

static void bench_one(FILE *csv,
    const char *version,
    void *fn_raw, int use_packed,
    const char *mode_name, int mode,
    int B, int C, int K_param, int I_param)
{
    size_t NI   = (size_t)B * C * I_param;
    size_t NW   = (size_t)C * K_param;
    size_t NBIA = (size_t)C;
    size_t NP   = (size_t)B * C * K_param;

    float *input   = malloc(NI * sizeof(float));
    float *output  = malloc(NI * sizeof(float));
    float *weights = (mode==0) ? malloc(NW*sizeof(float)) : NULL;
    float *bias    = (mode==0) ? malloc(NBIA*sizeof(float)) : NULL;
    float *packed  = use_packed ? malloc(NP * sizeof(float)) : NULL;
    int   *kidx    = malloc(K_param * sizeof(int));

    for (int k_loop = 0; k_loop < K_param; ++k_loop) kidx[k_loop] = k_loop;
    fill_random(input, NI);
    if (weights) fill_random(weights, NW);
    if (bias)    fill_random(bias, NBIA);

    if (use_packed)
        repack_blades(input, packed, B, C, I_param, K_param, kidx);


    for(int i_warmup =0; i_warmup < 20; i_warmup++){
        if (use_packed) {
            ((mvaf_packed_fn)fn_raw)(input, packed, weights, bias,
                        B, C, I_param, K_param, mode, output);
        } else {
            ((mvaf_fn)fn_raw)(input, weights, bias,
                    B, C, I_param, K_param, kidx, mode, output);
        }
    }

    uint64_t start = start_tsc();
    for (int r = 0; r < REPEAT; ++r) {
        if (use_packed) {
            ((mvaf_packed_fn)fn_raw)(input, packed, weights, bias,
                            B, C, I_param, K_param, mode, output);
        } else {
            ((mvaf_fn)fn_raw)(input, weights, bias,
                        B, C, I_param, K_param, kidx, mode, output);
        }
    }
    uint64_t cycles = stop_tsc(start);

    double avg_cycles = (double)cycles / REPEAT;
    double avg_time   = avg_cycles / FREQUENCY;
    
    const double sigmoid_total_flops = 32.0;
    double flops_pair; // FLOPs for one (Batch, Channel) slice

    // if (strcmp(version, "Baseline") == 0) {
    //     double cost_per_act_recalc;
    //     if (mode == 0) { // Linear
    //         cost_per_act_recalc = 2.0 * K_param + 1.0 + sigmoid_total_flops;
    //     } else if (mode == 1) { // Sum
    //         cost_per_act_recalc = 1.0 * K_param + sigmoid_total_flops;
    //     } else { // Mean (mode == 2)
    //         cost_per_act_recalc = 1.0 * K_param + 1.0 + sigmoid_total_flops;
    //     }
    //     flops_pair = (double)I_param * (cost_per_act_recalc + 1.0);
    // } else {
    //     double blades_for_output = (double)I_param;
    //     double act_calc_cost;
    //     if (mode == 0) { // Linear
    //         act_calc_cost = 2.0 * K_param + 1.0 + sigmoid_total_flops;
    //     } else if (mode == 1) { // Sum
    //         act_calc_cost = 1.0 * K_param + sigmoid_total_flops;
    //     } else { // Mean (mode == 2)
    //         act_calc_cost = 1.0 * K_param + 1.0 + sigmoid_total_flops;
    //     }
    //     flops_pair = act_calc_cost + blades_for_output;
    // }

    double blades_for_output = (double)I_param;
    double act_calc_cost;
    if (mode == 0) { // Linear
        act_calc_cost = 2.0 * K_param + 1.0 + sigmoid_total_flops;
    } else if (mode == 1) { // Sum
        act_calc_cost = 1.0 * K_param + sigmoid_total_flops;
    } else { // Mean (mode == 2)
        act_calc_cost = 1.0 * K_param + 1.0 + sigmoid_total_flops;
    }
    flops_pair = act_calc_cost + blades_for_output;

    double flops_call  = flops_pair * (double)B * C;
    double f_per_cycle = flops_call / avg_cycles;

    printf("%-8s | %-6s | n=%d I=%-2d | B=%-3d C=%-3d K=%-2d | %8.2f µs | %9.0f cyc | %7.2f F/cyc\n",
        version, mode_name, I_param==4?2:3, I_param, B, C, K_param,
        avg_time*1e6, avg_cycles, f_per_cycle);

    fprintf(csv,
        "%s,%s,%d,%d,%d,%d,%d,%g,%g,%g\n",
        version, mode_name,
        I_param==4?2:3, I_param, B, C, K_param,
        avg_time*1e6, avg_cycles, f_per_cycle);

    free(kidx); free(input); free(output);
    if (weights) free(weights);
    if (bias)    free(bias);
    if (packed)  free(packed);
}

int main(void){
    srand((unsigned)time(NULL));
    const char *mode_names[] = {"Linear", "Sum", "Mean"};
    const struct { const char *name; void *fn; int use_packed; } versions[] = {
        { "Baseline", (void*)multivector_act_forward_base,     0 },
        { "Opt1",     (void*)multivector_act_forward_opt1,     0 },
        { "Opt2",     (void*)multivector_act_forward_opt2,     1 },
        { "Opt3",     (void*)multivector_act_forward_opt3,     1 },
        { "Opt4",     (void*)multivector_act_forward_opt4,     1 },
    };

    const int n_set[] = {2,3};
    // const int B_set[] = {8,16, 32, 64, 128, 256, 512, 1024, 2048};
    // const int C_set[] = {8,16, 32, 64, 128, 256, 512, 1024, 2048};
    const int B_set[] = {8,16, 32, 64, 128, 256, 512, 1024};
    const int C_set[] = {8,16, 32, 64, 128, 256, 512, 1024};
    // const int B_set[] = {128};
    // const int C_set[] = {8,16, 32, 64, 128, 256, 512, 1024, 2048};

    FILE *csv = fopen("multivector_bench_x86_Max.csv","w");
    fprintf(csv,"Version,Mode,n,I,B,C,K,Time_us,Cycles,FLOP_per_cycle\n");
    printf("-----------------------------------------------------------------\n");

    for (size_t ni = 0; ni < sizeof(n_set)/sizeof(n_set[0]); ++ni) {
        int n_val = n_set[ni];
        int I_val = 1 << n_val;
        const int K_val_set[] = { I_val }; 
        size_t K_cnt = sizeof(K_val_set)/sizeof(K_val_set[0]);

        for (size_t bi = 0; bi < sizeof(B_set)/sizeof(B_set[0]); ++bi)
        for (size_t ci = 0; ci < sizeof(C_set)/sizeof(C_set[0]); ++ci)
        for (size_t ki = 0; ki < K_cnt; ++ki) {
            int B_val = B_set[bi];
            int C_val = C_set[ci];
            int K_val = K_val_set[ki];

            for (int mode_val = 0; mode_val < 3; ++mode_val)
                for (size_t v = 0; v < sizeof(versions)/sizeof(versions[0]); ++v)
                    bench_one(csv,
                              versions[v].name, versions[v].fn, versions[v].use_packed,
                              mode_names[mode_val], mode_val,
                              B_val, C_val, K_val, I_val);

            printf("-----------------------------------------------------------------\n");
        }
    }

    fclose(csv);
    printf("Results written to multivector_bench_x86_Max.csv\n");
    return 0;
}