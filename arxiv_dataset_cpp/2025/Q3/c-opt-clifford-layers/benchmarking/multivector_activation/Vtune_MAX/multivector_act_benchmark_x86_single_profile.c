// multivector_act_benchmark_x86_single_profile.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <sys/time.h>
#include <string.h>
#include "../tsc_x86.h"

#define REPEAT 2000
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

extern void multivector_act_forward_opt5(
    const float* input_ptr, const float* packed_input_ptr, const float* weights_ptr, const float* bias_ptr,
    int B, int C, int I, int K, int mode, float* output_ptr);

typedef void (*mvaf_fn)(
    const float*, const float*, const float*,
    int,int,int,int,const int*,int,float*);

typedef void (*mvaf_packed_fn)(
    const float*, const float*, const float*, const float*,
    int,int,int,int,int,float*);

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

static void bench_one_for_profiling( 
    const char *version,
    void *fn_raw, int use_packed,
    const char *mode_name, int mode,
    int B, int C, int K_param, int I_param)
{
    size_t NI   = (size_t)B * C * I_param;
    size_t NW   = (size_t)C * K_param;
    size_t NBIA = (size_t)C;
    size_t NP   = (size_t)B * C * K_param;

    // Allocate with alignment for Opt4/Opt5 if necessary, though malloc usually gives good alignment
    // For strictness, you could use _mm_malloc for data read by _mm_load_ps
    float *input   = (float*)malloc(NI * sizeof(float));
    float *output  = (float*)malloc(NI * sizeof(float));
    float *weights = (mode==0) ? (float*)malloc(NW*sizeof(float)) : NULL;
    float *bias    = (mode==0) ? (float*)malloc(NBIA*sizeof(float)) : NULL;
    float *packed  = use_packed ? (float*)malloc(NP * sizeof(float)) : NULL;
    int   *kidx    = (int*)malloc(K_param * sizeof(int));

    // if (!input || !output || (mode==0 && !weights) || (mode==0 && !bias) || (use_packed && !packed) || !kidx) {
    //     fprintf(stderr, "Memory allocation failed in bench_one_for_profiling\n");
    //     // Simple exit, or handle more gracefully
    //     if(input) free(input); if(output) free(output); if(weights) free(weights);
    //     if(bias) free(bias); if(packed) free(packed); if(kidx) free(kidx);
    //     return;
    // }


    for (int k_loop = 0; k_loop < K_param; ++k_loop) kidx[k_loop] = k_loop;
    fill_random(input, NI);
    if (weights) fill_random(weights, NW);
    if (bias)    fill_random(bias, NBIA);

    if (use_packed)
        repack_blades(input, packed, B, C, I_param, K_param, kidx);

    // // Warm-up (important for stable measurements, less critical if VTune does its own warm-up/filtering)
    // for(int i_warmup =0; i_warmup < 20; i_warmup++){
    //     if (use_packed) {
    //         ((mvaf_packed_fn)fn_raw)(input, packed, weights, bias,
    //                     B, C, I_param, K_param, mode, output);
    //     } else {
    //         ((mvaf_fn)fn_raw)(input, weights, bias,
    //                 B, C, I_param, K_param, kidx, mode, output);
    //     }
    // }

    // printf("Starting timed run for %s...\n", version);
    // uint64_t start = start_tsc();
    for (int r = 0; r < REPEAT; ++r) {
        if (use_packed) {
            ((mvaf_packed_fn)fn_raw)(input, packed, weights, bias,
                            B, C, I_param, K_param, mode, output);
        } else {
            ((mvaf_fn)fn_raw)(input, weights, bias,
                        B, C, I_param, K_param, kidx, mode, output);
        }
    }
    // uint64_t cycles = stop_tsc(start);
    // printf("Finished timed run. Cycles: %lu\n", cycles);

    free(kidx); free(input); free(output);
    if (weights) free(weights);
    if (bias)    free(bias);
    if (packed)  free(packed);
}



int main(int argc, char *argv[]) {
    srand((unsigned)time(NULL));
    const char *mode_names_map[] = {"Linear", "Sum", "Mean"};

    const char *arg_version_name_str = argv[1];
    const char *arg_mode_str = argv[2];

    int K_to_profile = 8;
    int B_profile = 512;
    int C_profile = 512;
    int I_profile = K_to_profile;

    const char *selected_version_display_name = NULL;
    void *func_ptr = NULL;
    int use_packed = 0;
    int selected_mode_idx = -1;

    // Determine version and function pointer
    if (strcmp(arg_version_name_str, "Base") == 0) {
        selected_version_display_name = "Baseline"; func_ptr = (void*)multivector_act_forward_base; use_packed = 0;
    } else if (strcmp(arg_version_name_str, "Opt1") == 0) {
        selected_version_display_name = "Opt1"; func_ptr = (void*)multivector_act_forward_opt1; use_packed = 0;
    } else if (strcmp(arg_version_name_str, "Opt2") == 0) {
        selected_version_display_name = "Opt2"; func_ptr = (void*)multivector_act_forward_opt2; use_packed = 1;
    } else if (strcmp(arg_version_name_str, "Opt3") == 0) {
        selected_version_display_name = "Opt3"; func_ptr = (void*)multivector_act_forward_opt3; use_packed = 1;
    } else if (strcmp(arg_version_name_str, "Opt4") == 0) {
        selected_version_display_name = "Opt4"; func_ptr = (void*)multivector_act_forward_opt4; use_packed = 1;
    } else if (strcmp(arg_version_name_str, "Opt5") == 0) {
        selected_version_display_name = "Opt5"; func_ptr = (void*)multivector_act_forward_opt5; use_packed = 1;
    } else {
        fprintf(stderr, "Error: Unknown version '%s'. Supported: Base, Opt1-5\n", arg_version_name_str);
        return 1;
    }

    // Determine mode
    if (strcmp(arg_mode_str, "Linear") == 0 || strcmp(arg_mode_str, "0") == 0) {
        selected_mode_idx = 0;
    } else if (strcmp(arg_mode_str, "Sum") == 0 || strcmp(arg_mode_str, "1") == 0) {
        selected_mode_idx = 1;
    } else if (strcmp(arg_mode_str, "Mean") == 0 || strcmp(arg_mode_str, "2") == 0) {
        selected_mode_idx = 2;
    } else {
        fprintf(stderr, "Error: Unknown mode '%s'. Use Linear, Sum, Mean or 0, 1, 2.\n", arg_mode_str);
        return 1;
    }

    // printf(">>> PROFILING SINGLE VERSION <<<\n");
    // printf("Version: %s\n", version_name);
    // printf("Mode:    %s (index %d)\n", mode_names[mode_profile], mode_profile);
    // printf("B:       %d\n", B_profile);
    // printf("C:       %d\n", C_profile);
    // printf("I (NB):  %d\n", I_profile); // For Opt2+, I_param corresponds to NB
    // printf("K:       %d\n", K_to_profile); // K_param corresponds to K
    // printf("Packed:  %s\n", use_packed ? "Yes" : "No");
    // printf("REPEAT:  %d\n", REPEAT);
    // printf("---------------------------------\n");

    bench_one_for_profiling(
        selected_version_display_name,
        func_ptr,
        use_packed,
        mode_names_map[selected_mode_idx],
        selected_mode_idx,
        B_profile, C_profile, K_to_profile, I_profile);

    printf(">>> PROFILING COMPLETED <<<\n");

    return 0;
}