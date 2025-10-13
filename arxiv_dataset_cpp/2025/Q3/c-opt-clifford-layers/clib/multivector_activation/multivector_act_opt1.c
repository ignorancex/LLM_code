#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "multivector_act_common.h"

/*
 * Optimizations:
 * - Inline Signmoid function 
 * - move conditional logic outside of the loop
 * - precompute base index 
 * - Manual unroll (4) + dual accumulators
 * - Loop-invariant “row” pointer
 * - Write-combine output pass (reuse row already in L1)
 */

void multivector_act_forward_opt1(
    const float * __restrict v,
    const float * __restrict conv_weights,
    const float * __restrict conv_bias,
    int B,
    int channels,
    int num_blades,
    int num_kernel_blades,
    const int * __restrict kernel_indices,
    int agg_mode,
    float * __restrict output
) {
    switch (agg_mode) {
        case 0: { // LINEAR
            int pairs = channels / 2;
            int leftover = channels & 1;
            for (int b = 0; b < B; ++b) {
                // batch-base pointers
                const float *v_ptr    = v + (size_t)b * channels * num_blades;
                float       *out_ptr  = output + (size_t)b * channels * num_blades;
                const float *w_ptr    = conv_weights;
                const float *bias_ptr = conv_bias;

                // process channels in pairs
                for (int p = 0; p < pairs; ++p) {
                    const float *row0 = v_ptr;
                    const float *row1 = v_ptr + num_blades;
                    float       *out0 = out_ptr;
                    float       *out1 = out_ptr + num_blades;

                    const float *w0   = w_ptr;
                    const float *w1   = w_ptr + num_kernel_blades;
                    float bias0       = bias_ptr[0];
                    float bias1       = bias_ptr[1];

                    float acc00 = 0.0f, acc01 = 0.0f;
                    float acc10 = 0.0f, acc11 = 0.0f;
                    int k = 0;
                    for (; k + 3 < num_kernel_blades; k += 4) {
                        int i0 = kernel_indices[k],
                            i1 = kernel_indices[k+1],
                            i2 = kernel_indices[k+2],
                            i3 = kernel_indices[k+3];
                        // channel c
                        acc00 += row0[i0] * w0[k];
                        acc01 += row0[i2] * w0[k+2];
                        acc00 += row0[i1] * w0[k+1];
                        acc01 += row0[i3] * w0[k+3];
                        // channel c+1
                        acc10 += row1[i0] * w1[k];
                        acc11 += row1[i2] * w1[k+2];
                        acc10 += row1[i1] * w1[k+1];
                        acc11 += row1[i3] * w1[k+3];
                    }
                    for (; k < num_kernel_blades; ++k) {
                        int idx = kernel_indices[k];
                        acc00 += row0[idx] * w0[k];
                        acc10 += row1[idx] * w1[k];
                    }
                    float act0 = sigmoid(acc00 + acc01 + bias0);
                    float act1 = sigmoid(acc10 + acc11 + bias1);

                    for (int i = 0; i < num_blades; ++i) {
                        out0[i] = row0[i] * act0;
                        out1[i] = row1[i] * act1;
                    }

                    // advance pointers by two channels
                    v_ptr    += 2 * num_blades;
                    out_ptr  += 2 * num_blades;
                    w_ptr    += 2 * num_kernel_blades;
                    bias_ptr += 2;
                }
                // leftover single channel
                if (leftover) {
                    const float *row   = v_ptr;
                    float       *out_r = out_ptr;
                    const float *w_r   = w_ptr;
                    float bias         = bias_ptr[0];

                    float acc0 = 0.0f, acc1 = 0.0f;
                    int k = 0;
                    for (; k + 3 < num_kernel_blades; k += 4) {
                        int i0 = kernel_indices[k],
                            i1 = kernel_indices[k+1],
                            i2 = kernel_indices[k+2],
                            i3 = kernel_indices[k+3];
                        acc0 += row[i0] * w_r[k];
                        acc1 += row[i2] * w_r[k+2];
                        acc0 += row[i1] * w_r[k+1];
                        acc1 += row[i3] * w_r[k+3];
                    }
                    for (; k < num_kernel_blades; ++k) {
                        int idx = kernel_indices[k];
                        acc0 += row[idx] * w_r[k];
                    }
                    float act = sigmoid(acc0 + acc1 + bias);
                    for (int i = 0; i < num_blades; ++i)
                        out_r[i] = row[i] * act;
                }
            }
            break;
        }
        case 1: { // SUM
            int pairs = channels / 2;
            int leftover = channels & 1;
            for (int b = 0; b < B; ++b) {
                const float *v_ptr    = v + (size_t)b * channels * num_blades;
                float       *out_ptr  = output + (size_t)b * channels * num_blades;
                const float *bias_ptr = conv_bias;

                for (int p = 0; p < pairs; ++p) {
                    const float *row0 = v_ptr;
                    const float *row1 = v_ptr + num_blades;
                    float       *out0 = out_ptr;
                    float       *out1 = out_ptr + num_blades;

                    float acc00 = 0.0f, acc01 = 0.0f;
                    float acc10 = 0.0f, acc11 = 0.0f;
                    int k = 0;
                    for (; k + 3 < num_kernel_blades; k += 4) {
                        int i0 = kernel_indices[k],
                            i1 = kernel_indices[k+1],
                            i2 = kernel_indices[k+2],
                            i3 = kernel_indices[k+3];
                        acc00 += row0[i0]; acc01 += row0[i2];
                        acc00 += row0[i1]; acc01 += row0[i3];
                        acc10 += row1[i0]; acc11 += row1[i2];
                        acc10 += row1[i1]; acc11 += row1[i3];
                    }
                    for (; k < num_kernel_blades; ++k) {
                        int idx = kernel_indices[k];
                        acc00 += row0[idx];
                        acc10 += row1[idx];
                    }
                    float act0 = sigmoid(acc00 + acc01);
                    float act1 = sigmoid(acc10 + acc11);
                    for (int i = 0; i < num_blades; ++i) {
                        out0[i] = row0[i] * act0;
                        out1[i] = row1[i] * act1;
                    }
                    v_ptr   += 2 * num_blades;
                    out_ptr += 2 * num_blades;
                    bias_ptr += 2;
                }
                if (leftover) {
                    const float *row   = v_ptr;
                    float       *out_r = out_ptr;
                    float bias         = bias_ptr[0];  // unused

                    float acc = 0.0f;
                    int k = 0;
                    for (; k + 3 < num_kernel_blades; k += 4) {
                        int i0 = kernel_indices[k],
                            i1 = kernel_indices[k+1],
                            i2 = kernel_indices[k+2],
                            i3 = kernel_indices[k+3];
                        acc += row[i0]; acc += row[i2];
                        acc += row[i1]; acc += row[i3];
                    }
                    for (; k < num_kernel_blades; ++k) {
                        acc += row[kernel_indices[k]];
                    }
                    float act = sigmoid(acc);
                    for (int i = 0; i < num_blades; ++i)
                        out_r[i] = row[i] * act;
                }
            }
            break;
        }
        case 2: { // MEAN
            int pairs = channels / 2;
            int leftover = channels & 1;
            float inv_nk = 1.0f / (float)num_kernel_blades;
            for (int b = 0; b < B; ++b) {
                const float *v_ptr    = v + (size_t)b * channels * num_blades;
                float       *out_ptr  = output + (size_t)b * channels * num_blades;

                for (int p = 0; p < pairs; ++p) {
                    const float *row0 = v_ptr;
                    const float *row1 = v_ptr + num_blades;
                    float       *out0 = out_ptr;
                    float       *out1 = out_ptr + num_blades;

                    float acc00 = 0.0f, acc01 = 0.0f;
                    float acc10 = 0.0f, acc11 = 0.0f;
                    int k = 0;
                    for (; k + 3 < num_kernel_blades; k += 4) {
                        int i0 = kernel_indices[k],
                            i1 = kernel_indices[k+1],
                            i2 = kernel_indices[k+2],
                            i3 = kernel_indices[k+3];
                        acc00 += row0[i0]; acc01 += row0[i2];
                        acc00 += row0[i1]; acc01 += row0[i3];
                        acc10 += row1[i0]; acc11 += row1[i2];
                        acc10 += row1[i1]; acc11 += row1[i3];
                    }
                    for (; k < num_kernel_blades; ++k) {
                        int idx = kernel_indices[k];
                        acc00 += row0[idx];
                        acc10 += row1[idx];
                    }
                    float act0 = sigmoid((acc00 + acc01) * inv_nk);
                    float act1 = sigmoid((acc10 + acc11) * inv_nk);
                    for (int i = 0; i < num_blades; ++i) {
                        out0[i] = row0[i] * act0;
                        out1[i] = row1[i] * act1;
                    }
                    v_ptr   += 2 * num_blades;
                    out_ptr += 2 * num_blades;
                }
                if (leftover) {
                    const float *row   = v_ptr;
                    float       *out_r = out_ptr;
                    float invnk       = inv_nk;

                    float acc = 0.0f;
                    int k = 0;
                    for (; k + 3 < num_kernel_blades; k += 4) {
                        int i0 = kernel_indices[k],
                            i1 = kernel_indices[k+1],
                            i2 = kernel_indices[k+2],
                            i3 = kernel_indices[k+3];
                        acc += row[i0]; acc += row[i2];
                        acc += row[i1]; acc += row[i3];
                    }
                    for (; k < num_kernel_blades; ++k) {
                        acc += row[kernel_indices[k]];
                    }
                    float act = sigmoid(acc * invnk);
                    for (int i = 0; i < num_blades; ++i)
                        out_r[i] = row[i] * act;
                }
            }
            break;
        }
        default: {
            size_t total = (size_t)B * channels * num_blades;
            for (size_t i = 0; i < total; ++i) output[i] = 0.0f;
            break;
        }
    }
}
