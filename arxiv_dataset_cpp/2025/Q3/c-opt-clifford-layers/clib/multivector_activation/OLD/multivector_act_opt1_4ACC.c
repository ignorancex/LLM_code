// multivector_act_opt1.c
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

 void multivector_act_forward(
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

    int channel_pairs = channels >> 1;
    int leftover_ch   = channels & 1;
    float inv_nk      = (agg_mode == 2) ? 1.0f / (float)num_kernel_blades : 0.0f;

    for (int b = 0; b < B; ++b) {
        const float *v_batch  = v     + (size_t)b * channels * num_blades;
        float       *o_batch  = output+ (size_t)b * channels * num_blades;
        // These pointers are advanced *only if* agg_mode == 0 inside the loop.
        // So, for agg_mode != 0, they remain pointing to the initial conv_weights/conv_bias (which are NULL).
        const float *w_batch_current_segment  = conv_weights;
        const float *bias_row_current_segment = conv_bias;

        /* ---- channel pairs, four independent accumulators each -----------*/
        for (int p = 0; p < channel_pairs; ++p) {
            const float *row0 = v_batch;
            const float *row1 = v_batch + num_blades;
            float       *out0 = o_batch;
            float       *out1 = o_batch + num_blades;

            // Declare and initialize variables that might depend on agg_mode
            const float *w0_ptr = NULL, *w1_ptr = NULL;
            float current_bias0 = 0.0f, current_bias1 = 0.0f;

            if (agg_mode == 0) {
                // Only access w_batch_current_segment and bias_row_current_segment if agg_mode == 0
                // (ensuring they are not NULL)
                w0_ptr = w_batch_current_segment;
                w1_ptr = w_batch_current_segment + num_kernel_blades;
                current_bias0 = bias_row_current_segment[0];
                current_bias1 = bias_row_current_segment[1];
            }

            float a00=0.0f,a01=0.0f,a02=0.0f,a03=0.0f;  /* channel 0 */
            float a10=0.0f,a11=0.0f,a12=0.0f,a13=0.0f;  /* channel 1 */

            int k = 0;
            for (; k + 3 < num_kernel_blades; k += 4) {
                int i0 = kernel_indices[k];
                int i1 = kernel_indices[k+1];
                int i2 = kernel_indices[k+2];
                int i3 = kernel_indices[k+3];

                if (agg_mode == 0) {
                    /* ---- channel 0 ---- */
                    a00 += row0[i0] * w0_ptr[k];
                    a01 += row0[i1] * w0_ptr[k+1];
                    a02 += row0[i2] * w0_ptr[k+2];
                    a03 += row0[i3] * w0_ptr[k+3];
                    /* ---- channel 1 ---- */
                    a10 += row1[i0] * w1_ptr[k];
                    a11 += row1[i1] * w1_ptr[k+1];
                    a12 += row1[i2] * w1_ptr[k+2];
                    a13 += row1[i3] * w1_ptr[k+3];
                } else {
                    /* SUM / MEAN */
                    a00 += row0[i0]; a01 += row0[i1]; a02 += row0[i2]; a03 += row0[i3];
                    a10 += row1[i0]; a11 += row1[i1]; a12 += row1[i2]; a13 += row1[i3];
                }
            }
            // Cleanup loop for k
            for (; k < num_kernel_blades; ++k) {
                int idx = kernel_indices[k];
                if (agg_mode == 0) {
                    a00 += row0[idx] * w0_ptr[k];
                    a10 += row1[idx] * w1_ptr[k];
                } else {
                    a00 += row0[idx];
                    a10 += row1[idx];
                }
            }

            float sum0 = a00 + a01 + a02 + a03;
            float sum1 = a10 + a11 + a12 + a13;

            float act0, act1;
            if (agg_mode == 0) {
                act0 = sigmoid(sum0 + current_bias0);
                act1 = sigmoid(sum1 + current_bias1);
            } else if (agg_mode == 1) {
                act0 = sigmoid(sum0);
                act1 = sigmoid(sum1);
            } else { // agg_mode == 2
                act0 = sigmoid(sum0 * inv_nk);
                act1 = sigmoid(sum1 * inv_nk);
            }

            scale_row_scalar(row0, out0, num_blades, act0);
            scale_row_scalar(row1, out1, num_blades, act1);

            /* advance */
            v_batch  += num_blades * 2;
            o_batch  += num_blades * 2;
            if (agg_mode == 0) {
                w_batch_current_segment  += num_kernel_blades * 2;
                bias_row_current_segment += 2;
            }
        }

        /* ---- leftover single channel -------------------------------------*/
        if (leftover_ch) {
            const float *row = v_batch;
            float       *out_ch = o_batch; // Renamed to avoid conflict with outer scope 'output'

            const float *w_r_ptr = NULL;
            float current_bias = 0.0f;

            if (agg_mode == 0) {
                w_r_ptr = w_batch_current_segment;
                current_bias = bias_row_current_segment[0];
            }

            float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;
            int k = 0;
            for (; k + 3 < num_kernel_blades; k += 4) {
                int i0 = kernel_indices[k];
                int i1 = kernel_indices[k+1];
                int i2 = kernel_indices[k+2];
                int i3 = kernel_indices[k+3];
                if (agg_mode == 0) {
                    a0 += row[i0] * w_r_ptr[k];
                    a1 += row[i1] * w_r_ptr[k+1];
                    a2 += row[i2] * w_r_ptr[k+2];
                    a3 += row[i3] * w_r_ptr[k+3];
                } else {
                    a0 += row[i0]; a1 += row[i1]; a2 += row[i2]; a3 += row[i3];
                }
            }
            // Cleanup loop for k
            for (; k < num_kernel_blades; ++k) {
                int idx = kernel_indices[k];
                if (agg_mode == 0) a0 += row[idx] * w_r_ptr[k];
                else               a0 += row[idx];
            }
            float sum = a0 + a1 + a2 + a3;
            float act;
            if (agg_mode == 0)      act = sigmoid(sum + current_bias);
            else if (agg_mode == 1) act = sigmoid(sum);
            else                    act = sigmoid(sum * inv_nk); // agg_mode == 2

            scale_row_scalar(row, out_ch, num_blades, act);
        }
    }
}