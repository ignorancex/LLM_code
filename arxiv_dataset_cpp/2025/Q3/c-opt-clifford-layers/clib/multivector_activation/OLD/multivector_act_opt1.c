// multivector_act_opt1.c
#include <math.h>
#include <stdlib.h>
#include <assert.h>


static inline float sigmoidf_opt1(float x) {
    return 1.0f / (1.0f + expf(-x));
}


/*
 * Function: multivector_act_forward_opt1
 * ---------------------------------
 * Applies an activation function to an input tensor based on a selected subset of "blades."
 *
 * Parameters:
 *   v                 - Input tensor of shape [B, channels, num_blades], row-major.
 *   conv_weights      - Optional weights for "linear" aggregation; shape: [channels, num_kernel_blades].
 *   conv_bias         - Optional bias for "linear" aggregation; shape: [channels].
 *   B                 - Batch size.
 *   channels          - Number of channels.
 *   num_blades        - Total number of blades per channel.
 *   num_kernel_blades - Number of blades used to compute the activation.
 *   kernel_indices    - Array of indices (length = num_kernel_blades) indicating which blades to use.
 *   agg_mode          - Aggregation mode: 0 = "linear", 1 = "sum", 2 = "mean".
 *   output            - Output tensor of same shape as input v.
 *
 * Notes:
 *   - For each (batch, channel) pair, an activation factor is computed using the selected blades.
 *   - If agg_mode == 0 ("linear"), a weighted sum plus bias is used.
 *   - If agg_mode == 1 ("sum") or 2 ("mean"), blades are summed or averaged.
 *   - The activation factor is passed through a sigmoid, then broadcast-multiplied across all blades
 *     in the same (batch, channel) slice.
 * 
 * Optimizations:
 * - Inline Signmoid function 
 * - move conditional logic outside of the loop
 * - precompute base index 
 */

void multivector_act_forward(
    const float* v,
    const float* conv_weights,
    const float* conv_bias,
    int B,
    int channels,
    int num_blades,
    int num_kernel_blades,
    const int* kernel_indices,
    int agg_mode,
    float* output
) {
    switch (agg_mode) {
        case 0:  // Linear aggregation
            for (int b = 0; b < B; b++) {
                for (int c = 0; c < channels; c++) {
                    int base = num_blades * (b * channels + c);
                    float act = 0.0f;
                    for (int k = 0; k < num_kernel_blades; k++) {
                        int blade = kernel_indices[k];
                        act += v[base + blade] * conv_weights[c * num_kernel_blades + k];
                    }
                    act += conv_bias[c];
                    act = sigmoidf_opt1(act);

                    for (int i = 0; i < num_blades; i++) {
                        output[base + i] = v[base + i] * act;
                    }
                }
            }
            break;

        case 1:  // Sum aggregation
            for (int b = 0; b < B; b++) {
                for (int c = 0; c < channels; c++) {
                    int base = num_blades * (b * channels + c);
                    float act = 0.0f;
                    for (int k = 0; k < num_kernel_blades; k++) {
                        int blade = kernel_indices[k];
                        act += v[base + blade];
                    }
                    act = sigmoidf_opt1(act);

                    for (int i = 0; i < num_blades; i++) {
                        output[base + i] = v[base + i] * act;
                    }
                }
            }
            break;

        case 2:  // Mean aggregation
            for (int b = 0; b < B; b++) {
                for (int c = 0; c < channels; c++) {
                    int base = num_blades * (b * channels + c);
                    float act = 0.0f;
                    for (int k = 0; k < num_kernel_blades; k++) {
                        int blade = kernel_indices[k];
                        act += v[base + blade];
                    }
                    act = sigmoidf_opt1(act / num_kernel_blades);

                    for (int i = 0; i < num_blades; i++) {
                        output[base + i] = v[base + i] * act;
                    }
                }
            }
            break;

        default:
            // Invalid mode; zero everything
            for (int b = 0; b < B; b++) {
                for (int c = 0; c < channels; c++) {
                    int base = num_blades * (b * channels + c);
                    for (int i = 0; i < num_blades; i++) {
                        output[base + i] = 0.0f;
                    }
                }
            }
            break;
    }
}