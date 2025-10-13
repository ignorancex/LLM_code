// multivector_act.c
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <stddef.h>

// --- Tunable Parameters ---
// Size of blocks for the output broadcast loop (in number of floats)
// Aim to keep a block of v and output in L1/L2 cache. Should be multiple of cache line (e.g., 16 floats).
#ifndef BLADE_BLOCK_SIZE
#define BLADE_BLOCK_SIZE 128
#endif

// How many iterations ahead to prefetch in the k-loop
#ifndef PREFETCH_K_DIST
#define PREFETCH_K_DIST 8
#endif

// --- Prefetch Intrinsic (GCC/Clang) ---
// Adjust if using a different compiler (e.g., MSVC _mm_prefetch)
#ifdef __GNUC__
#define PREFETCH(addr, rw, locality) __builtin_prefetch(addr, rw, locality)
#else
#define PREFETCH(addr, rw, locality) /* Prefetch not available */
#endif

// Prefetch for read access with low temporal locality
#define PREFETCH_READ_LOW(addr) PREFETCH(addr, 0, 0)
// Prefetch for read access with moderate temporal locality
#define PREFETCH_READ_MODERATE(addr) PREFETCH(addr, 0, 1)


// Sigmoid remains the same
static inline float sigmoidf_opt3(float x) {
    return 1.0f / (1.0f + expf(-x));
}


/*
 * Function: multivector_act_forward (opt3)
 * ---------------------------------------
 * Optimization Notes (Building on opt2):
 *  + Blade Blocking: Process output broadcast in blocks for better cache reuse of v.
 *  + Explicit Prefetching: Use __builtin_prefetch for indirect reads in k-loop
 *    and potentially sequential reads in blocked output loop.
 *  + Retains opt2 features: restrict, switch, channel pairing, k-unroll, dual accumulators, base pointers.
 *
 * Parameters: (Same as before)
 *   v                 - Input tensor [B, channels, num_blades], row-major.
 *   conv_weights      - Optional weights [channels, num_kernel_blades].
 *   conv_bias         - Optional bias [channels].
 *   B, channels, num_blades, num_kernel_blades
 *   kernel_indices    - Indices [num_kernel_blades].
 *   agg_mode          - 0="linear", 1="sum", 2="mean".
 *   output            - Output tensor [B, channels, num_blades].
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
    switch (agg_mode) {
        case 0: { // LINEAR
            int pairs = channels / 2;
            int leftover = channels & 1;
            for (int b = 0; b < B; ++b) {
                const float *v_batch_ptr = v + (size_t)b * channels * num_blades;
                float       *out_batch_ptr = output + (size_t)b * channels * num_blades;
                const float *w_base_ptr = conv_weights;
                const float *bias_base_ptr = conv_bias;

                // process channels in pairs
                for (int p = 0; p < pairs; ++p) {
                    const float *row0 = v_batch_ptr;
                    const float *row1 = v_batch_ptr + num_blades;
                    float       *out0 = out_batch_ptr;
                    float       *out1 = out_batch_ptr + num_blades;
                    const float *w0   = w_base_ptr;
                    const float *w1   = w_base_ptr + num_kernel_blades;
                    float bias0       = bias_base_ptr[0];
                    float bias1       = bias_base_ptr[1];

                    float acc00 = 0.0f, acc01 = 0.0f;
                    float acc10 = 0.0f, acc11 = 0.0f;
                    int k = 0;

                    // Unrolled aggregation loop with prefetching
                    for (; k + 3 < num_kernel_blades; k += 4) {
                         // Prefetch kernel indices for future iterations
                        if (k + PREFETCH_K_DIST < num_kernel_blades) {
                            PREFETCH_READ_LOW(&kernel_indices[k + PREFETCH_K_DIST]);
                             // Prefetch weights (sequential access, likely less critical but doesn't hurt)
                            PREFETCH_READ_LOW(&w0[k + PREFETCH_K_DIST]);
                            PREFETCH_READ_LOW(&w1[k + PREFETCH_K_DIST]);
                        }
                        if (k + 3 + PREFETCH_K_DIST < num_kernel_blades) {
                             PREFETCH_READ_LOW(&kernel_indices[k + 3 + PREFETCH_K_DIST]);
                        }


                        int i0 = kernel_indices[k],
                            i1 = kernel_indices[k+1],
                            i2 = kernel_indices[k+2],
                            i3 = kernel_indices[k+3];

                        // Prefetch v data corresponding to future kernel indices
                        // This is the most critical prefetch due to indirect access
                        if (k + PREFETCH_K_DIST < num_kernel_blades) {
                           PREFETCH_READ_LOW(&row0[kernel_indices[k + PREFETCH_K_DIST]]);
                           PREFETCH_READ_LOW(&row1[kernel_indices[k + PREFETCH_K_DIST]]);
                        }
                         if (k + 1 + PREFETCH_K_DIST < num_kernel_blades) {
                           PREFETCH_READ_LOW(&row0[kernel_indices[k + 1 + PREFETCH_K_DIST]]);
                           PREFETCH_READ_LOW(&row1[kernel_indices[k + 1 + PREFETCH_K_DIST]]);
                        }
                        // ... (add more prefetches for k+2, k+3 if PREFETCH_K_DIST is small)


                        // Actual computation
                        acc00 += row0[i0] * w0[k];
                        acc01 += row0[i2] * w0[k+2];
                        acc10 += row1[i0] * w1[k];
                        acc11 += row1[i2] * w1[k+2];

                        acc00 += row0[i1] * w0[k+1];
                        acc01 += row0[i3] * w0[k+3];
                        acc10 += row1[i1] * w1[k+1];
                        acc11 += row1[i3] * w1[k+3];
                    }
                    // Remainder loop (no unrolling, but could add prefetch)
                    for (; k < num_kernel_blades; ++k) {
                        int idx = kernel_indices[k];
                         // Optional prefetch for next iteration
                        if (k + 1 < num_kernel_blades) {
                            PREFETCH_READ_LOW(&kernel_indices[k+1]);
                            PREFETCH_READ_LOW(&w0[k+1]);
                            PREFETCH_READ_LOW(&w1[k+1]);
                            PREFETCH_READ_LOW(&row0[kernel_indices[k+1]]);
                            PREFETCH_READ_LOW(&row1[kernel_indices[k+1]]);
                        }
                        acc00 += row0[idx] * w0[k];
                        acc10 += row1[idx] * w1[k];
                    }
                    float act0 = sigmoidf_opt3(acc00 + acc01 + bias0);
                    float act1 = sigmoidf_opt3(acc10 + acc11 + bias1);

                    // Blocked broadcast loop
                    for (int i_block = 0; i_block < num_blades; i_block += BLADE_BLOCK_SIZE) {
                        int i_end = i_block + BLADE_BLOCK_SIZE;
                        if (i_end > num_blades) i_end = num_blades;

                        // Optional: Prefetch next block of v (sequential access, maybe helps L2->L1)
                        // int next_block_start = i_block + BLADE_BLOCK_SIZE;
                        // if (next_block_start < num_blades) {
                        //     PREFETCH_READ_MODERATE(&row0[next_block_start]);
                        //     PREFETCH_READ_MODERATE(&row1[next_block_start]);
                        // }
                         // Prefetching output location might sometimes help avoid write stalls
                        // if (next_block_start < num_blades) {
                        //     PREFETCH(&out0[next_block_start], 1, 1); // Prefetch for write
                        //     PREFETCH(&out1[next_block_start], 1, 1);
                        // }


                        for (int i = i_block; i < i_end; ++i) {
                            out0[i] = row0[i] * act0;
                            out1[i] = row1[i] * act1;
                        }
                    }

                    // advance pointers by two channels
                    v_batch_ptr    += 2 * num_blades;
                    out_batch_ptr  += 2 * num_blades;
                    w_base_ptr    += 2 * num_kernel_blades;
                    bias_base_ptr += 2;
                }
                // leftover single channel (apply similar logic: prefetching + blocking)
                if (leftover) {
                    const float *row   = v_batch_ptr;
                    float       *out_r = out_batch_ptr;
                    const float *w_r   = w_base_ptr;
                    float bias         = bias_base_ptr[0];

                    float acc0 = 0.0f, acc1 = 0.0f;
                    int k = 0;
                    for (; k + 3 < num_kernel_blades; k += 4) {
                         if (k + PREFETCH_K_DIST < num_kernel_blades) {
                             PREFETCH_READ_LOW(&kernel_indices[k + PREFETCH_K_DIST]);
                             PREFETCH_READ_LOW(&w_r[k + PREFETCH_K_DIST]);
                             PREFETCH_READ_LOW(&row[kernel_indices[k + PREFETCH_K_DIST]]);
                         }
                         // Add more prefetches similar to paired case if needed

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
                        if (k + 1 < num_kernel_blades) {
                            PREFETCH_READ_LOW(&kernel_indices[k+1]);
                            PREFETCH_READ_LOW(&w_r[k+1]);
                            PREFETCH_READ_LOW(&row[kernel_indices[k+1]]);
                        }
                        int idx = kernel_indices[k];
                        acc0 += row[idx] * w_r[k];
                    }
                    float act = sigmoidf_opt3(acc0 + acc1 + bias);

                    // Blocked broadcast
                    for (int i_block = 0; i_block < num_blades; i_block += BLADE_BLOCK_SIZE) {
                        int i_end = i_block + BLADE_BLOCK_SIZE;
                        if (i_end > num_blades) i_end = num_blades;
                        // Optional prefetch for next block of row
                        // int next_block_start = i_block + BLADE_BLOCK_SIZE;
                        // if (next_block_start < num_blades) {
                        //    PREFETCH_READ_MODERATE(&row[next_block_start]);
                        // }
                        for (int i = i_block; i < i_end; ++i)
                            out_r[i] = row[i] * act;
                    }
                }
            }
            break;
        }
        case 1: { // SUM - Apply similar prefetching (no weights) and blocking
            int pairs = channels / 2;
            int leftover = channels & 1;
            for (int b = 0; b < B; ++b) {
                const float *v_batch_ptr = v + (size_t)b * channels * num_blades;
                float       *out_batch_ptr = output + (size_t)b * channels * num_blades;

                for (int p = 0; p < pairs; ++p) {
                    const float *row0 = v_batch_ptr;
                    const float *row1 = v_batch_ptr + num_blades;
                    float       *out0 = out_batch_ptr;
                    float       *out1 = out_batch_ptr + num_blades;

                    float acc00 = 0.0f, acc01 = 0.0f;
                    float acc10 = 0.0f, acc11 = 0.0f;
                    int k = 0;
                    for (; k + 3 < num_kernel_blades; k += 4) {
                        if (k + PREFETCH_K_DIST < num_kernel_blades) {
                            PREFETCH_READ_LOW(&kernel_indices[k + PREFETCH_K_DIST]);
                            PREFETCH_READ_LOW(&row0[kernel_indices[k + PREFETCH_K_DIST]]);
                            PREFETCH_READ_LOW(&row1[kernel_indices[k + PREFETCH_K_DIST]]);
                        }
                         // Add more prefetches similar to linear case if needed

                        int i0 = kernel_indices[k],
                            i1 = kernel_indices[k+1],
                            i2 = kernel_indices[k+2],
                            i3 = kernel_indices[k+3];
                        acc00 += row0[i0]; acc01 += row0[i2];
                        acc10 += row1[i0]; acc11 += row1[i2];
                        acc00 += row0[i1]; acc01 += row0[i3];
                        acc10 += row1[i1]; acc11 += row1[i3];
                    }
                    for (; k < num_kernel_blades; ++k) {
                        if (k + 1 < num_kernel_blades) {
                            PREFETCH_READ_LOW(&kernel_indices[k+1]);
                            PREFETCH_READ_LOW(&row0[kernel_indices[k+1]]);
                            PREFETCH_READ_LOW(&row1[kernel_indices[k+1]]);
                        }
                        int idx = kernel_indices[k];
                        acc00 += row0[idx];
                        acc10 += row1[idx];
                    }
                    float act0 = sigmoidf_opt3(acc00 + acc01);
                    float act1 = sigmoidf_opt3(acc10 + acc11);

                    // Blocked broadcast loop (same as linear)
                    for (int i_block = 0; i_block < num_blades; i_block += BLADE_BLOCK_SIZE) {
                        int i_end = i_block + BLADE_BLOCK_SIZE;
                        if (i_end > num_blades) i_end = num_blades;
                        for (int i = i_block; i < i_end; ++i) {
                            out0[i] = row0[i] * act0;
                            out1[i] = row1[i] * act1;
                        }
                    }

                    v_batch_ptr   += 2 * num_blades;
                    out_batch_ptr += 2 * num_blades;
                }
                if (leftover) {
                     const float *row   = v_batch_ptr;
                    float       *out_r = out_batch_ptr;

                    float acc0 = 0.0f, acc1 = 0.0f; // Use two accumulators even for single for consistency
                    int k = 0;
                    for (; k + 3 < num_kernel_blades; k += 4) {
                         if (k + PREFETCH_K_DIST < num_kernel_blades) {
                             PREFETCH_READ_LOW(&kernel_indices[k + PREFETCH_K_DIST]);
                             PREFETCH_READ_LOW(&row[kernel_indices[k + PREFETCH_K_DIST]]);
                         }
                         // Add more prefetches if needed

                        int i0 = kernel_indices[k],
                            i1 = kernel_indices[k+1],
                            i2 = kernel_indices[k+2],
                            i3 = kernel_indices[k+3];
                        acc0 += row[i0]; acc1 += row[i2];
                        acc0 += row[i1]; acc1 += row[i3];
                    }
                    for (; k < num_kernel_blades; ++k) {
                         if (k + 1 < num_kernel_blades) {
                            PREFETCH_READ_LOW(&kernel_indices[k+1]);
                            PREFETCH_READ_LOW(&row[kernel_indices[k+1]]);
                        }
                        acc0 += row[kernel_indices[k]];
                    }
                    float act = sigmoidf_opt3(acc0 + acc1); // Combine accumulators

                    // Blocked broadcast (same as linear, just one channel)
                    for (int i_block = 0; i_block < num_blades; i_block += BLADE_BLOCK_SIZE) {
                        int i_end = i_block + BLADE_BLOCK_SIZE;
                        if (i_end > num_blades) i_end = num_blades;
                        for (int i = i_block; i < i_end; ++i)
                            out_r[i] = row[i] * act;
                    }
                }
            }
            break;
        }
        case 2: { // MEAN - Apply similar prefetching and blocking, add scaling factor
            int pairs = channels / 2;
            int leftover = channels & 1;
            float inv_nk = 1.0f / (float)num_kernel_blades; // Loop invariant
            for (int b = 0; b < B; ++b) {
                const float *v_batch_ptr = v + (size_t)b * channels * num_blades;
                float       *out_batch_ptr = output + (size_t)b * channels * num_blades;

                for (int p = 0; p < pairs; ++p) {
                    const float *row0 = v_batch_ptr;
                    const float *row1 = v_batch_ptr + num_blades;
                    float       *out0 = out_batch_ptr;
                    float       *out1 = out_batch_ptr + num_blades;

                    float acc00 = 0.0f, acc01 = 0.0f;
                    float acc10 = 0.0f, acc11 = 0.0f;
                    int k = 0;
                    // Aggregation loop with prefetch (same as SUM mode)
                    for (; k + 3 < num_kernel_blades; k += 4) {
                         if (k + PREFETCH_K_DIST < num_kernel_blades) {
                            PREFETCH_READ_LOW(&kernel_indices[k + PREFETCH_K_DIST]);
                            PREFETCH_READ_LOW(&row0[kernel_indices[k + PREFETCH_K_DIST]]);
                            PREFETCH_READ_LOW(&row1[kernel_indices[k + PREFETCH_K_DIST]]);
                        }
                         // Add more prefetches similar to linear case if needed

                        int i0 = kernel_indices[k],
                            i1 = kernel_indices[k+1],
                            i2 = kernel_indices[k+2],
                            i3 = kernel_indices[k+3];
                        acc00 += row0[i0]; acc01 += row0[i2];
                        acc10 += row1[i0]; acc11 += row1[i2];
                        acc00 += row0[i1]; acc01 += row0[i3];
                        acc10 += row1[i1]; acc11 += row1[i3];
                    }
                     // Remainder loop with prefetch
                    for (; k < num_kernel_blades; ++k) {
                        if (k + 1 < num_kernel_blades) {
                            PREFETCH_READ_LOW(&kernel_indices[k+1]);
                            PREFETCH_READ_LOW(&row0[kernel_indices[k+1]]);
                            PREFETCH_READ_LOW(&row1[kernel_indices[k+1]]);
                        }
                        int idx = kernel_indices[k];
                        acc00 += row0[idx];
                        acc10 += row1[idx];
                    }
                    // Apply scaling factor *before* sigmoid
                    float act0 = sigmoidf_opt3((acc00 + acc01) * inv_nk);
                    float act1 = sigmoidf_opt3((acc10 + acc11) * inv_nk);

                    // Blocked broadcast loop (same as linear/sum)
                    for (int i_block = 0; i_block < num_blades; i_block += BLADE_BLOCK_SIZE) {
                        int i_end = i_block + BLADE_BLOCK_SIZE;
                        if (i_end > num_blades) i_end = num_blades;
                        for (int i = i_block; i < i_end; ++i) {
                            out0[i] = row0[i] * act0;
                            out1[i] = row1[i] * act1;
                        }
                    }

                    v_batch_ptr   += 2 * num_blades;
                    out_batch_ptr += 2 * num_blades;
                }
                 if (leftover) {
                    const float *row   = v_batch_ptr;
                    float       *out_r = out_batch_ptr;
                    float invnk_local  = inv_nk; // Local copy might help optimizer

                    float acc0 = 0.0f, acc1 = 0.0f;
                    int k = 0;
                    // Aggregation loop with prefetch (same as SUM leftover)
                    for (; k + 3 < num_kernel_blades; k += 4) {
                        if (k + PREFETCH_K_DIST < num_kernel_blades) {
                             PREFETCH_READ_LOW(&kernel_indices[k + PREFETCH_K_DIST]);
                             PREFETCH_READ_LOW(&row[kernel_indices[k + PREFETCH_K_DIST]]);
                         }
                         // Add more prefetches if needed

                        int i0 = kernel_indices[k],
                            i1 = kernel_indices[k+1],
                            i2 = kernel_indices[k+2],
                            i3 = kernel_indices[k+3];
                        acc0 += row[i0]; acc1 += row[i2];
                        acc0 += row[i1]; acc1 += row[i3];
                    }
                     // Remainder loop with prefetch
                    for (; k < num_kernel_blades; ++k) {
                         if (k + 1 < num_kernel_blades) {
                            PREFETCH_READ_LOW(&kernel_indices[k+1]);
                            PREFETCH_READ_LOW(&row[kernel_indices[k+1]]);
                        }
                        acc0 += row[kernel_indices[k]];
                    }
                    // Apply scaling factor *before* sigmoid
                    float act = sigmoidf_opt3((acc0 + acc1) * invnk_local);

                    // Blocked broadcast loop (same as SUM leftover)
                    for (int i_block = 0; i_block < num_blades; i_block += BLADE_BLOCK_SIZE) {
                        int i_end = i_block + BLADE_BLOCK_SIZE;
                        if (i_end > num_blades) i_end = num_blades;
                        for (int i = i_block; i < i_end; ++i)
                            out_r[i] = row[i] * act;
                    }
                }
            }
            break;
        }
        default: { // Zero output (unchanged)
            size_t total = (size_t)B * channels * num_blades;
            // Consider using memset if appropriate and faster on the platform
            // #include <string.h>
            // memset(output, 0, total * sizeof(float));
            for (size_t i = 0; i < total; ++i) output[i] = 0.0f;
            break;
        }
    }
}