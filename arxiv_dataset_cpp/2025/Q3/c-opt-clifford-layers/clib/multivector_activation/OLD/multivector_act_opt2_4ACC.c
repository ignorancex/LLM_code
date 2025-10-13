// multivector_act_opt1_acc.c – scalar Alder‑Lake tuned reference kernels
// plus "packed" baseline that uses pre‑extracted blades.
//
// ───────────────────────────────────────────────────────────────────────────
//  Compile (GCC 14 / Clang 18):
//      gcc -O3 -march=alderlake -mfma -ffast-math -fno-math-errno \
//          -c multivector_act_opt1_acc.c // Or your specific filename
// ───────────────────────────────────────────────────────────────────────────
// 2025‑05‑21  •  Max @ ETH + ChatGPT  •  Public domain / CC0
//
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "multivector_act_common.h" // Assuming this contains sigmoid() and RESTRICT

/***************************************************************************
 *  Packed-blades implementation with two accumulators, unrolled by 2.
 ***************************************************************************/
void multivector_act_forward(
        const float *RESTRICT v_full,   /* [B][C][NB] */
        const float *RESTRICT v_pack,   /* [B][C][K]  */
        const float *RESTRICT w,        /* [C][K]     */
        const float *RESTRICT bias,     /* [C]        */
        int B, int C, int NB, int K,
        int agg_mode,                   /* 0=LINEAR,1=SUM,2=MEAN */
        float *RESTRICT out)            /* [B][C][NB] */
{
    assert((agg_mode == 0 && w && bias) || (agg_mode != 0));
    const float invK = (agg_mode == 2) ? (1.0f / (float)K) : 0.0f;

    for (int b = 0; b < B; ++b) {
        const float *v_full_b = v_full + (size_t)b * C * NB;
        const float *v_pack_b = v_pack + (size_t)b * C * K;
        float       *out_b    = out    + (size_t)b * C * NB;

        for (int c = 0; c < C; ++c) {
            const float *row_full = v_full_b + (size_t)c * NB;
            const float *row_pack = v_pack_b + (size_t)c * K;
            float       *row_out  = out_b    + (size_t)c * NB;

            float acc0 = 0.0f;
            float acc1 = 0.0f; // Second accumulator

            if (agg_mode == 0) { // LINEAR
                const float *w_row = w + (size_t)c * K;
                int k = 0;
                // Unroll by 2
                for (; k + 1 < K; k += 2) {
                    acc0 += row_pack[k    ] * w_row[k    ];
                    acc1 += row_pack[k + 1] * w_row[k + 1];
                }
                // Cleanup loop for remaining elements (if K is odd)
                for (; k < K; ++k) {
                    acc0 += row_pack[k] * w_row[k];
                }
                acc0 += acc1; // Sum the two accumulators
                acc0 += bias[c];
            } else { // SUM or MEAN
                int k = 0;
                // Unroll by 2
                for (; k + 1 < K; k += 2) {
                    acc0 += row_pack[k    ];
                    acc1 += row_pack[k + 1];
                }
                // Cleanup loop for remaining elements (if K is odd)
                for (; k < K; ++k) {
                    acc0 += row_pack[k];
                }
                acc0 += acc1; // Sum the two accumulators
                if (agg_mode == 2) { // MEAN
                    acc0 *= invK;
                }
            }

            float act_val = sigmoid(acc0); // Renamed to avoid conflict if sigmoid() is a macro

            for (int i = 0; i < NB; ++i) {
                row_out[i] = row_full[i] * act_val;
            }
        }
    }
}