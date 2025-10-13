#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "multivector_act_common.h"

void multivector_act_forward_opt2(
        const float * RESTRICT v_full,   /* [B][C][NB] – untouched input  */
        const float * RESTRICT v_pack,   /* [B][C][K]  – extracted blades */
        const float * RESTRICT w,        /* [C][K]     – optional weights */
        const float * RESTRICT bias,     /* [C]        – optional bias    */
        int B, int C, int NB, int K,
        int agg_mode,                    /* 0=LINEAR, 1=SUM, 2=MEAN       */
        float * RESTRICT out)            /* [B][C][NB]                    */
{
    assert((agg_mode==0 && w && bias) || (agg_mode!=0));

    for (int b = 0; b < B; ++b) {
        const float *v_full_b = v_full + (size_t)b * C * NB;
        const float *v_pack_b = v_pack + (size_t)b * C * K;
        float       *out_b    = out     + (size_t)b * C * NB;

        for (int c = 0; c < C; ++c) {
            const float *row_full = v_full_b + (size_t)c * NB;
            const float *row_pack = v_pack_b + (size_t)c * K;
            float       *row_out  = out_b    + (size_t)c * NB;

            /* ---------------- activation factor ---------------------- */
            float act = 0.0f;
            if (agg_mode == 0) {                /* LINEAR */
                const float *w_row = w + (size_t)c * K;
                for (int k = 0; k < K; ++k)
                    act += row_pack[k] * w_row[k];
                act += bias[c];
            } else {                            /* SUM or MEAN */
                for (int k = 0; k < K; ++k)
                    act += row_pack[k];
                if (agg_mode == 2)              /* MEAN */
                    act /= (float)K;
            }
            act = sigmoid(act);

            /* ------------- broadcast‑scale entire blade row ---------- */
            for (int i = 0; i < NB; ++i)
                row_out[i] = row_full[i] * act;
        }
    }
}
