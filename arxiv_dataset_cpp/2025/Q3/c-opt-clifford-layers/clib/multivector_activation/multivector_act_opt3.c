#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <immintrin.h>
#include "multivector_act_common.h"

void multivector_act_forward_opt3(
    const float * RESTRICT v_full,
    const float * RESTRICT v_pack,
    const float * RESTRICT w,
    const float * RESTRICT bias,
    int B, int C, int NB, int K,
    int agg_mode,
    float * RESTRICT out)
{
    assert((agg_mode == 0 && w && bias) || (agg_mode != 0));
    assert(K == 4 || K == 8);
    assert(NB >= 0);
    assert(C % 8 == 0);

    float presig_scalars_on_stack[8] __attribute__((aligned(32)));
    float postsig_scalars_on_stack[8] __attribute__((aligned(32)));

    __m256 bc_act_c0, bc_act_c1, bc_act_c2, bc_act_c3, bc_act_c4, bc_act_c5, bc_act_c6, bc_act_c7;

    for (int b_idx = 0; b_idx < B; ++b_idx) {
        const float * const b_v_full_base = v_full + (size_t)b_idx * C * NB;
        const float * const b_v_pack_base = v_pack + (size_t)b_idx * C * K;
        float       * const b_out_base    = out    + (size_t)b_idx * C * NB;

        const float * const w_ptr_base = w; // effectively const for this function call
        const float * const bias_ptr_base = bias; // effectively const

        for (int c_base_offset = 0; c_base_offset < C; c_base_offset += 8) {
            for (int i = 0; i < 8; ++i) {
                const int c_idx_actual = c_base_offset + i;
                const float * const pack_row_ptr = b_v_pack_base + (size_t)c_idx_actual * K;
                float current_presig_val;

                if (agg_mode == 0) {
                    const float * const w_row_ptr = w_ptr_base + (size_t)c_idx_actual * K;
                    const float c_bias_val = bias_ptr_base[c_idx_actual];
                    if (K == 8) {
                        const __m256 vp8 = _mm256_loadu_ps(pack_row_ptr);
                        const __m256 wp8 = _mm256_loadu_ps(w_row_ptr);
                        const float sum_prod = hsum_8x_float(_mm256_mul_ps(vp8, wp8));
                        current_presig_val = sum_prod + c_bias_val;
                    } else { // K == 4
                        const __m128 vp4 = _mm_loadu_ps(pack_row_ptr);
                        const __m128 wp4 = _mm_loadu_ps(w_row_ptr);
                        const float sum_prod = _mm_cvtss_f32(_mm_dp_ps(vp4, wp4, 0xF1));
                        current_presig_val = sum_prod + c_bias_val;
                    }
                } else { // SUM or MEAN
                    float sum_val;
                    if (K == 8) {
                        const __m256 vp8 = _mm256_loadu_ps(pack_row_ptr);
                        sum_val = hsum_8x_float(vp8);
                    } else { // K == 4
                        const __m128 vp4 = _mm_loadu_ps(pack_row_ptr);
                        sum_val = hsum_4x_float(vp4);
                    }
                    if (agg_mode == 2) { // MEAN
                        current_presig_val = sum_val / (float)K;
                    } else { // SUM
                        current_presig_val = sum_val;
                    }
                }
                presig_scalars_on_stack[i] = current_presig_val;
            }

            const __m256 presig_avx_vec = _mm256_load_ps(presig_scalars_on_stack);
            const __m256 postsig_avx_vec = sigmoid256_ps(presig_avx_vec);
            _mm256_store_ps(postsig_scalars_on_stack, postsig_avx_vec);

            bc_act_c0 = _mm256_set1_ps(postsig_scalars_on_stack[0]);
            bc_act_c1 = _mm256_set1_ps(postsig_scalars_on_stack[1]);
            bc_act_c2 = _mm256_set1_ps(postsig_scalars_on_stack[2]);
            bc_act_c3 = _mm256_set1_ps(postsig_scalars_on_stack[3]);
            bc_act_c4 = _mm256_set1_ps(postsig_scalars_on_stack[4]);
            bc_act_c5 = _mm256_set1_ps(postsig_scalars_on_stack[5]);
            bc_act_c6 = _mm256_set1_ps(postsig_scalars_on_stack[6]);
            bc_act_c7 = _mm256_set1_ps(postsig_scalars_on_stack[7]);

            const float * const full_row0_ptr = b_v_full_base + (size_t)(c_base_offset + 0) * NB;
            const float * const full_row1_ptr = b_v_full_base + (size_t)(c_base_offset + 1) * NB;
            const float * const full_row2_ptr = b_v_full_base + (size_t)(c_base_offset + 2) * NB;
            const float * const full_row3_ptr = b_v_full_base + (size_t)(c_base_offset + 3) * NB;
            const float * const full_row4_ptr = b_v_full_base + (size_t)(c_base_offset + 4) * NB;
            const float * const full_row5_ptr = b_v_full_base + (size_t)(c_base_offset + 5) * NB;
            const float * const full_row6_ptr = b_v_full_base + (size_t)(c_base_offset + 6) * NB;
            const float * const full_row7_ptr = b_v_full_base + (size_t)(c_base_offset + 7) * NB;

            float * const out_row0_ptr = b_out_base + (size_t)(c_base_offset + 0) * NB;
            float * const out_row1_ptr = b_out_base + (size_t)(c_base_offset + 1) * NB;
            float * const out_row2_ptr = b_out_base + (size_t)(c_base_offset + 2) * NB;
            float * const out_row3_ptr = b_out_base + (size_t)(c_base_offset + 3) * NB;
            float * const out_row4_ptr = b_out_base + (size_t)(c_base_offset + 4) * NB;
            float * const out_row5_ptr = b_out_base + (size_t)(c_base_offset + 5) * NB;
            float * const out_row6_ptr = b_out_base + (size_t)(c_base_offset + 6) * NB;
            float * const out_row7_ptr = b_out_base + (size_t)(c_base_offset + 7) * NB;

            int nb_idx = 0;
            const int NB_UNROLL_FACTOR = 2;
            const int NB_AVX_STEP_SIZE = 8 * NB_UNROLL_FACTOR;
            const int nb_main_loop_end = NB - (NB % NB_AVX_STEP_SIZE);

            for (; nb_idx < nb_main_loop_end; nb_idx += NB_AVX_STEP_SIZE) {
                const __m256 v_nb0_0 = _mm256_loadu_ps(full_row0_ptr + nb_idx);
                const __m256 v_nb1_0 = _mm256_loadu_ps(full_row1_ptr + nb_idx);
                const __m256 v_nb2_0 = _mm256_loadu_ps(full_row2_ptr + nb_idx);
                const __m256 v_nb3_0 = _mm256_loadu_ps(full_row3_ptr + nb_idx);
                const __m256 v_nb4_0 = _mm256_loadu_ps(full_row4_ptr + nb_idx);
                const __m256 v_nb5_0 = _mm256_loadu_ps(full_row5_ptr + nb_idx);
                const __m256 v_nb6_0 = _mm256_loadu_ps(full_row6_ptr + nb_idx);
                const __m256 v_nb7_0 = _mm256_loadu_ps(full_row7_ptr + nb_idx);

                _mm256_storeu_ps(out_row0_ptr + nb_idx, _mm256_mul_ps(v_nb0_0, bc_act_c0));
                _mm256_storeu_ps(out_row1_ptr + nb_idx, _mm256_mul_ps(v_nb1_0, bc_act_c1));
                _mm256_storeu_ps(out_row2_ptr + nb_idx, _mm256_mul_ps(v_nb2_0, bc_act_c2));
                _mm256_storeu_ps(out_row3_ptr + nb_idx, _mm256_mul_ps(v_nb3_0, bc_act_c3));
                _mm256_storeu_ps(out_row4_ptr + nb_idx, _mm256_mul_ps(v_nb4_0, bc_act_c4));
                _mm256_storeu_ps(out_row5_ptr + nb_idx, _mm256_mul_ps(v_nb5_0, bc_act_c5));
                _mm256_storeu_ps(out_row6_ptr + nb_idx, _mm256_mul_ps(v_nb6_0, bc_act_c6));
                _mm256_storeu_ps(out_row7_ptr + nb_idx, _mm256_mul_ps(v_nb7_0, bc_act_c7));

                const __m256 v_nb0_1 = _mm256_loadu_ps(full_row0_ptr + nb_idx + 8);
                const __m256 v_nb1_1 = _mm256_loadu_ps(full_row1_ptr + nb_idx + 8);
                const __m256 v_nb2_1 = _mm256_loadu_ps(full_row2_ptr + nb_idx + 8);
                const __m256 v_nb3_1 = _mm256_loadu_ps(full_row3_ptr + nb_idx + 8);
                const __m256 v_nb4_1 = _mm256_loadu_ps(full_row4_ptr + nb_idx + 8);
                const __m256 v_nb5_1 = _mm256_loadu_ps(full_row5_ptr + nb_idx + 8);
                const __m256 v_nb6_1 = _mm256_loadu_ps(full_row6_ptr + nb_idx + 8);
                const __m256 v_nb7_1 = _mm256_loadu_ps(full_row7_ptr + nb_idx + 8);

                _mm256_storeu_ps(out_row0_ptr + nb_idx + 8, _mm256_mul_ps(v_nb0_1, bc_act_c0));
                _mm256_storeu_ps(out_row1_ptr + nb_idx + 8, _mm256_mul_ps(v_nb1_1, bc_act_c1));
                _mm256_storeu_ps(out_row2_ptr + nb_idx + 8, _mm256_mul_ps(v_nb2_1, bc_act_c2));
                _mm256_storeu_ps(out_row3_ptr + nb_idx + 8, _mm256_mul_ps(v_nb3_1, bc_act_c3));
                _mm256_storeu_ps(out_row4_ptr + nb_idx + 8, _mm256_mul_ps(v_nb4_1, bc_act_c4));
                _mm256_storeu_ps(out_row5_ptr + nb_idx + 8, _mm256_mul_ps(v_nb5_1, bc_act_c5));
                _mm256_storeu_ps(out_row6_ptr + nb_idx + 8, _mm256_mul_ps(v_nb6_1, bc_act_c6));
                _mm256_storeu_ps(out_row7_ptr + nb_idx + 8, _mm256_mul_ps(v_nb7_1, bc_act_c7));
            }

            const int nb_single_vec_loop_end = NB - (NB % 8);
            if (nb_idx < nb_single_vec_loop_end) {
                const __m256 v_nb0 = _mm256_loadu_ps(full_row0_ptr + nb_idx);
                const __m256 v_nb1 = _mm256_loadu_ps(full_row1_ptr + nb_idx);
                const __m256 v_nb2 = _mm256_loadu_ps(full_row2_ptr + nb_idx);
                const __m256 v_nb3 = _mm256_loadu_ps(full_row3_ptr + nb_idx);
                const __m256 v_nb4 = _mm256_loadu_ps(full_row4_ptr + nb_idx);
                const __m256 v_nb5 = _mm256_loadu_ps(full_row5_ptr + nb_idx);
                const __m256 v_nb6 = _mm256_loadu_ps(full_row6_ptr + nb_idx);
                const __m256 v_nb7 = _mm256_loadu_ps(full_row7_ptr + nb_idx);

                _mm256_storeu_ps(out_row0_ptr + nb_idx, _mm256_mul_ps(v_nb0, bc_act_c0));
                _mm256_storeu_ps(out_row1_ptr + nb_idx, _mm256_mul_ps(v_nb1, bc_act_c1));
                _mm256_storeu_ps(out_row2_ptr + nb_idx, _mm256_mul_ps(v_nb2, bc_act_c2));
                _mm256_storeu_ps(out_row3_ptr + nb_idx, _mm256_mul_ps(v_nb3, bc_act_c3));
                _mm256_storeu_ps(out_row4_ptr + nb_idx, _mm256_mul_ps(v_nb4, bc_act_c4));
                _mm256_storeu_ps(out_row5_ptr + nb_idx, _mm256_mul_ps(v_nb5, bc_act_c5));
                _mm256_storeu_ps(out_row6_ptr + nb_idx, _mm256_mul_ps(v_nb6, bc_act_c6));
                _mm256_storeu_ps(out_row7_ptr + nb_idx, _mm256_mul_ps(v_nb7, bc_act_c7));
                nb_idx += 8;
            }

            const float s_act0 = postsig_scalars_on_stack[0];
            const float s_act1 = postsig_scalars_on_stack[1];
            const float s_act2 = postsig_scalars_on_stack[2];
            const float s_act3 = postsig_scalars_on_stack[3];
            const float s_act4 = postsig_scalars_on_stack[4];
            const float s_act5 = postsig_scalars_on_stack[5];
            const float s_act6 = postsig_scalars_on_stack[6];
            const float s_act7 = postsig_scalars_on_stack[7];

            for (; nb_idx < NB; ++nb_idx) {
                out_row0_ptr[nb_idx] = full_row0_ptr[nb_idx] * s_act0;
                out_row1_ptr[nb_idx] = full_row1_ptr[nb_idx] * s_act1;
                out_row2_ptr[nb_idx] = full_row2_ptr[nb_idx] * s_act2;
                out_row3_ptr[nb_idx] = full_row3_ptr[nb_idx] * s_act3;
                out_row4_ptr[nb_idx] = full_row4_ptr[nb_idx] * s_act4;
                out_row5_ptr[nb_idx] = full_row5_ptr[nb_idx] * s_act5;
                out_row6_ptr[nb_idx] = full_row6_ptr[nb_idx] * s_act6;
                out_row7_ptr[nb_idx] = full_row7_ptr[nb_idx] * s_act7;
            }
        }
    }
}