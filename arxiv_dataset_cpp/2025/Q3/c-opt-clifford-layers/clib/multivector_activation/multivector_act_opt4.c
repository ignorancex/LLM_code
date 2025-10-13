#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <immintrin.h>
#include "multivector_act_common.h"

void multivector_act_forward_opt4(
    const float * RESTRICT v_full_input,
    const float * RESTRICT v_pack_input,
    const float * RESTRICT w_input,
    const float * RESTRICT bias_input,
    int B, int C, int NB, int K,
    int agg_mode,
    float * RESTRICT out_output)
{
    assert((agg_mode == 0 && w_input && bias_input) || (agg_mode != 0));
    assert((K == 8 && NB == 8) || (K == 4 && NB == 4));
    assert(NB >= 0);
    assert(C % 8 == 0);

    if (K == 8 && NB == 8) {
        float presig_scalars_arr[8] __attribute__((aligned(32)));
        float postsig_scalars_arr[8] __attribute__((aligned(32)));

        if (agg_mode == 0) {
            const float * const w_base_ptr = w_input;
            const float * const bias_base_ptr = bias_input;
            for (int b_loop_idx = 0; b_loop_idx < B; ++b_loop_idx) {
                const float * const v_full_base_b = v_full_input + (size_t)b_loop_idx * C * NB;
                const float * const v_pack_base_b = v_pack_input + (size_t)b_loop_idx * C * K;
                float       * const out_base_b    = out_output   + (size_t)b_loop_idx * C * NB;

                for (int c_loop_base = 0; c_loop_base < C; c_loop_base += 8) {
                    for (int i_channel_in_group = 0; i_channel_in_group < 8; ++i_channel_in_group) {
                        const int current_c_eff_idx = c_loop_base + i_channel_in_group;
                        const float * const pack_row_for_c = v_pack_base_b + (size_t)current_c_eff_idx * K;
                        const float * const w_row_for_c = w_base_ptr + (size_t)current_c_eff_idx * K;
                        const float bias_for_c = bias_base_ptr[current_c_eff_idx];
                        const __m256 vp_k8_data = _mm256_loadu_ps(pack_row_for_c);
                        const __m256 wp_k8_data = _mm256_loadu_ps(w_row_for_c);
                        const float hsum_prod_k8 = hsum_8x_float(_mm256_mul_ps(vp_k8_data, wp_k8_data));
                        presig_scalars_arr[i_channel_in_group] = hsum_prod_k8 + bias_for_c;
                    }

                    const __m256 presig_avx_vals = _mm256_load_ps(presig_scalars_arr);
                    const __m256 postsig_avx_vals = sigmoid256_ps(presig_avx_vals);
                    _mm256_store_ps(postsig_scalars_arr, postsig_avx_vals);

                    const __m256 broadcast_act_c0 = _mm256_set1_ps(postsig_scalars_arr[0]);
                    const __m256 broadcast_act_c1 = _mm256_set1_ps(postsig_scalars_arr[1]);
                    const __m256 broadcast_act_c2 = _mm256_set1_ps(postsig_scalars_arr[2]);
                    const __m256 broadcast_act_c3 = _mm256_set1_ps(postsig_scalars_arr[3]);
                    const __m256 broadcast_act_c4 = _mm256_set1_ps(postsig_scalars_arr[4]);
                    const __m256 broadcast_act_c5 = _mm256_set1_ps(postsig_scalars_arr[5]);
                    const __m256 broadcast_act_c6 = _mm256_set1_ps(postsig_scalars_arr[6]);
                    const __m256 broadcast_act_c7 = _mm256_set1_ps(postsig_scalars_arr[7]);

                    const float * const full_row0_base = v_full_base_b + (size_t)(c_loop_base + 0) * NB;
                    const float * const full_row1_base = v_full_base_b + (size_t)(c_loop_base + 1) * NB;
                    const float * const full_row2_base = v_full_base_b + (size_t)(c_loop_base + 2) * NB;
                    const float * const full_row3_base = v_full_base_b + (size_t)(c_loop_base + 3) * NB;
                    const float * const full_row4_base = v_full_base_b + (size_t)(c_loop_base + 4) * NB;
                    const float * const full_row5_base = v_full_base_b + (size_t)(c_loop_base + 5) * NB;
                    const float * const full_row6_base = v_full_base_b + (size_t)(c_loop_base + 6) * NB;
                    const float * const full_row7_base = v_full_base_b + (size_t)(c_loop_base + 7) * NB;

                    float * const out_row0_target = out_base_b + (size_t)(c_loop_base + 0) * NB;
                    float * const out_row1_target = out_base_b + (size_t)(c_loop_base + 1) * NB;
                    float * const out_row2_target = out_base_b + (size_t)(c_loop_base + 2) * NB;
                    float * const out_row3_target = out_base_b + (size_t)(c_loop_base + 3) * NB;
                    float * const out_row4_target = out_base_b + (size_t)(c_loop_base + 4) * NB;
                    float * const out_row5_target = out_base_b + (size_t)(c_loop_base + 5) * NB;
                    float * const out_row6_target = out_base_b + (size_t)(c_loop_base + 6) * NB;
                    float * const out_row7_target = out_base_b + (size_t)(c_loop_base + 7) * NB;
                    
                    const __m256 v_nb0 = _mm256_load_ps(full_row0_base);
                    const __m256 v_nb1 = _mm256_load_ps(full_row1_base);
                    const __m256 v_nb2 = _mm256_load_ps(full_row2_base);
                    const __m256 v_nb3 = _mm256_load_ps(full_row3_base);
                    const __m256 v_nb4 = _mm256_load_ps(full_row4_base);
                    const __m256 v_nb5 = _mm256_load_ps(full_row5_base);
                    const __m256 v_nb6 = _mm256_load_ps(full_row6_base);
                    const __m256 v_nb7 = _mm256_load_ps(full_row7_base);

                    _mm256_store_ps(out_row0_target, _mm256_mul_ps(v_nb0, broadcast_act_c0));
                    _mm256_store_ps(out_row1_target, _mm256_mul_ps(v_nb1, broadcast_act_c1));
                    _mm256_store_ps(out_row2_target, _mm256_mul_ps(v_nb2, broadcast_act_c2));
                    _mm256_store_ps(out_row3_target, _mm256_mul_ps(v_nb3, broadcast_act_c3));
                    _mm256_store_ps(out_row4_target, _mm256_mul_ps(v_nb4, broadcast_act_c4));
                    _mm256_store_ps(out_row5_target, _mm256_mul_ps(v_nb5, broadcast_act_c5));
                    _mm256_store_ps(out_row6_target, _mm256_mul_ps(v_nb6, broadcast_act_c6));
                    _mm256_store_ps(out_row7_target, _mm256_mul_ps(v_nb7, broadcast_act_c7));
                }
            }
        } else if (agg_mode == 1) {
            for (int b_loop_idx = 0; b_loop_idx < B; ++b_loop_idx) {
                const float * const v_full_base_b = v_full_input + (size_t)b_loop_idx * C * NB;
                const float * const v_pack_base_b = v_pack_input + (size_t)b_loop_idx * C * K;
                float       * const out_base_b    = out_output   + (size_t)b_loop_idx * C * NB;

                for (int c_loop_base = 0; c_loop_base < C; c_loop_base += 8) {
                    for (int i_channel_in_group = 0; i_channel_in_group < 8; ++i_channel_in_group) {
                        const int current_c_eff_idx = c_loop_base + i_channel_in_group;
                        const float * const pack_row_for_c = v_pack_base_b + (size_t)current_c_eff_idx * K;
                        const __m256 vp_k8_sum_data = _mm256_loadu_ps(pack_row_for_c);
                        float hsum_val_pack_row = hsum_8x_float(vp_k8_sum_data);
                        presig_scalars_arr[i_channel_in_group] = hsum_val_pack_row;
                    }

                    const __m256 presig_avx_vals = _mm256_load_ps(presig_scalars_arr);
                    const __m256 postsig_avx_vals = sigmoid256_ps(presig_avx_vals);
                    _mm256_store_ps(postsig_scalars_arr, postsig_avx_vals);

                    const __m256 broadcast_act_c0 = _mm256_set1_ps(postsig_scalars_arr[0]);
                    const __m256 broadcast_act_c1 = _mm256_set1_ps(postsig_scalars_arr[1]);
                    const __m256 broadcast_act_c2 = _mm256_set1_ps(postsig_scalars_arr[2]);
                    const __m256 broadcast_act_c3 = _mm256_set1_ps(postsig_scalars_arr[3]);
                    const __m256 broadcast_act_c4 = _mm256_set1_ps(postsig_scalars_arr[4]);
                    const __m256 broadcast_act_c5 = _mm256_set1_ps(postsig_scalars_arr[5]);
                    const __m256 broadcast_act_c6 = _mm256_set1_ps(postsig_scalars_arr[6]);
                    const __m256 broadcast_act_c7 = _mm256_set1_ps(postsig_scalars_arr[7]);

                    const float * const full_row0_base = v_full_base_b + (size_t)(c_loop_base + 0) * NB;
                    const float * const full_row1_base = v_full_base_b + (size_t)(c_loop_base + 1) * NB;
                    const float * const full_row2_base = v_full_base_b + (size_t)(c_loop_base + 2) * NB;
                    const float * const full_row3_base = v_full_base_b + (size_t)(c_loop_base + 3) * NB;
                    const float * const full_row4_base = v_full_base_b + (size_t)(c_loop_base + 4) * NB;
                    const float * const full_row5_base = v_full_base_b + (size_t)(c_loop_base + 5) * NB;
                    const float * const full_row6_base = v_full_base_b + (size_t)(c_loop_base + 6) * NB;
                    const float * const full_row7_base = v_full_base_b + (size_t)(c_loop_base + 7) * NB;

                    float * const out_row0_target = out_base_b + (size_t)(c_loop_base + 0) * NB;
                    float * const out_row1_target = out_base_b + (size_t)(c_loop_base + 1) * NB;
                    float * const out_row2_target = out_base_b + (size_t)(c_loop_base + 2) * NB;
                    float * const out_row3_target = out_base_b + (size_t)(c_loop_base + 3) * NB;
                    float * const out_row4_target = out_base_b + (size_t)(c_loop_base + 4) * NB;
                    float * const out_row5_target = out_base_b + (size_t)(c_loop_base + 5) * NB;
                    float * const out_row6_target = out_base_b + (size_t)(c_loop_base + 6) * NB;
                    float * const out_row7_target = out_base_b + (size_t)(c_loop_base + 7) * NB;
                    
                    const __m256 v_nb0 = _mm256_load_ps(full_row0_base);
                    const __m256 v_nb1 = _mm256_load_ps(full_row1_base);
                    const __m256 v_nb2 = _mm256_load_ps(full_row2_base);
                    const __m256 v_nb3 = _mm256_load_ps(full_row3_base);
                    const __m256 v_nb4 = _mm256_load_ps(full_row4_base);
                    const __m256 v_nb5 = _mm256_load_ps(full_row5_base);
                    const __m256 v_nb6 = _mm256_load_ps(full_row6_base);
                    const __m256 v_nb7 = _mm256_load_ps(full_row7_base);

                    _mm256_store_ps(out_row0_target, _mm256_mul_ps(v_nb0, broadcast_act_c0));
                    _mm256_store_ps(out_row1_target, _mm256_mul_ps(v_nb1, broadcast_act_c1));
                    _mm256_store_ps(out_row2_target, _mm256_mul_ps(v_nb2, broadcast_act_c2));
                    _mm256_store_ps(out_row3_target, _mm256_mul_ps(v_nb3, broadcast_act_c3));
                    _mm256_store_ps(out_row4_target, _mm256_mul_ps(v_nb4, broadcast_act_c4));
                    _mm256_store_ps(out_row5_target, _mm256_mul_ps(v_nb5, broadcast_act_c5));
                    _mm256_store_ps(out_row6_target, _mm256_mul_ps(v_nb6, broadcast_act_c6));
                    _mm256_store_ps(out_row7_target, _mm256_mul_ps(v_nb7, broadcast_act_c7));
                }
            }
        } else { // agg_mode == 2
            const float inv_K_val = 1.0f / (float)K;
            for (int b_loop_idx = 0; b_loop_idx < B; ++b_loop_idx) {
                const float * const v_full_base_b = v_full_input + (size_t)b_loop_idx * C * NB;
                const float * const v_pack_base_b = v_pack_input + (size_t)b_loop_idx * C * K;
                float       * const out_base_b    = out_output   + (size_t)b_loop_idx * C * NB;

                for (int c_loop_base = 0; c_loop_base < C; c_loop_base += 8) {
                    for (int i_channel_in_group = 0; i_channel_in_group < 8; ++i_channel_in_group) {
                        const int current_c_eff_idx = c_loop_base + i_channel_in_group;
                        const float * const pack_row_for_c = v_pack_base_b + (size_t)current_c_eff_idx * K;
                        const __m256 vp_k8_sum_data = _mm256_loadu_ps(pack_row_for_c);
                        float hsum_val_pack_row = hsum_8x_float(vp_k8_sum_data);
                        presig_scalars_arr[i_channel_in_group] = hsum_val_pack_row * inv_K_val;
                    }

                    const __m256 presig_avx_vals = _mm256_load_ps(presig_scalars_arr);
                    const __m256 postsig_avx_vals = sigmoid256_ps(presig_avx_vals);
                    _mm256_store_ps(postsig_scalars_arr, postsig_avx_vals);

                    const __m256 broadcast_act_c0 = _mm256_set1_ps(postsig_scalars_arr[0]);
                    const __m256 broadcast_act_c1 = _mm256_set1_ps(postsig_scalars_arr[1]);
                    const __m256 broadcast_act_c2 = _mm256_set1_ps(postsig_scalars_arr[2]);
                    const __m256 broadcast_act_c3 = _mm256_set1_ps(postsig_scalars_arr[3]);
                    const __m256 broadcast_act_c4 = _mm256_set1_ps(postsig_scalars_arr[4]);
                    const __m256 broadcast_act_c5 = _mm256_set1_ps(postsig_scalars_arr[5]);
                    const __m256 broadcast_act_c6 = _mm256_set1_ps(postsig_scalars_arr[6]);
                    const __m256 broadcast_act_c7 = _mm256_set1_ps(postsig_scalars_arr[7]);

                    const float * const full_row0_base = v_full_base_b + (size_t)(c_loop_base + 0) * NB;
                    const float * const full_row1_base = v_full_base_b + (size_t)(c_loop_base + 1) * NB;
                    const float * const full_row2_base = v_full_base_b + (size_t)(c_loop_base + 2) * NB;
                    const float * const full_row3_base = v_full_base_b + (size_t)(c_loop_base + 3) * NB;
                    const float * const full_row4_base = v_full_base_b + (size_t)(c_loop_base + 4) * NB;
                    const float * const full_row5_base = v_full_base_b + (size_t)(c_loop_base + 5) * NB;
                    const float * const full_row6_base = v_full_base_b + (size_t)(c_loop_base + 6) * NB;
                    const float * const full_row7_base = v_full_base_b + (size_t)(c_loop_base + 7) * NB;

                    float * const out_row0_target = out_base_b + (size_t)(c_loop_base + 0) * NB;
                    float * const out_row1_target = out_base_b + (size_t)(c_loop_base + 1) * NB;
                    float * const out_row2_target = out_base_b + (size_t)(c_loop_base + 2) * NB;
                    float * const out_row3_target = out_base_b + (size_t)(c_loop_base + 3) * NB;
                    float * const out_row4_target = out_base_b + (size_t)(c_loop_base + 4) * NB;
                    float * const out_row5_target = out_base_b + (size_t)(c_loop_base + 5) * NB;
                    float * const out_row6_target = out_base_b + (size_t)(c_loop_base + 6) * NB;
                    float * const out_row7_target = out_base_b + (size_t)(c_loop_base + 7) * NB;
                    
                    const __m256 v_nb0 = _mm256_load_ps(full_row0_base);
                    const __m256 v_nb1 = _mm256_load_ps(full_row1_base);
                    const __m256 v_nb2 = _mm256_load_ps(full_row2_base);
                    const __m256 v_nb3 = _mm256_load_ps(full_row3_base);
                    const __m256 v_nb4 = _mm256_load_ps(full_row4_base);
                    const __m256 v_nb5 = _mm256_load_ps(full_row5_base);
                    const __m256 v_nb6 = _mm256_load_ps(full_row6_base);
                    const __m256 v_nb7 = _mm256_load_ps(full_row7_base);

                    _mm256_store_ps(out_row0_target, _mm256_mul_ps(v_nb0, broadcast_act_c0));
                    _mm256_store_ps(out_row1_target, _mm256_mul_ps(v_nb1, broadcast_act_c1));
                    _mm256_store_ps(out_row2_target, _mm256_mul_ps(v_nb2, broadcast_act_c2));
                    _mm256_store_ps(out_row3_target, _mm256_mul_ps(v_nb3, broadcast_act_c3));
                    _mm256_store_ps(out_row4_target, _mm256_mul_ps(v_nb4, broadcast_act_c4));
                    _mm256_store_ps(out_row5_target, _mm256_mul_ps(v_nb5, broadcast_act_c5));
                    _mm256_store_ps(out_row6_target, _mm256_mul_ps(v_nb6, broadcast_act_c6));
                    _mm256_store_ps(out_row7_target, _mm256_mul_ps(v_nb7, broadcast_act_c7));
                }
            }
        }
    }
    else { // K == 4 && NB == 4
        float presig_scalars_arr[8] __attribute__((aligned(32)));
        float postsig_scalars_arr[8] __attribute__((aligned(32)));

        if (agg_mode == 0) { // LINEAR
            const float * const w_base_ptr = w_input;
            const float * const bias_base_ptr = bias_input;
            for (int b_loop_idx = 0; b_loop_idx < B; ++b_loop_idx) {
                const float * const v_full_base_b = v_full_input + (size_t)b_loop_idx * C * NB;
                const float * const v_pack_base_b = v_pack_input + (size_t)b_loop_idx * C * K;
                float       * const out_base_b    = out_output   + (size_t)b_loop_idx * C * NB;
                
                for (int c_loop_base = 0; c_loop_base < C; c_loop_base += 8) { 
                    const __m256 vp_p01 = _mm256_loadu_ps(v_pack_base_b + (size_t)(c_loop_base + 0) * K);
                    const __m256 vp_p23 = _mm256_loadu_ps(v_pack_base_b + (size_t)(c_loop_base + 2) * K);
                    const __m256 vp_p45 = _mm256_loadu_ps(v_pack_base_b + (size_t)(c_loop_base + 4) * K);
                    const __m256 vp_p67 = _mm256_loadu_ps(v_pack_base_b + (size_t)(c_loop_base + 6) * K);

                    const __m256 wp_p01 = _mm256_loadu_ps(w_base_ptr + (size_t)(c_loop_base + 0) * K);
                    const __m256 wp_p23 = _mm256_loadu_ps(w_base_ptr + (size_t)(c_loop_base + 2) * K);
                    const __m256 wp_p45 = _mm256_loadu_ps(w_base_ptr + (size_t)(c_loop_base + 4) * K);
                    const __m256 wp_p67 = _mm256_loadu_ps(w_base_ptr + (size_t)(c_loop_base + 6) * K);

                    __m256 dp01 = _mm256_mul_ps(vp_p01, wp_p01);
                    __m256 dp23 = _mm256_mul_ps(vp_p23, wp_p23);
                    __m256 dp45 = _mm256_mul_ps(vp_p45, wp_p45);
                    __m256 dp67 = _mm256_mul_ps(vp_p67, wp_p67);
                    
                    dp01 = _mm256_hadd_ps(dp01, dp01); dp01 = _mm256_hadd_ps(dp01, dp01);
                    dp23 = _mm256_hadd_ps(dp23, dp23); dp23 = _mm256_hadd_ps(dp23, dp23);
                    dp45 = _mm256_hadd_ps(dp45, dp45); dp45 = _mm256_hadd_ps(dp45, dp45);
                    dp67 = _mm256_hadd_ps(dp67, dp67); dp67 = _mm256_hadd_ps(dp67, dp67);

                    float temp_dps[8];
                    _mm256_storeu_ps(temp_dps, dp01); presig_scalars_arr[0] = temp_dps[0] + bias_base_ptr[c_loop_base + 0]; presig_scalars_arr[1] = temp_dps[4] + bias_base_ptr[c_loop_base + 1];
                    _mm256_storeu_ps(temp_dps, dp23); presig_scalars_arr[2] = temp_dps[0] + bias_base_ptr[c_loop_base + 2]; presig_scalars_arr[3] = temp_dps[4] + bias_base_ptr[c_loop_base + 3];
                    _mm256_storeu_ps(temp_dps, dp45); presig_scalars_arr[4] = temp_dps[0] + bias_base_ptr[c_loop_base + 4]; presig_scalars_arr[5] = temp_dps[4] + bias_base_ptr[c_loop_base + 5];
                    _mm256_storeu_ps(temp_dps, dp67); presig_scalars_arr[6] = temp_dps[0] + bias_base_ptr[c_loop_base + 6]; presig_scalars_arr[7] = temp_dps[4] + bias_base_ptr[c_loop_base + 7];
                    
                    const __m256 presig_avx_vals = _mm256_load_ps(presig_scalars_arr);
                    const __m256 postsig_avx_vals = sigmoid256_ps(presig_avx_vals);
                    _mm256_store_ps(postsig_scalars_arr, postsig_avx_vals);

                    const __m256 v_nb01 = _mm256_loadu_ps(v_full_base_b + (size_t)(c_loop_base + 0) * NB);
                    const __m256 v_nb23 = _mm256_loadu_ps(v_full_base_b + (size_t)(c_loop_base + 2) * NB);
                    const __m256 v_nb45 = _mm256_loadu_ps(v_full_base_b + (size_t)(c_loop_base + 4) * NB);
                    const __m256 v_nb67 = _mm256_loadu_ps(v_full_base_b + (size_t)(c_loop_base + 6) * NB);

                    const __m256 bcast01 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[1]), _mm_set1_ps(postsig_scalars_arr[0]));
                    const __m256 bcast23 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[3]), _mm_set1_ps(postsig_scalars_arr[2]));
                    const __m256 bcast45 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[5]), _mm_set1_ps(postsig_scalars_arr[4]));
                    const __m256 bcast67 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[7]), _mm_set1_ps(postsig_scalars_arr[6]));
                    
                    _mm256_storeu_ps(out_base_b + (size_t)(c_loop_base + 0) * NB, _mm256_mul_ps(v_nb01, bcast01));
                    _mm256_storeu_ps(out_base_b + (size_t)(c_loop_base + 2) * NB, _mm256_mul_ps(v_nb23, bcast23));
                    _mm256_storeu_ps(out_base_b + (size_t)(c_loop_base + 4) * NB, _mm256_mul_ps(v_nb45, bcast45));
                    _mm256_storeu_ps(out_base_b + (size_t)(c_loop_base + 6) * NB, _mm256_mul_ps(v_nb67, bcast67));
                }
            }
        } else if (agg_mode == 1) { // SUM
            for (int b_loop_idx = 0; b_loop_idx < B; ++b_loop_idx) {
                const float * const v_full_base_b = v_full_input + (size_t)b_loop_idx * C * NB;
                const float * const v_pack_base_b = v_pack_input + (size_t)b_loop_idx * C * K;
                float       * const out_base_b    = out_output   + (size_t)b_loop_idx * C * NB;

                for (int c_loop_base = 0; c_loop_base < C; c_loop_base += 8) { 
                    const __m256 vp_p01 = _mm256_loadu_ps(v_pack_base_b + (size_t)(c_loop_base + 0) * K);
                    const __m256 vp_p23 = _mm256_loadu_ps(v_pack_base_b + (size_t)(c_loop_base + 2) * K);
                    const __m256 vp_p45 = _mm256_loadu_ps(v_pack_base_b + (size_t)(c_loop_base + 4) * K);
                    const __m256 vp_p67 = _mm256_loadu_ps(v_pack_base_b + (size_t)(c_loop_base + 6) * K);
                    
                    __m256 sum01 = _mm256_hadd_ps(vp_p01, vp_p01); sum01 = _mm256_hadd_ps(sum01, sum01);
                    __m256 sum23 = _mm256_hadd_ps(vp_p23, vp_p23); sum23 = _mm256_hadd_ps(sum23, sum23);
                    __m256 sum45 = _mm256_hadd_ps(vp_p45, vp_p45); sum45 = _mm256_hadd_ps(sum45, sum45);
                    __m256 sum67 = _mm256_hadd_ps(vp_p67, vp_p67); sum67 = _mm256_hadd_ps(sum67, sum67);

                    float temp_sums[8];
                    _mm256_storeu_ps(temp_sums, sum01); presig_scalars_arr[0] = temp_sums[0]; presig_scalars_arr[1] = temp_sums[4];
                    _mm256_storeu_ps(temp_sums, sum23); presig_scalars_arr[2] = temp_sums[0]; presig_scalars_arr[3] = temp_sums[4];
                    _mm256_storeu_ps(temp_sums, sum45); presig_scalars_arr[4] = temp_sums[0]; presig_scalars_arr[5] = temp_sums[4];
                    _mm256_storeu_ps(temp_sums, sum67); presig_scalars_arr[6] = temp_sums[0]; presig_scalars_arr[7] = temp_sums[4];

                    const __m256 presig_avx_vals = _mm256_load_ps(presig_scalars_arr);
                    const __m256 postsig_avx_vals = sigmoid256_ps(presig_avx_vals);
                    _mm256_store_ps(postsig_scalars_arr, postsig_avx_vals);

                    const __m256 v_nb01 = _mm256_loadu_ps(v_full_base_b + (size_t)(c_loop_base + 0) * NB);
                    const __m256 v_nb23 = _mm256_loadu_ps(v_full_base_b + (size_t)(c_loop_base + 2) * NB);
                    const __m256 v_nb45 = _mm256_loadu_ps(v_full_base_b + (size_t)(c_loop_base + 4) * NB);
                    const __m256 v_nb67 = _mm256_loadu_ps(v_full_base_b + (size_t)(c_loop_base + 6) * NB);

                    const __m256 bcast01 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[1]), _mm_set1_ps(postsig_scalars_arr[0]));
                    const __m256 bcast23 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[3]), _mm_set1_ps(postsig_scalars_arr[2]));
                    const __m256 bcast45 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[5]), _mm_set1_ps(postsig_scalars_arr[4]));
                    const __m256 bcast67 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[7]), _mm_set1_ps(postsig_scalars_arr[6]));
                    
                    _mm256_storeu_ps(out_base_b + (size_t)(c_loop_base + 0) * NB, _mm256_mul_ps(v_nb01, bcast01));
                    _mm256_storeu_ps(out_base_b + (size_t)(c_loop_base + 2) * NB, _mm256_mul_ps(v_nb23, bcast23));
                    _mm256_storeu_ps(out_base_b + (size_t)(c_loop_base + 4) * NB, _mm256_mul_ps(v_nb45, bcast45));
                    _mm256_storeu_ps(out_base_b + (size_t)(c_loop_base + 6) * NB, _mm256_mul_ps(v_nb67, bcast67));
                }
            }
        } else { // agg_mode == 2 (MEAN)
            const float inv_K_val = 1.0f / (float)K;
            for (int b_loop_idx = 0; b_loop_idx < B; ++b_loop_idx) {
                const float * const v_full_base_b = v_full_input + (size_t)b_loop_idx * C * NB;
                const float * const v_pack_base_b = v_pack_input + (size_t)b_loop_idx * C * K;
                float       * const out_base_b    = out_output   + (size_t)b_loop_idx * C * NB;

                for (int c_loop_base = 0; c_loop_base < C; c_loop_base += 8) { 
                    const __m256 vp_p01 = _mm256_loadu_ps(v_pack_base_b + (size_t)(c_loop_base + 0) * K);
                    const __m256 vp_p23 = _mm256_loadu_ps(v_pack_base_b + (size_t)(c_loop_base + 2) * K);
                    const __m256 vp_p45 = _mm256_loadu_ps(v_pack_base_b + (size_t)(c_loop_base + 4) * K);
                    const __m256 vp_p67 = _mm256_loadu_ps(v_pack_base_b + (size_t)(c_loop_base + 6) * K);
                    
                    __m256 sum01 = _mm256_hadd_ps(vp_p01, vp_p01); sum01 = _mm256_hadd_ps(sum01, sum01);
                    __m256 sum23 = _mm256_hadd_ps(vp_p23, vp_p23); sum23 = _mm256_hadd_ps(sum23, sum23);
                    __m256 sum45 = _mm256_hadd_ps(vp_p45, vp_p45); sum45 = _mm256_hadd_ps(sum45, sum45);
                    __m256 sum67 = _mm256_hadd_ps(vp_p67, vp_p67); sum67 = _mm256_hadd_ps(sum67, sum67);

                    float temp_sums[8];
                    _mm256_storeu_ps(temp_sums, sum01); presig_scalars_arr[0] = temp_sums[0] * inv_K_val; presig_scalars_arr[1] = temp_sums[4] * inv_K_val;
                    _mm256_storeu_ps(temp_sums, sum23); presig_scalars_arr[2] = temp_sums[0] * inv_K_val; presig_scalars_arr[3] = temp_sums[4] * inv_K_val;
                    _mm256_storeu_ps(temp_sums, sum45); presig_scalars_arr[4] = temp_sums[0] * inv_K_val; presig_scalars_arr[5] = temp_sums[4] * inv_K_val;
                    _mm256_storeu_ps(temp_sums, sum67); presig_scalars_arr[6] = temp_sums[0] * inv_K_val; presig_scalars_arr[7] = temp_sums[4] * inv_K_val;
                    
                    const __m256 presig_avx_vals = _mm256_load_ps(presig_scalars_arr);
                    const __m256 postsig_avx_vals = sigmoid256_ps(presig_avx_vals);
                    _mm256_store_ps(postsig_scalars_arr, postsig_avx_vals);

                    const __m256 v_nb01 = _mm256_loadu_ps(v_full_base_b + (size_t)(c_loop_base + 0) * NB);
                    const __m256 v_nb23 = _mm256_loadu_ps(v_full_base_b + (size_t)(c_loop_base + 2) * NB);
                    const __m256 v_nb45 = _mm256_loadu_ps(v_full_base_b + (size_t)(c_loop_base + 4) * NB);
                    const __m256 v_nb67 = _mm256_loadu_ps(v_full_base_b + (size_t)(c_loop_base + 6) * NB);

                    const __m256 bcast01 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[1]), _mm_set1_ps(postsig_scalars_arr[0]));
                    const __m256 bcast23 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[3]), _mm_set1_ps(postsig_scalars_arr[2]));
                    const __m256 bcast45 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[5]), _mm_set1_ps(postsig_scalars_arr[4]));
                    const __m256 bcast67 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[7]), _mm_set1_ps(postsig_scalars_arr[6]));
                    
                    _mm256_storeu_ps(out_base_b + (size_t)(c_loop_base + 0) * NB, _mm256_mul_ps(v_nb01, bcast01));
                    _mm256_storeu_ps(out_base_b + (size_t)(c_loop_base + 2) * NB, _mm256_mul_ps(v_nb23, bcast23));
                    _mm256_storeu_ps(out_base_b + (size_t)(c_loop_base + 4) * NB, _mm256_mul_ps(v_nb45, bcast45));
                    _mm256_storeu_ps(out_base_b + (size_t)(c_loop_base + 6) * NB, _mm256_mul_ps(v_nb67, bcast67));
                }
            }
        }
    }
}