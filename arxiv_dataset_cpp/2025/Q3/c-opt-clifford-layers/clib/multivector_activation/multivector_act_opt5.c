#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <immintrin.h>
#include "multivector_act_common.h"

void multivector_act_forward_opt5(
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

        __m256 vp_k8_data0, vp_k8_data1, vp_k8_data2, vp_k8_data3;
        __m256 vp_k8_data4, vp_k8_data5, vp_k8_data6, vp_k8_data7;

        __m256 v_nb0, v_nb1, v_nb2, v_nb3, v_nb4, v_nb5, v_nb6, v_nb7;
        __m256 broadcast_act_c0, broadcast_act_c1, broadcast_act_c2, broadcast_act_c3;
        __m256 broadcast_act_c4, broadcast_act_c5, broadcast_act_c6, broadcast_act_c7;
        __m256 out_val0, out_val1, out_val2, out_val3, out_val4, out_val5, out_val6, out_val7;

        if (agg_mode == 0) {
            const float * const w_base_ptr = w_input;
            const float * const bias_base_ptr = bias_input;

            for (int b_loop_idx = 0; b_loop_idx < B; ++b_loop_idx) {
                const float * const v_full_base_b = v_full_input + (size_t)b_loop_idx * C * NB;
                const float * const v_pack_base_b = v_pack_input + (size_t)b_loop_idx * C * K;
                float       * const out_base_b    = out_output   + (size_t)b_loop_idx * C * NB;

                for (int c_loop_base = 0; c_loop_base < C; c_loop_base += 8) {

                    const int c_eff0 = c_loop_base + 0, c_eff1 = c_loop_base + 1, c_eff2 = c_loop_base + 2, c_eff3 = c_loop_base + 3;
                    const int c_eff4 = c_loop_base + 4, c_eff5 = c_loop_base + 5, c_eff6 = c_loop_base + 6, c_eff7 = c_loop_base + 7;

                    vp_k8_data0 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff0 * K);
                    vp_k8_data1 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff1 * K);
                    vp_k8_data2 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff2 * K);
                    vp_k8_data3 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff3 * K);

                    const __m256 wp_k8_data0 = _mm256_loadu_ps(w_base_ptr + (size_t)c_eff0 * K);
                    const __m256 wp_k8_data1 = _mm256_loadu_ps(w_base_ptr + (size_t)c_eff1 * K);
                    const __m256 wp_k8_data2 = _mm256_loadu_ps(w_base_ptr + (size_t)c_eff2 * K);
                    const __m256 wp_k8_data3 = _mm256_loadu_ps(w_base_ptr + (size_t)c_eff3 * K);

                    vp_k8_data4 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff4 * K);
                    vp_k8_data5 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff5 * K);
                    vp_k8_data6 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff6 * K);
                    vp_k8_data7 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff7 * K);

                    const __m256 wp_k8_data4 = _mm256_loadu_ps(w_base_ptr + (size_t)c_eff4 * K);
                    const __m256 wp_k8_data5 = _mm256_loadu_ps(w_base_ptr + (size_t)c_eff5 * K);
                    const __m256 wp_k8_data6 = _mm256_loadu_ps(w_base_ptr + (size_t)c_eff6 * K);
                    const __m256 wp_k8_data7 = _mm256_loadu_ps(w_base_ptr + (size_t)c_eff7 * K);

                    const __m256 prod_k8_0 = _mm256_mul_ps(vp_k8_data0, wp_k8_data0);
                    const __m256 prod_k8_1 = _mm256_mul_ps(vp_k8_data1, wp_k8_data1);
                    const __m256 prod_k8_2 = _mm256_mul_ps(vp_k8_data2, wp_k8_data2);
                    const __m256 prod_k8_3 = _mm256_mul_ps(vp_k8_data3, wp_k8_data3);

                    const __m256 prod_k8_4 = _mm256_mul_ps(vp_k8_data4, wp_k8_data4);
                    const __m256 prod_k8_5 = _mm256_mul_ps(vp_k8_data5, wp_k8_data5);
                    const __m256 prod_k8_6 = _mm256_mul_ps(vp_k8_data6, wp_k8_data6);
                    const __m256 prod_k8_7 = _mm256_mul_ps(vp_k8_data7, wp_k8_data7);

                    presig_scalars_arr[0] = hsum_8x_float(prod_k8_0) + bias_base_ptr[c_eff0];
                    presig_scalars_arr[1] = hsum_8x_float(prod_k8_1) + bias_base_ptr[c_eff1];
                    presig_scalars_arr[2] = hsum_8x_float(prod_k8_2) + bias_base_ptr[c_eff2];
                    presig_scalars_arr[3] = hsum_8x_float(prod_k8_3) + bias_base_ptr[c_eff3];
                    presig_scalars_arr[4] = hsum_8x_float(prod_k8_4) + bias_base_ptr[c_eff4];
                    presig_scalars_arr[5] = hsum_8x_float(prod_k8_5) + bias_base_ptr[c_eff5];
                    presig_scalars_arr[6] = hsum_8x_float(prod_k8_6) + bias_base_ptr[c_eff6];
                    presig_scalars_arr[7] = hsum_8x_float(prod_k8_7) + bias_base_ptr[c_eff7];

                    const __m256 presig_avx_vals = _mm256_load_ps(presig_scalars_arr);
                    const __m256 postsig_avx_vals_computed = sigmoid256_ps(presig_avx_vals);
                    _mm256_store_ps(postsig_scalars_arr, postsig_avx_vals_computed);

                    const float * full_row0_ptr = v_full_base_b + (size_t)c_eff0 * NB;
                    const float * full_row1_ptr = v_full_base_b + (size_t)c_eff1 * NB;
                    const float * full_row2_ptr = v_full_base_b + (size_t)c_eff2 * NB;
                    const float * full_row3_ptr = v_full_base_b + (size_t)c_eff3 * NB;
                    const float * full_row4_ptr = v_full_base_b + (size_t)c_eff4 * NB;
                    const float * full_row5_ptr = v_full_base_b + (size_t)c_eff5 * NB;
                    const float * full_row6_ptr = v_full_base_b + (size_t)c_eff6 * NB;
                    const float * full_row7_ptr = v_full_base_b + (size_t)c_eff7 * NB;

                    v_nb0 = _mm256_load_ps(full_row0_ptr);
                    v_nb1 = _mm256_load_ps(full_row1_ptr);
                    v_nb2 = _mm256_load_ps(full_row2_ptr);
                    v_nb3 = _mm256_load_ps(full_row3_ptr);
                    v_nb4 = _mm256_load_ps(full_row4_ptr);
                    v_nb5 = _mm256_load_ps(full_row5_ptr);
                    v_nb6 = _mm256_load_ps(full_row6_ptr);
                    v_nb7 = _mm256_load_ps(full_row7_ptr);

                    broadcast_act_c0 = _mm256_set1_ps(postsig_scalars_arr[0]);
                    broadcast_act_c1 = _mm256_set1_ps(postsig_scalars_arr[1]);
                    broadcast_act_c2 = _mm256_set1_ps(postsig_scalars_arr[2]);
                    broadcast_act_c3 = _mm256_set1_ps(postsig_scalars_arr[3]);
                    broadcast_act_c4 = _mm256_set1_ps(postsig_scalars_arr[4]);
                    broadcast_act_c5 = _mm256_set1_ps(postsig_scalars_arr[5]);
                    broadcast_act_c6 = _mm256_set1_ps(postsig_scalars_arr[6]);
                    broadcast_act_c7 = _mm256_set1_ps(postsig_scalars_arr[7]);

                    out_val0 = _mm256_mul_ps(v_nb0, broadcast_act_c0);
                    out_val1 = _mm256_mul_ps(v_nb1, broadcast_act_c1);
                    out_val2 = _mm256_mul_ps(v_nb2, broadcast_act_c2);
                    out_val3 = _mm256_mul_ps(v_nb3, broadcast_act_c3);
                    out_val4 = _mm256_mul_ps(v_nb4, broadcast_act_c4);
                    out_val5 = _mm256_mul_ps(v_nb5, broadcast_act_c5);
                    out_val6 = _mm256_mul_ps(v_nb6, broadcast_act_c6);
                    out_val7 = _mm256_mul_ps(v_nb7, broadcast_act_c7);

                    float * out_row0_target = out_base_b + (size_t)c_eff0 * NB;
                    float * out_row1_target = out_base_b + (size_t)c_eff1 * NB;
                    float * out_row2_target = out_base_b + (size_t)c_eff2 * NB;
                    float * out_row3_target = out_base_b + (size_t)c_eff3 * NB;
                    float * out_row4_target = out_base_b + (size_t)c_eff4 * NB;
                    float * out_row5_target = out_base_b + (size_t)c_eff5 * NB;
                    float * out_row6_target = out_base_b + (size_t)c_eff6 * NB;
                    float * out_row7_target = out_base_b + (size_t)c_eff7 * NB;

                    _mm256_store_ps(out_row0_target, out_val0);
                    _mm256_store_ps(out_row1_target, out_val1);
                    _mm256_store_ps(out_row2_target, out_val2);
                    _mm256_store_ps(out_row3_target, out_val3);
                    _mm256_store_ps(out_row4_target, out_val4);
                    _mm256_store_ps(out_row5_target, out_val5);
                    _mm256_store_ps(out_row6_target, out_val6);
                    _mm256_store_ps(out_row7_target, out_val7);
                }
            }
        } else if (agg_mode == 1) { 
            for (int b_loop_idx = 0; b_loop_idx < B; ++b_loop_idx) {
                const float * const v_full_base_b = v_full_input + (size_t)b_loop_idx * C * NB;
                const float * const v_pack_base_b = v_pack_input + (size_t)b_loop_idx * C * K;
                float       * const out_base_b    = out_output   + (size_t)b_loop_idx * C * NB;

                for (int c_loop_base = 0; c_loop_base < C; c_loop_base += 8) {
                    const int c_eff0 = c_loop_base + 0, c_eff1 = c_loop_base + 1, c_eff2 = c_loop_base + 2, c_eff3 = c_loop_base + 3;
                    const int c_eff4 = c_loop_base + 4, c_eff5 = c_loop_base + 5, c_eff6 = c_loop_base + 6, c_eff7 = c_loop_base + 7;

                    vp_k8_data0 = _mm256_load_ps(v_pack_base_b + (size_t)c_eff0 * K);
                    vp_k8_data1 = _mm256_load_ps(v_pack_base_b + (size_t)c_eff1 * K);
                    vp_k8_data2 = _mm256_load_ps(v_pack_base_b + (size_t)c_eff2 * K);
                    vp_k8_data3 = _mm256_load_ps(v_pack_base_b + (size_t)c_eff3 * K);
                    vp_k8_data4 = _mm256_load_ps(v_pack_base_b + (size_t)c_eff4 * K);
                    vp_k8_data5 = _mm256_load_ps(v_pack_base_b + (size_t)c_eff5 * K);
                    vp_k8_data6 = _mm256_load_ps(v_pack_base_b + (size_t)c_eff6 * K);
                    vp_k8_data7 = _mm256_load_ps(v_pack_base_b + (size_t)c_eff7 * K);

                    presig_scalars_arr[0] = hsum_8x_float(vp_k8_data0);
                    presig_scalars_arr[1] = hsum_8x_float(vp_k8_data1);
                    presig_scalars_arr[2] = hsum_8x_float(vp_k8_data2);
                    presig_scalars_arr[3] = hsum_8x_float(vp_k8_data3);
                    presig_scalars_arr[4] = hsum_8x_float(vp_k8_data4);
                    presig_scalars_arr[5] = hsum_8x_float(vp_k8_data5);
                    presig_scalars_arr[6] = hsum_8x_float(vp_k8_data6);
                    presig_scalars_arr[7] = hsum_8x_float(vp_k8_data7);

                    const __m256 presig_avx_vals = _mm256_load_ps(presig_scalars_arr);
                    const __m256 postsig_avx_vals_computed = sigmoid256_ps(presig_avx_vals);
                    _mm256_store_ps(postsig_scalars_arr, postsig_avx_vals_computed);

                    const float * full_row0_ptr = v_full_base_b + (size_t)c_eff0 * NB;
                    const float * full_row1_ptr = v_full_base_b + (size_t)c_eff1 * NB;
                    const float * full_row2_ptr = v_full_base_b + (size_t)c_eff2 * NB;
                    const float * full_row3_ptr = v_full_base_b + (size_t)c_eff3 * NB;
                    const float * full_row4_ptr = v_full_base_b + (size_t)c_eff4 * NB;
                    const float * full_row5_ptr = v_full_base_b + (size_t)c_eff5 * NB;
                    const float * full_row6_ptr = v_full_base_b + (size_t)c_eff6 * NB;
                    const float * full_row7_ptr = v_full_base_b + (size_t)c_eff7 * NB;
                    v_nb0 = _mm256_load_ps(full_row0_ptr); v_nb1 = _mm256_load_ps(full_row1_ptr);
                    v_nb2 = _mm256_load_ps(full_row2_ptr); v_nb3 = _mm256_load_ps(full_row3_ptr);
                    v_nb4 = _mm256_load_ps(full_row4_ptr); v_nb5 = _mm256_load_ps(full_row5_ptr);
                    v_nb6 = _mm256_load_ps(full_row6_ptr); v_nb7 = _mm256_load_ps(full_row7_ptr);

                    broadcast_act_c0 = _mm256_set1_ps(postsig_scalars_arr[0]); broadcast_act_c1 = _mm256_set1_ps(postsig_scalars_arr[1]);
                    broadcast_act_c2 = _mm256_set1_ps(postsig_scalars_arr[2]); broadcast_act_c3 = _mm256_set1_ps(postsig_scalars_arr[3]);
                    broadcast_act_c4 = _mm256_set1_ps(postsig_scalars_arr[4]); broadcast_act_c5 = _mm256_set1_ps(postsig_scalars_arr[5]);
                    broadcast_act_c6 = _mm256_set1_ps(postsig_scalars_arr[6]); broadcast_act_c7 = _mm256_set1_ps(postsig_scalars_arr[7]);
                    out_val0 = _mm256_mul_ps(v_nb0, broadcast_act_c0); out_val1 = _mm256_mul_ps(v_nb1, broadcast_act_c1);
                    out_val2 = _mm256_mul_ps(v_nb2, broadcast_act_c2); out_val3 = _mm256_mul_ps(v_nb3, broadcast_act_c3);
                    out_val4 = _mm256_mul_ps(v_nb4, broadcast_act_c4); out_val5 = _mm256_mul_ps(v_nb5, broadcast_act_c5);
                    out_val6 = _mm256_mul_ps(v_nb6, broadcast_act_c6); out_val7 = _mm256_mul_ps(v_nb7, broadcast_act_c7);

                    float * out_row0_target = out_base_b + (size_t)c_eff0 * NB; float * out_row1_target = out_base_b + (size_t)c_eff1 * NB;
                    float * out_row2_target = out_base_b + (size_t)c_eff2 * NB; float * out_row3_target = out_base_b + (size_t)c_eff3 * NB;
                    float * out_row4_target = out_base_b + (size_t)c_eff4 * NB; float * out_row5_target = out_base_b + (size_t)c_eff5 * NB;
                    float * out_row6_target = out_base_b + (size_t)c_eff6 * NB; float * out_row7_target = out_base_b + (size_t)c_eff7 * NB;
                    _mm256_store_ps(out_row0_target, out_val0); _mm256_store_ps(out_row1_target, out_val1);
                    _mm256_store_ps(out_row2_target, out_val2); _mm256_store_ps(out_row3_target, out_val3);
                    _mm256_store_ps(out_row4_target, out_val4); _mm256_store_ps(out_row5_target, out_val5);
                    _mm256_store_ps(out_row6_target, out_val6); _mm256_store_ps(out_row7_target, out_val7);
                }
            }
        } else { 
            const float inv_K_val = 1.0f / (float)K; 
            for (int b_loop_idx = 0; b_loop_idx < B; ++b_loop_idx) {
                const float * const v_full_base_b = v_full_input + (size_t)b_loop_idx * C * NB;
                const float * const v_pack_base_b = v_pack_input + (size_t)b_loop_idx * C * K;
                float       * const out_base_b    = out_output   + (size_t)b_loop_idx * C * NB;

                for (int c_loop_base = 0; c_loop_base < C; c_loop_base += 8) {
                    const int c_eff0 = c_loop_base + 0, c_eff1 = c_loop_base + 1, c_eff2 = c_loop_base + 2, c_eff3 = c_loop_base + 3;
                    const int c_eff4 = c_loop_base + 4, c_eff5 = c_loop_base + 5, c_eff6 = c_loop_base + 6, c_eff7 = c_loop_base + 7;

                    vp_k8_data0 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff0 * K);
                    vp_k8_data1 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff1 * K);
                    vp_k8_data2 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff2 * K);
                    vp_k8_data3 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff3 * K);
                    vp_k8_data4 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff4 * K);
                    vp_k8_data5 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff5 * K);
                    vp_k8_data6 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff6 * K);
                    vp_k8_data7 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff7 * K);

                    presig_scalars_arr[0] = hsum_8x_float(vp_k8_data0) * inv_K_val;
                    presig_scalars_arr[1] = hsum_8x_float(vp_k8_data1) * inv_K_val;
                    presig_scalars_arr[2] = hsum_8x_float(vp_k8_data2) * inv_K_val;
                    presig_scalars_arr[3] = hsum_8x_float(vp_k8_data3) * inv_K_val;
                    presig_scalars_arr[4] = hsum_8x_float(vp_k8_data4) * inv_K_val;
                    presig_scalars_arr[5] = hsum_8x_float(vp_k8_data5) * inv_K_val;
                    presig_scalars_arr[6] = hsum_8x_float(vp_k8_data6) * inv_K_val;
                    presig_scalars_arr[7] = hsum_8x_float(vp_k8_data7) * inv_K_val;

                    const __m256 presig_avx_vals = _mm256_load_ps(presig_scalars_arr);
                    const __m256 postsig_avx_vals_computed = sigmoid256_ps(presig_avx_vals);
                    _mm256_store_ps(postsig_scalars_arr, postsig_avx_vals_computed);

                    const float * full_row0_ptr = v_full_base_b + (size_t)c_eff0 * NB;
                    const float * full_row1_ptr = v_full_base_b + (size_t)c_eff1 * NB;
                    const float * full_row2_ptr = v_full_base_b + (size_t)c_eff2 * NB;
                    const float * full_row3_ptr = v_full_base_b + (size_t)c_eff3 * NB;
                    const float * full_row4_ptr = v_full_base_b + (size_t)c_eff4 * NB;
                    const float * full_row5_ptr = v_full_base_b + (size_t)c_eff5 * NB;
                    const float * full_row6_ptr = v_full_base_b + (size_t)c_eff6 * NB;
                    const float * full_row7_ptr = v_full_base_b + (size_t)c_eff7 * NB;
                    v_nb0 = _mm256_load_ps(full_row0_ptr); v_nb1 = _mm256_load_ps(full_row1_ptr);
                    v_nb2 = _mm256_load_ps(full_row2_ptr); v_nb3 = _mm256_load_ps(full_row3_ptr);
                    v_nb4 = _mm256_load_ps(full_row4_ptr); v_nb5 = _mm256_load_ps(full_row5_ptr);
                    v_nb6 = _mm256_load_ps(full_row6_ptr); v_nb7 = _mm256_load_ps(full_row7_ptr);

                    broadcast_act_c0 = _mm256_set1_ps(postsig_scalars_arr[0]); broadcast_act_c1 = _mm256_set1_ps(postsig_scalars_arr[1]);
                    broadcast_act_c2 = _mm256_set1_ps(postsig_scalars_arr[2]); broadcast_act_c3 = _mm256_set1_ps(postsig_scalars_arr[3]);
                    broadcast_act_c4 = _mm256_set1_ps(postsig_scalars_arr[4]); broadcast_act_c5 = _mm256_set1_ps(postsig_scalars_arr[5]);
                    broadcast_act_c6 = _mm256_set1_ps(postsig_scalars_arr[6]); broadcast_act_c7 = _mm256_set1_ps(postsig_scalars_arr[7]);
                    out_val0 = _mm256_mul_ps(v_nb0, broadcast_act_c0); out_val1 = _mm256_mul_ps(v_nb1, broadcast_act_c1);
                    out_val2 = _mm256_mul_ps(v_nb2, broadcast_act_c2); out_val3 = _mm256_mul_ps(v_nb3, broadcast_act_c3);
                    out_val4 = _mm256_mul_ps(v_nb4, broadcast_act_c4); out_val5 = _mm256_mul_ps(v_nb5, broadcast_act_c5);
                    out_val6 = _mm256_mul_ps(v_nb6, broadcast_act_c6); out_val7 = _mm256_mul_ps(v_nb7, broadcast_act_c7);

                    float * out_row0_target = out_base_b + (size_t)c_eff0 * NB; float * out_row1_target = out_base_b + (size_t)c_eff1 * NB;
                    float * out_row2_target = out_base_b + (size_t)c_eff2 * NB; float * out_row3_target = out_base_b + (size_t)c_eff3 * NB;
                    float * out_row4_target = out_base_b + (size_t)c_eff4 * NB; float * out_row5_target = out_base_b + (size_t)c_eff5 * NB;
                    float * out_row6_target = out_base_b + (size_t)c_eff6 * NB; float * out_row7_target = out_base_b + (size_t)c_eff7 * NB;
                    _mm256_store_ps(out_row0_target, out_val0); _mm256_store_ps(out_row1_target, out_val1);
                    _mm256_store_ps(out_row2_target, out_val2); _mm256_store_ps(out_row3_target, out_val3);
                    _mm256_store_ps(out_row4_target, out_val4); _mm256_store_ps(out_row5_target, out_val5);
                    _mm256_store_ps(out_row6_target, out_val6); _mm256_store_ps(out_row7_target, out_val7);
                }
            }
        }
    } else { 
        float presig_scalars_arr[8] __attribute__((aligned(32)));
        float postsig_scalars_arr[8] __attribute__((aligned(32)));

        __m256 vp_p01, vp_p23, vp_p45, vp_p67;
        __m256 v_nb_p01, v_nb_p23, v_nb_p45, v_nb_p67;
        __m256 broadcast_act_p01, broadcast_act_p23, broadcast_act_p45, broadcast_act_p67;
        __m256 out_val_p01, out_val_p23, out_val_p45, out_val_p67;

        if (agg_mode == 0) { 
            const float * const w_base_ptr = w_input;
            const float * const bias_base_ptr = bias_input;
            for (int b_loop_idx = 0; b_loop_idx < B; ++b_loop_idx) {
                const float * const v_full_base_b = v_full_input + (size_t)b_loop_idx * C * NB;
                const float * const v_pack_base_b = v_pack_input + (size_t)b_loop_idx * C * K;
                float       * const out_base_b    = out_output   + (size_t)b_loop_idx * C * NB;

                for (int c_loop_base = 0; c_loop_base < C; c_loop_base += 8) {
                    const int c_eff0 = c_loop_base + 0, c_eff2 = c_loop_base + 2, c_eff4 = c_loop_base + 4, c_eff6 = c_loop_base + 6;

                    vp_p01 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff0 * K);
                    vp_p23 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff2 * K);
                    vp_p45 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff4 * K);
                    vp_p67 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff6 * K);

                    const __m256 wp_p01 = _mm256_loadu_ps(w_base_ptr + (size_t)c_eff0 * K);
                    const __m256 wp_p23 = _mm256_loadu_ps(w_base_ptr + (size_t)c_eff2 * K);
                    const __m256 wp_p45 = _mm256_loadu_ps(w_base_ptr + (size_t)c_eff4 * K);
                    const __m256 wp_p67 = _mm256_loadu_ps(w_base_ptr + (size_t)c_eff6 * K);

                    v_nb_p01 = _mm256_loadu_ps(v_full_base_b + (size_t)c_eff0 * NB);
                    v_nb_p23 = _mm256_loadu_ps(v_full_base_b + (size_t)c_eff2 * NB);
                    v_nb_p45 = _mm256_loadu_ps(v_full_base_b + (size_t)c_eff4 * NB);
                    v_nb_p67 = _mm256_loadu_ps(v_full_base_b + (size_t)c_eff6 * NB);

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
                    const __m256 postsig_avx_vals_computed = sigmoid256_ps(presig_avx_vals);
                    _mm256_store_ps(postsig_scalars_arr, postsig_avx_vals_computed);

                    broadcast_act_p01 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[1]), _mm_set1_ps(postsig_scalars_arr[0]));
                    broadcast_act_p23 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[3]), _mm_set1_ps(postsig_scalars_arr[2]));
                    broadcast_act_p45 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[5]), _mm_set1_ps(postsig_scalars_arr[4]));
                    broadcast_act_p67 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[7]), _mm_set1_ps(postsig_scalars_arr[6]));

                    out_val_p01 = _mm256_mul_ps(v_nb_p01, broadcast_act_p01);
                    out_val_p23 = _mm256_mul_ps(v_nb_p23, broadcast_act_p23);
                    out_val_p45 = _mm256_mul_ps(v_nb_p45, broadcast_act_p45);
                    out_val_p67 = _mm256_mul_ps(v_nb_p67, broadcast_act_p67);

                    float* out_row0_target = out_base_b + (size_t)c_eff0 * NB;
                    float* out_row2_target = out_base_b + (size_t)c_eff2 * NB;
                    float* out_row4_target = out_base_b + (size_t)c_eff4 * NB;
                    float* out_row6_target = out_base_b + (size_t)c_eff6 * NB;

                    _mm256_storeu_ps(out_row0_target, out_val_p01);
                    _mm256_storeu_ps(out_row2_target, out_val_p23);
                    _mm256_storeu_ps(out_row4_target, out_val_p45);
                    _mm256_storeu_ps(out_row6_target, out_val_p67);
                }
            }
        } else if (agg_mode == 1) { 
            for (int b_loop_idx = 0; b_loop_idx < B; ++b_loop_idx) {
                const float * const v_full_base_b = v_full_input + (size_t)b_loop_idx * C * NB;
                const float * const v_pack_base_b = v_pack_input + (size_t)b_loop_idx * C * K;
                float       * const out_base_b    = out_output   + (size_t)b_loop_idx * C * NB;

                for (int c_loop_base = 0; c_loop_base < C; c_loop_base += 8) {
                    const int c_eff0 = c_loop_base + 0, c_eff2 = c_loop_base + 2, c_eff4 = c_loop_base + 4, c_eff6 = c_loop_base + 6;

                    vp_p01 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff0 * K);
                    vp_p23 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff2 * K);
                    vp_p45 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff4 * K);
                    vp_p67 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff6 * K);

                    v_nb_p01 = _mm256_loadu_ps(v_full_base_b + (size_t)c_eff0 * NB);
                    v_nb_p23 = _mm256_loadu_ps(v_full_base_b + (size_t)c_eff2 * NB);
                    v_nb_p45 = _mm256_loadu_ps(v_full_base_b + (size_t)c_eff4 * NB);
                    v_nb_p67 = _mm256_loadu_ps(v_full_base_b + (size_t)c_eff6 * NB);

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
                    const __m256 postsig_avx_vals_computed = sigmoid256_ps(presig_avx_vals);
                    _mm256_store_ps(postsig_scalars_arr, postsig_avx_vals_computed);

                    broadcast_act_p01 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[1]), _mm_set1_ps(postsig_scalars_arr[0]));
                    broadcast_act_p23 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[3]), _mm_set1_ps(postsig_scalars_arr[2]));
                    broadcast_act_p45 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[5]), _mm_set1_ps(postsig_scalars_arr[4]));
                    broadcast_act_p67 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[7]), _mm_set1_ps(postsig_scalars_arr[6]));

                    out_val_p01 = _mm256_mul_ps(v_nb_p01, broadcast_act_p01);
                    out_val_p23 = _mm256_mul_ps(v_nb_p23, broadcast_act_p23);
                    out_val_p45 = _mm256_mul_ps(v_nb_p45, broadcast_act_p45);
                    out_val_p67 = _mm256_mul_ps(v_nb_p67, broadcast_act_p67);

                    float* out_row0_target = out_base_b + (size_t)c_eff0 * NB;
                    float* out_row2_target = out_base_b + (size_t)c_eff2 * NB;
                    float* out_row4_target = out_base_b + (size_t)c_eff4 * NB;
                    float* out_row6_target = out_base_b + (size_t)c_eff6 * NB;

                    _mm256_storeu_ps(out_row0_target, out_val_p01);
                    _mm256_storeu_ps(out_row2_target, out_val_p23);
                    _mm256_storeu_ps(out_row4_target, out_val_p45);
                    _mm256_storeu_ps(out_row6_target, out_val_p67);
                }
            }
        } else { 
            const float inv_K_val = 1.0f / (float)K;
            const __m256 inv_K_vec = _mm256_set1_ps(inv_K_val);
            for (int b_loop_idx = 0; b_loop_idx < B; ++b_loop_idx) {
                const float * const v_full_base_b = v_full_input + (size_t)b_loop_idx * C * NB;
                const float * const v_pack_base_b = v_pack_input + (size_t)b_loop_idx * C * K;
                float       * const out_base_b    = out_output   + (size_t)b_loop_idx * C * NB;

                for (int c_loop_base = 0; c_loop_base < C; c_loop_base += 8) {
                    const int c_eff0 = c_loop_base + 0, c_eff2 = c_loop_base + 2, c_eff4 = c_loop_base + 4, c_eff6 = c_loop_base + 6;

                    vp_p01 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff0 * K);
                    vp_p23 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff2 * K);
                    vp_p45 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff4 * K);
                    vp_p67 = _mm256_loadu_ps(v_pack_base_b + (size_t)c_eff6 * K);

                    v_nb_p01 = _mm256_loadu_ps(v_full_base_b + (size_t)c_eff0 * NB);
                    v_nb_p23 = _mm256_loadu_ps(v_full_base_b + (size_t)c_eff2 * NB);
                    v_nb_p45 = _mm256_loadu_ps(v_full_base_b + (size_t)c_eff4 * NB);
                    v_nb_p67 = _mm256_loadu_ps(v_full_base_b + (size_t)c_eff6 * NB);

                    __m256 sum01 = _mm256_hadd_ps(vp_p01, vp_p01); sum01 = _mm256_hadd_ps(sum01, sum01);
                    __m256 sum23 = _mm256_hadd_ps(vp_p23, vp_p23); sum23 = _mm256_hadd_ps(sum23, sum23);
                    __m256 sum45 = _mm256_hadd_ps(vp_p45, vp_p45); sum45 = _mm256_hadd_ps(sum45, sum45);
                    __m256 sum67 = _mm256_hadd_ps(vp_p67, vp_p67); sum67 = _mm256_hadd_ps(sum67, sum67);

                    float temp_sums[8];
                    _mm256_storeu_ps(temp_sums, sum01); presig_scalars_arr[0] = temp_sums[0]; presig_scalars_arr[1] = temp_sums[4];
                    _mm256_storeu_ps(temp_sums, sum23); presig_scalars_arr[2] = temp_sums[0]; presig_scalars_arr[3] = temp_sums[4];
                    _mm256_storeu_ps(temp_sums, sum45); presig_scalars_arr[4] = temp_sums[0]; presig_scalars_arr[5] = temp_sums[4];
                    _mm256_storeu_ps(temp_sums, sum67); presig_scalars_arr[6] = temp_sums[0]; presig_scalars_arr[7] = temp_sums[4];

                    __m256 presig_avx_vals = _mm256_load_ps(presig_scalars_arr);
                    presig_avx_vals = _mm256_mul_ps(presig_avx_vals, inv_K_vec);

                    const __m256 postsig_avx_vals_computed = sigmoid256_ps(presig_avx_vals);
                    _mm256_store_ps(postsig_scalars_arr, postsig_avx_vals_computed);

                    broadcast_act_p01 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[1]), _mm_set1_ps(postsig_scalars_arr[0]));
                    broadcast_act_p23 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[3]), _mm_set1_ps(postsig_scalars_arr[2]));
                    broadcast_act_p45 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[5]), _mm_set1_ps(postsig_scalars_arr[4]));
                    broadcast_act_p67 = _mm256_set_m128(_mm_set1_ps(postsig_scalars_arr[7]), _mm_set1_ps(postsig_scalars_arr[6]));

                    out_val_p01 = _mm256_mul_ps(v_nb_p01, broadcast_act_p01);
                    out_val_p23 = _mm256_mul_ps(v_nb_p23, broadcast_act_p23);
                    out_val_p45 = _mm256_mul_ps(v_nb_p45, broadcast_act_p45);
                    out_val_p67 = _mm256_mul_ps(v_nb_p67, broadcast_act_p67);

                    float* out_row0_target = out_base_b + (size_t)c_eff0 * NB;
                    float* out_row2_target = out_base_b + (size_t)c_eff2 * NB;
                    float* out_row4_target = out_base_b + (size_t)c_eff4 * NB;
                    float* out_row6_target = out_base_b + (size_t)c_eff6 * NB;

                    _mm256_storeu_ps(out_row0_target, out_val_p01);
                    _mm256_storeu_ps(out_row2_target, out_val_p23);
                    _mm256_storeu_ps(out_row4_target, out_val_p45);
                    _mm256_storeu_ps(out_row6_target, out_val_p67);
                }
            }
        }
    }
} 