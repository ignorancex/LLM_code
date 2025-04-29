#include "../headers/conv_kernels.h"

/*
Unoptimized CPU implementations, just to validate the correctness of GPU results
*/

inline void load_tile_fw(fms_dt *ifms, fms_dt ifms_tile[][TILE_H_FW][FW_MAX_PADDED_TILE_W],
                         const int layer_depth,
                         const int ifm_height, const int ifm_width,
                         const int filter_dim, const int strides,
                         const int padding_top,
                         const int padding_bottom,
                         const int padding_left,
                         const int padding_right,
                         const int starting_h,
                         const fms_dt ifms_zp)
{

    const int ifms_hw = ifm_height * ifm_width;
    const int padded_tile_width = ifm_width + padding_left + padding_right;

    if (starting_h == 0)
    {
        // padding top
        for (int d = 0; d < layer_depth; d++)
        {
            for (int h = 0; h < padding_top; h++)
            {
                for (int w = 0; w < padded_tile_width; w++)
                {
                    ifms_tile[d][h][w] = ifms_zp;
                }
            }
        }
    }

    // padding left
    for (int d = 0; d < layer_depth; d++)
    {
        for (int h = 0; h < TILE_H_FW; h++)
        {
            for (int w = 0; w < padding_left; w++)
            {
                ifms_tile[d][h][w] = ifms_zp;
            }
        }
    }

    // padding right
    for (int d = 0; d < layer_depth; d++)
    {
        for (int h = 0; h < TILE_H_FW; h++)
        {
            for (int w = 0; w < padding_right; w++)
            {
                ifms_tile[d][h][w + padding_left + ifm_width] = ifms_zp;
            }
        }
    }

    const int starting_h_in_tile = starting_h == 0 ? padding_top : 0;
    const int shift_h_in_ifms = starting_h == 0 ? 0 : strides - (filter_dim - padding_bottom);

    for (int d = 0; d < layer_depth; d++)
    {
        for (int h = starting_h_in_tile; h < TILE_H_FW; h++)
        {
            for (int w = padding_left; w < padded_tile_width; w++)
            {
                if ((h + starting_h + shift_h_in_ifms - starting_h_in_tile) < ifm_height &&
                    w < ifm_width + padding_left)
                {
                    ifms_tile[d][h][w] =
                        ifms[d * ifms_hw + (h + starting_h + shift_h_in_ifms - starting_h_in_tile) * ifm_width +
                             w - padding_left];
                }
                else
                {
                    ifms_tile[d][h][w] = ifms_zp;
                }
            }
        }
    }
}

void pw_conv_fw(fms_dt *ifms, fms_dt *ofms, weights_dt *weights,
                fused_scales_dt *fused_scales,
                biases_dt *fused_zps,
                int *fused_params_offsets,
                layer_specs l_specs)
{

    const int layer_strides = l_specs.strides;
    const int layer_num_fils = l_specs.layer_num_fils;
    const int layer_depth = l_specs.layer_depth;
    const int layer_ofm_height = l_specs.layer_ofm_height;
    const int layer_ofm_width = l_specs.layer_ofm_width;
    const int ofms_h_w = layer_ofm_height * layer_ofm_width;
    const int layer_fused_params_offset = fused_params_offsets[l_specs.layer_index];
    const int layer_weights_offset = l_specs.layer_weights_offset;

    for (int f = 0; f < layer_num_fils; f++)
    {
        for (int h = 0; h < layer_ofm_height; h++)
        {
            for (int w = 0; w < layer_ofm_width; w++)
            {
                pss_dt sum = 0;
                for (int d = 0; d < layer_depth; d++)
                {
                    sum += ifms[d * ofms_h_w * layer_strides * layer_strides +
                                h * layer_ofm_width * layer_strides + w] *
                           weights[layer_weights_offset + f * layer_depth + d];
                }
#if DATA_TYPE == FLOAT_DTYPE
                if (sum < 0)
                {
                    sum = 0;
                }
                else
                {
                    sum = sum * DUMMY_SCALE + DUMMY_BIAS;
                }
                ofms[f * ofms_h_w + h * layer_ofm_width + w] = sum;
#elif DATA_TYPE == INT8_DTYPE
                if (l_specs.layer_activation == RELU6)
                {
                    // if (f == 0 && h == 0 && w == 0)
                    // {
                    //     printf("\n cpu: %d >> %d\n", sum, quant_relu6(sum, fused_scales[layer_fused_params_offset + f],
                    //                 fused_zps[layer_fused_params_offset + f],
                    //                 l_specs.layer_ofms_zero_point, l_specs.relu_threshold));
                    // }
                    ofms[f * ofms_h_w + h * layer_ofm_width + w] =
                        quant_relu6(sum, fused_scales[layer_fused_params_offset + f],
                                    fused_zps[layer_fused_params_offset + f],
                                    l_specs.layer_ofms_zero_point, l_specs.relu_threshold);
                }
                else if (l_specs.layer_activation == 0)
                {
                    ofms[f * ofms_h_w + h * layer_ofm_width + w] =
                        quant_no_activation(sum, fused_scales[layer_fused_params_offset + f],
                                            fused_zps[layer_fused_params_offset + f],
                                            l_specs.layer_ofms_zero_point);
                }
#endif
            }
        }
    }
}

void dw_conv_fw(fms_dt *ifms, fms_dt *ofms, weights_dt *weights,
                fused_scales_dt *fused_scales,
                biases_dt *fused_zps,
                int *fused_params_offsets,
                const int layer_depth,
                layer_specs l_specs)
{
    const int filter_dim = l_specs.filter_size;
    int filter_wh = filter_dim * filter_dim;
    const int strides = l_specs.strides;
    const int layer_num_fils = l_specs.layer_num_fils;
    const int layer_ofm_height = l_specs.layer_ofm_height;
    const int layer_ofm_width = l_specs.layer_ofm_width;
    const int layer_ifm_width = l_specs.layer_ifm_width;
    const int num_of_tiles_h = l_specs.layer_num_of_ifm_tiles_h;
    const int num_of_tiles_w = l_specs.layer_num_of_ifm_tiles_w;
    const int ofms_hw = layer_ofm_height * layer_ofm_width;
    const int padding_top = l_specs.padding_top;
    const int padding_bottom = l_specs.padding_bottom;
    const int padding_left = l_specs.padding_left;
    const int padding_right = l_specs.padding_right;
    const int layer_weights_offset = l_specs.layer_weights_offset;
    const int layer_fused_params_offset = fused_params_offsets[l_specs.layer_index];
    const fms_dt ofms_zp = l_specs.layer_ofms_zero_point;
    const fms_dt ifms_zp = l_specs.layer_ifms_zero_point;
    const scales_dt relu_threshold = l_specs.relu_threshold;

    fms_dt ifms_tile[layer_depth][TILE_H_FW][FW_MAX_PADDED_TILE_W];

    const int rows_consumed_each_time = (TILE_H_FW - (filter_dim - strides));
    const int rows_produced_each_time = (rows_consumed_each_time / strides);

#if WEIGHTS_SHARED_WITH_GPU && PADDED_DW_WEIGHTS
    filter_wh = least_pow_of_2_geq(filter_wh);
#endif

    for (int o_h = 0; o_h < (layer_ofm_height + rows_produced_each_time - 1) / rows_produced_each_time; o_h++)
    {
        load_tile_fw(ifms, ifms_tile, layer_depth, layer_ofm_height * strides,
                     layer_ofm_width * strides, filter_dim, strides, padding_top, padding_bottom,
                     padding_left, padding_right, o_h * rows_consumed_each_time, ifms_zp);

        for (int h = 0; h < rows_produced_each_time; h++)
        {
            if ((o_h * rows_produced_each_time + h) < layer_ofm_height)
            {
                for (int w = 0; w < layer_ifm_width / strides; w++)
                {
                    for (int d = 0; d < layer_depth; d++)
                    {
                        pss_dt sum = 0;
                        for (int c_h = 0; c_h < filter_dim; c_h++)
                        {
                            for (int c_w = 0; c_w < filter_dim; c_w++)
                            {
                                sum += ifms_tile[d][h * strides + c_h][w * strides + c_w] *
                                       weights[layer_weights_offset + (c_h * filter_dim + c_w) + d * filter_wh];
                                //                                if (o_h == 0 && h == 0 && w == 0 && d == 0)
                                //                                {
                                // #if DATA_TYPE == INT8_DTYPE
                                //                                     printf("%d * %d, ", ifms_tile[d][h * strides + c_h][w * strides + c_w],
                                //                                            weights[layer_weights_offset + (c_h * filter_dim + c_w) + d * filter_wh]);
                                // #elif DATA_TYPE == FLOAT_DTYPE
                                //                                     printf("%f * %f, ", ifms_tile[d][h * strides + c_h][w * strides + c_w],
                                //                                            weights[layer_weights_offset + (c_h * filter_dim + c_w) + d * filter_wh]);
                                // #endif
                                //                                }
                            }
                            // if (o_h == 0 && h == 0 && w == 0 && d == 0)
                            // {
                            //     printf("\n");
                            // }
                        }
                        //                         if (o_h == 0 && h == 0 && w == 0 && d == 0)
                        //                         {
                        // #if DATA_TYPE == INT8_DTYPE
                        //                             printf("\n%d\n", sum);
                        //                             printf("\n%d\n", quant_relu6(sum, fused_scales[layer_fused_params_offset + d],
                        //                                             fused_zps[layer_fused_params_offset + d], ofms_zp, relu_threshold));
                        // #elif DATA_TYPE == FLOAT_DTYPE
                        //                             printf("\n%f\n", sum);
                        // #endif
                        //                         }
                        // if (o_h * rows_produced_each_time + h == 14 && w == 29 && d == 12)
                        //                     {
                        //                         int sum0 = sum + fused_zps[layer_fused_params_offset + d];
                        //                         float scaled_pss = sum0 * fused_scales[layer_fused_params_offset + d];
                        //                         printf("%f\n", fused_scales[layer_fused_params_offset + d]);
                        //                         printf("%f\n", scaled_pss);
                        //                         scaled_pss += (int8_t)ofms_zp;
                        //                         printf("%f\n", scaled_pss);
                        //                         scaled_pss += 0.5 - (scaled_pss < 0);
                        //                         printf("%f\n", scaled_pss);
                        //                         printf("%d\n", clamp((int16_t)scaled_pss));
                        //                     }
#if DATA_TYPE == FLOAT_DTYPE
                        if (sum < 0)
                        {
                            sum = 0;
                        }
                        else
                        {
                            sum = sum * DUMMY_SCALE + DUMMY_BIAS;
                        }
                        ofms[d * ofms_hw + (o_h * rows_produced_each_time + h) * layer_ofm_width + w] = sum;
#elif DATA_TYPE == INT8_DTYPE
                        if (l_specs.layer_activation == RELU6)
                        {
                            ofms[d * ofms_hw + (o_h * rows_produced_each_time + h) * layer_ofm_width + w] =
                                quant_relu6(sum, fused_scales[layer_fused_params_offset + d],
                                            fused_zps[layer_fused_params_offset + d], ofms_zp, relu_threshold);
                        }
                        else if (l_specs.layer_activation == 0)
                        {
                            ofms[d * ofms_hw + (o_h * rows_produced_each_time + h) * layer_ofm_width + w] =
                                quant_no_activation(sum, fused_scales[layer_fused_params_offset + d],
                                                    fused_zps[layer_fused_params_offset + d], ofms_zp);
                        }
#endif
                    }
                }
            }
        }
    }
}

void convolutionCPU_fw(fms_dt *ifms, fms_dt *ofms,
                       weights_dt *pw_weights,
                       weights_dt *dw_weights,
                       fused_scales_dt *fused_scales,
                       biases_dt *fused_zps,
                       int *fused_params_offsets,
                       layer_specs l_specs)
{
    if (l_specs.conv_layer_type == PW_CONV)
    {
        pw_conv_fw(ifms, ofms, pw_weights, fused_scales, fused_zps, fused_params_offsets, l_specs);
    }
    else if (l_specs.conv_layer_type == DW_CONV)
    {
        dw_conv_fw(ifms, ofms, dw_weights,
                   fused_scales,
                   fused_zps,
                   fused_params_offsets,
                   l_specs.layer_depth,
                   l_specs);
    }
}