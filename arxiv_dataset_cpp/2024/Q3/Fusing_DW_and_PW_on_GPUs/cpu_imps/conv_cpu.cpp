#include "../headers/conv_kernels.h"

/*
Unoptimized CPU implementations, just to validate the correctness of GPU results
*/

inline void load_tile(fms_dt *ifms, fms_dt ifms_tile[][MAX_PADDED_TILE_H][MAX_PADDED_TILE_W],
                      const int layer_depth,
                      const int ifm_height, const int ifm_width,
                      const int filter_dim, const int strides,
                      const int padding_top,
                      const int padding_bottom,
                      const int padding_left,
                      const int padding_right,
                      const int starting_h, const int starting_w,
                      const fms_dt ifms_zp) 
{

    const int ifms_hw = ifm_height * ifm_width;
    const int padded_tile_width = TILE_W + padding_left + padding_right;
    const int padded_tile_height = TILE_H + padding_top + padding_bottom;

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

    if (starting_w == 0)
    {
        for (int d = 0; d < layer_depth; d++)
        {
            for (int h = 0; h < padded_tile_height; h++)
            {
                for (int w = 0; w < padding_left; w++)
                {
                    ifms_tile[d][h][w] = ifms_zp;
                }
            }
        }
    }
    else
    {
        for (int d = 0; d < layer_depth; d++)
        {
            for (int h = 0; h < padded_tile_height; h++)
            {
                for (int w = 0; w < filter_dim - strides; w++)
                {
                    ifms_tile[d][h][w] = ifms_tile[d][h][w + TILE_W];
                }
            }
        }
    }

    const int starting_h_in_tile = starting_h == 0 ? padding_top : 0;
    const int shift_h_in_ifms = starting_h == 0 ? 0 : strides - (filter_dim - padding_bottom);
    const int starting_w_in_tile = starting_w == 0 ? padding_left : filter_dim - strides;
    const int shift_w_in_ifms = starting_w == 0 ? 0 : padding_right;

    for (int d = 0; d < layer_depth; d++)
    {
        for (int h = starting_h_in_tile; h < padded_tile_height; h++)
        {
            for (int w = starting_w_in_tile; w < padded_tile_width; w++)
            {
                if ((h + starting_h + shift_h_in_ifms - starting_h_in_tile) < ifm_height &&
                    (starting_w + w + shift_w_in_ifms - starting_w_in_tile) < ifm_width)
                {
                    ifms_tile[d][h][w] =
                        ifms[d * ifms_hw + (h + starting_h + shift_h_in_ifms - starting_h_in_tile) * ifm_width +
                             (starting_w + w - starting_w_in_tile + shift_w_in_ifms)];
                }
                else
                {
                    ifms_tile[d][h][w] = ifms_zp;
                }
            }
        }
    }
}

void pw_conv(fms_dt *ifms, fms_dt *ofms, weights_dt *weights,
             fused_scales_dt *fused_scales,
             biases_dt *fused_zps,
             layer_specs l_specs)
{
    const int layer_strides = l_specs.strides;
    const int layer_num_fils = l_specs.layer_num_fils;
    const int layer_depth = l_specs.layer_depth;
    const int layer_ofm_height = l_specs.layer_ofm_height;
    const int layer_ofm_width = l_specs.layer_ofm_width;
    const int ofms_h_w = layer_ofm_height * layer_ofm_width;

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
                           weights[f * layer_depth + d];
                }
                if (sum < 0)
                {
                    sum = 0;
                }
                ofms[f * ofms_h_w + h * layer_ofm_width + w] =
                    quant_relu6(sum, fused_scales[64 + f], fused_zps[64 + f], l_specs.layer_ofms_zero_point, l_specs.relu_threshold);
                if (f == 0 && h == 0 && w == 0)
                {
                    printf("%d**\n", quant_relu6(sum, fused_scales[64 + f], fused_zps[64 + f], l_specs.layer_ofms_zero_point, l_specs.relu_threshold));
                }
            }
        }
    }
}

void dw_conv(fms_dt *ifms, fms_dt *ofms, weights_dt *weights,
             fused_scales_dt *fused_scales,
             biases_dt *fused_zps,
             const int layer_depth,
             layer_specs l_specs)
{
    const int filter_dim = l_specs.filter_size;
    const int strides = l_specs.strides;
    const int layer_num_fils = l_specs.layer_num_fils;
    const int layer_ofm_height = l_specs.layer_ofm_height;
    const int layer_ofm_width = l_specs.layer_ofm_width;
    const int num_of_tiles_h = l_specs.layer_num_of_ifm_tiles_h;
    const int num_of_tiles_w = l_specs.layer_num_of_ifm_tiles_w;
    const int ofms_hw = layer_ofm_height * layer_ofm_width;
    const int padding_top = l_specs.padding_top;
    const int padding_bottom = l_specs.padding_bottom;
    const int padding_left = l_specs.padding_left;
    const int padding_right = l_specs.padding_right;
    const int layer_Weights_offset = l_specs.layer_weights_offset;
    const fms_dt ofms_zp = l_specs.layer_ofms_zero_point;
    const fms_dt ifms_zp = l_specs.layer_ifms_zero_point;
    const scales_dt relu_threshold = l_specs.relu_threshold;

    fms_dt ifms_tile[layer_depth][MAX_PADDED_TILE_H][MAX_PADDED_TILE_W];

    for (int o_h = 0; o_h < num_of_tiles_h; o_h++)
    {
        for (int o_w = 0; o_w < num_of_tiles_w; o_w++)
        {
            load_tile(ifms, ifms_tile, layer_depth, layer_ofm_height * strides,
                      layer_ofm_width * strides, filter_dim, strides, padding_top, padding_bottom,
                      padding_left, padding_right, o_h * TILE_H, o_w * TILE_W, ifms_zp);

            for (int h = 0; h < TILE_H / strides; h++)
            {
                for (int w = 0; w < TILE_W / strides; w++)
                {
                    for (int d = 0; d < layer_depth; d++)
                    {
                        pss_dt sum = 0;
                        for (int c_h = 0; c_h < filter_dim; c_h++)
                        {
                            for (int c_w = 0; c_w < filter_dim; c_w++)
                            {
                                sum += ifms_tile[d][h * strides + c_h][w * strides + c_w] *
                                       weights[layer_Weights_offset + (c_h * filter_dim + c_w) * layer_depth + d];
                                // if (h == 3 && w == 36 && d == 1)
                                // {
                                //     cout << ifms_tile[d][h * strides + c_h][w * strides + c_w] << " * "
                                //          << weights[layer_Weights_offset + (c_h * filter_dim + c_w) * layer_depth + d]
                                //          << " + ";
                                // }
                            }
                            // if (h == 3 && w == 36 && d == 1)
                            // {
                            //     cout << "\n";
                            // }
                        }
                        ofms[d * ofms_hw + (o_h * TILE_H / strides + h) * layer_ofm_width + (o_w * TILE_W / strides + w)] =
                            quant_relu6(sum, fused_scales[128 + d],
                                        fused_zps[128 + d], ofms_zp, relu_threshold);
                    }
                }
            }
        }
    }
}

void s_conv(fms_dt *ifms, fms_dt *ofms, fms_dt *weights,
            layer_specs l_specs)
{
    const int filter_dim = l_specs.filter_size;
    const int layer_strides = l_specs.strides;
    const int layer_num_fils = l_specs.layer_num_fils;
    const int layer_depth = l_specs.layer_depth;
    const int layer_ofm_height = l_specs.layer_ofm_height;
    const int layer_ofm_width = l_specs.layer_ofm_width;
    const int ofms_h_w = layer_ofm_height * layer_ofm_width;

    for (int f = 0; f < layer_num_fils; f++)
    {
        for (int h = 0; h < layer_ofm_height; h++)
        {
            for (int w = 0; w < layer_ofm_width; w++)
            {
                int sum = 0;
                for (int d = 0; d < layer_depth; d++)
                {
                    for (int c_h = 0; c_h < filter_dim; c_h++)
                    {
                        for (int c_w = 0; c_w < filter_dim; c_w++)
                        {
                            sum += ifms[d * ofms_h_w * layer_strides * layer_strides +
                                        h * layer_ofm_width * layer_strides + w] *
                                   weights[f * layer_depth + d];
                        }
                    }
                }
                ofms[f * ofms_h_w + h * layer_ofm_width + w] = sum;
            }
        }
    }
}

void convolutionCPU(fms_dt *ifms, fms_dt *ofms, weights_dt *weights,
                    fused_scales_dt *fused_scales,
                    biases_dt *fused_zps,
                    layer_specs l_specs)
{
    if (l_specs.conv_layer_type == PW_CONV)
    {
        pw_conv(ifms, ofms, weights, fused_scales, fused_zps, l_specs);
    }
    else if (l_specs.conv_layer_type == DW_CONV)
    {
        dw_conv(ifms, ofms, weights,
                fused_scales,
                fused_zps,
                l_specs.layer_depth,
                l_specs);
    }
}