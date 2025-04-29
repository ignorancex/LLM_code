
#include "../headers/other_layers.h"
#include "../headers/common_funcs.h"

void cpu_add(fms_dt *src, fms_dt *dst, layer_specs conv_l_specs)
{
    const int layer_ofms_size = conv_l_specs.layer_ofm_height * conv_l_specs.layer_ofm_width *
                                conv_l_specs.layer_num_fils;

    scales_dt close_layer_scale = conv_l_specs.layer_ofms_scale;
    fms_dt close_layer_zp = conv_l_specs.layer_ofms_zero_point;

    scales_dt far_layer_scale = conv_l_specs.skip_connection_other_layer_scale;
    fms_dt far_layer_zp = conv_l_specs.skip_connection_other_layer_zero_point;

    scales_dt add_layer_scale_rec = conv_l_specs.add_layer_scale_reciprocal;
    fms_dt add_layer_zp = conv_l_specs.add_layer_zero_point;

    for (int i = 0; i < layer_ofms_size; i++)
    {
        // if(i < 112)printf("%d, %d, \n", dst[i], src[i]);
        scales_dt scaled_result = (close_layer_scale * (dst[i] - close_layer_zp) +
                                   far_layer_scale * (src[i] - far_layer_zp)) *
                                      add_layer_scale_rec +
                                  add_layer_zp;
        scaled_result = scaled_result + 0.5 - (scaled_result < 0);
        dst[i] = clamp((int16_t)scaled_result);
    }
}

void cpu_avgpool_all_hw(fms_dt *fms,
                 const pooling_layer_specs layer_specs_struct)
{

    const int avgpool_input_depth = layer_specs_struct.ifm_depth;
    const int avgpool_input_height = layer_specs_struct.ifm_height;
    const int avgpool_input_width = layer_specs_struct.ifm_width;
    const int avgpool_input_hw = avgpool_input_height * avgpool_input_width;
    const int start_writing_index = MAX_FMS_SIZE - avgpool_input_depth; 

    for (int d = 0; d < avgpool_input_depth; d++)
    {
        pss_dt tmp = 0;
        for (int hw = 0; hw < avgpool_input_hw; hw++)
        {
            tmp += fms[d * avgpool_input_hw + hw];
        }

        pss_f_dt scaled_tmp = (tmp / avgpool_input_hw - layer_specs_struct.ifms_zero_point) *
                                  layer_specs_struct.fused_scale +
                              layer_specs_struct.ofms_zero_point;
        fms[start_writing_index + d] = clamp(scaled_tmp);
        //printf("%d\n", fms[d]);
    }
}
