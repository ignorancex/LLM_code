#include "layers_specs.h"

#if MODEL_ID == MOB_V1 

void layer_specs_init(layer_specs *layer_specs_seq, pooling_layer_specs * pooling_layer_specs_seq)
{

    layer_specs_seq[1].layer_index = 1;
    layer_specs_seq[1].conv_layer_type = S_CONV; 
    layer_specs_seq[1].layer_num_fils = 32;
    layer_specs_seq[1].strides = 2;
    layer_specs_seq[1].filter_size = 3;
    layer_specs_seq[1].padding_left = 0;
    layer_specs_seq[1].padding_right = 1;
    layer_specs_seq[1].padding_top = 0;
    layer_specs_seq[1].padding_bottom = 1;
    layer_specs_seq[1].layer_depth = 3;
    layer_specs_seq[1].layer_ifm_height = 224;
    layer_specs_seq[1].layer_ifm_width = 224;
    layer_specs_seq[1].layer_ofm_height = 112;
    layer_specs_seq[1].layer_ofm_width = 112;
    layer_specs_seq[1].layer_activation = RELU6;
    layer_specs_seq[1].layer_num_of_ifm_tiles_h = (224 + TILE_H - 1) / TILE_H;
    layer_specs_seq[1].layer_num_of_ifm_tiles_w = (224 + TILE_W - 1) / TILE_W;
    layer_specs_seq[1].layer_num_of_ofm_tiles_h = (112 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(112 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 1;
    layer_specs_seq[1].layer_weights_offset = 0;
    layer_specs_seq[1].write_to_result_or_channels = 1;
    layer_specs_seq[1].write_to_tmp = 0;
    layer_specs_seq[1].followed_by = 0;
    layer_specs_seq[1].layer_ifms_zero_point = -1;
    layer_specs_seq[1].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[1].relu_threshold = 255 ;
    layer_specs_seq[1].layer_ofms_zero_point = -128;
    layer_specs_seq[1].add_layer_scale_reciprocal = 1;
    layer_specs_seq[1].add_layer_zero_point = 0;
    layer_specs_seq[1].skip_connection_other_layer_scale = 1;
    layer_specs_seq[1].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[1].data_layout = CHW;


    layer_specs_seq[2].layer_index = 2;
    layer_specs_seq[2].conv_layer_type = DW_CONV; 
    layer_specs_seq[2].layer_num_fils = 32;
    layer_specs_seq[2].strides = 1;
    layer_specs_seq[2].filter_size = 3;
    layer_specs_seq[2].padding_left = 1;
    layer_specs_seq[2].padding_right = 1;
    layer_specs_seq[2].padding_top = 1;
    layer_specs_seq[2].padding_bottom = 1;
    layer_specs_seq[2].layer_depth = 32;
    layer_specs_seq[2].layer_ifm_height = 112;
    layer_specs_seq[2].layer_ifm_width = 112;
    layer_specs_seq[2].layer_ofm_height = 112;
    layer_specs_seq[2].layer_ofm_width = 112;
    layer_specs_seq[2].layer_activation = RELU6;
    layer_specs_seq[2].layer_num_of_ifm_tiles_h = (112 + TILE_H - 1) / TILE_H;
    layer_specs_seq[2].layer_num_of_ifm_tiles_w = (112 + TILE_W - 1) / TILE_W;
    layer_specs_seq[2].layer_num_of_ofm_tiles_h = (112 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(112 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 2;
    layer_specs_seq[2].layer_weights_offset = 0;
    layer_specs_seq[2].write_to_result_or_channels = 1;
    layer_specs_seq[2].write_to_tmp = 0;
    layer_specs_seq[2].followed_by = 0;
    layer_specs_seq[2].layer_ifms_zero_point = -128;
    layer_specs_seq[2].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[2].relu_threshold = 255 ;
    layer_specs_seq[2].layer_ofms_zero_point = -128;
    layer_specs_seq[2].add_layer_scale_reciprocal = 1;
    layer_specs_seq[2].add_layer_zero_point = 0;
    layer_specs_seq[2].skip_connection_other_layer_scale = 1;
    layer_specs_seq[2].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[2].data_layout = CHW;


    layer_specs_seq[3].layer_index = 3;
    layer_specs_seq[3].conv_layer_type = PW_CONV; 
    layer_specs_seq[3].layer_num_fils = 64;
    layer_specs_seq[3].strides = 1;
    layer_specs_seq[3].filter_size = 1;
    layer_specs_seq[3].padding_left = 0;
    layer_specs_seq[3].padding_right = 0;
    layer_specs_seq[3].padding_top = 0;
    layer_specs_seq[3].padding_bottom = 0;
    layer_specs_seq[3].layer_depth = 32;
    layer_specs_seq[3].layer_ifm_height = 112;
    layer_specs_seq[3].layer_ifm_width = 112;
    layer_specs_seq[3].layer_ofm_height = 112;
    layer_specs_seq[3].layer_ofm_width = 112;
    layer_specs_seq[3].layer_activation = RELU6;
    layer_specs_seq[3].layer_num_of_ifm_tiles_h = (112 + TILE_H - 1) / TILE_H;
    layer_specs_seq[3].layer_num_of_ifm_tiles_w = (112 + TILE_W - 1) / TILE_W;
    layer_specs_seq[3].layer_num_of_ofm_tiles_h = (112 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(112 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 3;
    layer_specs_seq[3].layer_weights_offset = 0;
    layer_specs_seq[3].write_to_result_or_channels = 1;
    layer_specs_seq[3].write_to_tmp = 0;
    layer_specs_seq[3].followed_by = 0;
    layer_specs_seq[3].layer_ifms_zero_point = -128;
    layer_specs_seq[3].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[3].relu_threshold = 255 ;
    layer_specs_seq[3].layer_ofms_zero_point = -128;
    layer_specs_seq[3].add_layer_scale_reciprocal = 1;
    layer_specs_seq[3].add_layer_zero_point = 0;
    layer_specs_seq[3].skip_connection_other_layer_scale = 1;
    layer_specs_seq[3].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[3].data_layout = CHW;



    layer_specs_seq[5].layer_index = 5;
    layer_specs_seq[5].conv_layer_type = DW_CONV; 
    layer_specs_seq[5].layer_num_fils = 64;
    layer_specs_seq[5].strides = 2;
    layer_specs_seq[5].filter_size = 3;
    layer_specs_seq[5].padding_left = 0;
    layer_specs_seq[5].padding_right = 1;
    layer_specs_seq[5].padding_top = 0;
    layer_specs_seq[5].padding_bottom = 1;
    layer_specs_seq[5].layer_depth = 64;
    layer_specs_seq[5].layer_ifm_height = 112;
    layer_specs_seq[5].layer_ifm_width = 112;
    layer_specs_seq[5].layer_ofm_height = 56;
    layer_specs_seq[5].layer_ofm_width = 56;
    layer_specs_seq[5].layer_activation = RELU6;
    layer_specs_seq[5].layer_num_of_ifm_tiles_h = (112 + TILE_H - 1) / TILE_H;
    layer_specs_seq[5].layer_num_of_ifm_tiles_w = (112 + TILE_W - 1) / TILE_W;
    layer_specs_seq[5].layer_num_of_ofm_tiles_h = (56 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(56 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 5;
    layer_specs_seq[5].layer_weights_offset = 512;
    layer_specs_seq[5].write_to_result_or_channels = 1;
    layer_specs_seq[5].write_to_tmp = 0;
    layer_specs_seq[5].followed_by = 0;
    layer_specs_seq[5].layer_ifms_zero_point = -128;
    layer_specs_seq[5].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[5].relu_threshold = 255 ;
    layer_specs_seq[5].layer_ofms_zero_point = -128;
    layer_specs_seq[5].add_layer_scale_reciprocal = 1;
    layer_specs_seq[5].add_layer_zero_point = 0;
    layer_specs_seq[5].skip_connection_other_layer_scale = 1;
    layer_specs_seq[5].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[5].data_layout = CHW;


    layer_specs_seq[6].layer_index = 6;
    layer_specs_seq[6].conv_layer_type = PW_CONV; 
    layer_specs_seq[6].layer_num_fils = 128;
    layer_specs_seq[6].strides = 1;
    layer_specs_seq[6].filter_size = 1;
    layer_specs_seq[6].padding_left = 0;
    layer_specs_seq[6].padding_right = 0;
    layer_specs_seq[6].padding_top = 0;
    layer_specs_seq[6].padding_bottom = 0;
    layer_specs_seq[6].layer_depth = 64;
    layer_specs_seq[6].layer_ifm_height = 56;
    layer_specs_seq[6].layer_ifm_width = 56;
    layer_specs_seq[6].layer_ofm_height = 56;
    layer_specs_seq[6].layer_ofm_width = 56;
    layer_specs_seq[6].layer_activation = RELU6;
    layer_specs_seq[6].layer_num_of_ifm_tiles_h = (56 + TILE_H - 1) / TILE_H;
    layer_specs_seq[6].layer_num_of_ifm_tiles_w = (56 + TILE_W - 1) / TILE_W;
    layer_specs_seq[6].layer_num_of_ofm_tiles_h = (56 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(56 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 6;
    layer_specs_seq[6].layer_weights_offset = 2048;
    layer_specs_seq[6].write_to_result_or_channels = 1;
    layer_specs_seq[6].write_to_tmp = 0;
    layer_specs_seq[6].followed_by = 0;
    layer_specs_seq[6].layer_ifms_zero_point = -128;
    layer_specs_seq[6].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[6].relu_threshold = 255 ;
    layer_specs_seq[6].layer_ofms_zero_point = -128;
    layer_specs_seq[6].add_layer_scale_reciprocal = 1;
    layer_specs_seq[6].add_layer_zero_point = 0;
    layer_specs_seq[6].skip_connection_other_layer_scale = 1;
    layer_specs_seq[6].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[6].data_layout = CHW;


    layer_specs_seq[7].layer_index = 7;
    layer_specs_seq[7].conv_layer_type = DW_CONV; 
    layer_specs_seq[7].layer_num_fils = 128;
    layer_specs_seq[7].strides = 1;
    layer_specs_seq[7].filter_size = 3;
    layer_specs_seq[7].padding_left = 1;
    layer_specs_seq[7].padding_right = 1;
    layer_specs_seq[7].padding_top = 1;
    layer_specs_seq[7].padding_bottom = 1;
    layer_specs_seq[7].layer_depth = 128;
    layer_specs_seq[7].layer_ifm_height = 56;
    layer_specs_seq[7].layer_ifm_width = 56;
    layer_specs_seq[7].layer_ofm_height = 56;
    layer_specs_seq[7].layer_ofm_width = 56;
    layer_specs_seq[7].layer_activation = RELU6;
    layer_specs_seq[7].layer_num_of_ifm_tiles_h = (56 + TILE_H - 1) / TILE_H;
    layer_specs_seq[7].layer_num_of_ifm_tiles_w = (56 + TILE_W - 1) / TILE_W;
    layer_specs_seq[7].layer_num_of_ofm_tiles_h = (56 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(56 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 7;
    layer_specs_seq[7].layer_weights_offset = 1536;
    layer_specs_seq[7].write_to_result_or_channels = 1;
    layer_specs_seq[7].write_to_tmp = 0;
    layer_specs_seq[7].followed_by = 0;
    layer_specs_seq[7].layer_ifms_zero_point = -128;
    layer_specs_seq[7].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[7].relu_threshold = 255 ;
    layer_specs_seq[7].layer_ofms_zero_point = -128;
    layer_specs_seq[7].add_layer_scale_reciprocal = 1;
    layer_specs_seq[7].add_layer_zero_point = 0;
    layer_specs_seq[7].skip_connection_other_layer_scale = 1;
    layer_specs_seq[7].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[7].data_layout = CHW;


    layer_specs_seq[8].layer_index = 8;
    layer_specs_seq[8].conv_layer_type = PW_CONV; 
    layer_specs_seq[8].layer_num_fils = 128;
    layer_specs_seq[8].strides = 1;
    layer_specs_seq[8].filter_size = 1;
    layer_specs_seq[8].padding_left = 0;
    layer_specs_seq[8].padding_right = 0;
    layer_specs_seq[8].padding_top = 0;
    layer_specs_seq[8].padding_bottom = 0;
    layer_specs_seq[8].layer_depth = 128;
    layer_specs_seq[8].layer_ifm_height = 56;
    layer_specs_seq[8].layer_ifm_width = 56;
    layer_specs_seq[8].layer_ofm_height = 56;
    layer_specs_seq[8].layer_ofm_width = 56;
    layer_specs_seq[8].layer_activation = RELU6;
    layer_specs_seq[8].layer_num_of_ifm_tiles_h = (56 + TILE_H - 1) / TILE_H;
    layer_specs_seq[8].layer_num_of_ifm_tiles_w = (56 + TILE_W - 1) / TILE_W;
    layer_specs_seq[8].layer_num_of_ofm_tiles_h = (56 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(56 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 8;
    layer_specs_seq[8].layer_weights_offset = 10240;
    layer_specs_seq[8].write_to_result_or_channels = 1;
    layer_specs_seq[8].write_to_tmp = 0;
    layer_specs_seq[8].followed_by = 0;
    layer_specs_seq[8].layer_ifms_zero_point = -128;
    layer_specs_seq[8].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[8].relu_threshold = 255 ;
    layer_specs_seq[8].layer_ofms_zero_point = -128;
    layer_specs_seq[8].add_layer_scale_reciprocal = 1;
    layer_specs_seq[8].add_layer_zero_point = 0;
    layer_specs_seq[8].skip_connection_other_layer_scale = 1;
    layer_specs_seq[8].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[8].data_layout = CHW;



    layer_specs_seq[10].layer_index = 10;
    layer_specs_seq[10].conv_layer_type = DW_CONV; 
    layer_specs_seq[10].layer_num_fils = 128;
    layer_specs_seq[10].strides = 2;
    layer_specs_seq[10].filter_size = 3;
    layer_specs_seq[10].padding_left = 0;
    layer_specs_seq[10].padding_right = 1;
    layer_specs_seq[10].padding_top = 0;
    layer_specs_seq[10].padding_bottom = 1;
    layer_specs_seq[10].layer_depth = 128;
    layer_specs_seq[10].layer_ifm_height = 56;
    layer_specs_seq[10].layer_ifm_width = 56;
    layer_specs_seq[10].layer_ofm_height = 28;
    layer_specs_seq[10].layer_ofm_width = 28;
    layer_specs_seq[10].layer_activation = RELU6;
    layer_specs_seq[10].layer_num_of_ifm_tiles_h = (56 + TILE_H - 1) / TILE_H;
    layer_specs_seq[10].layer_num_of_ifm_tiles_w = (56 + TILE_W - 1) / TILE_W;
    layer_specs_seq[10].layer_num_of_ofm_tiles_h = (28 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(28 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 10;
    layer_specs_seq[10].layer_weights_offset = 3584;
    layer_specs_seq[10].write_to_result_or_channels = 1;
    layer_specs_seq[10].write_to_tmp = 0;
    layer_specs_seq[10].followed_by = 0;
    layer_specs_seq[10].layer_ifms_zero_point = -128;
    layer_specs_seq[10].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[10].relu_threshold = 255 ;
    layer_specs_seq[10].layer_ofms_zero_point = -128;
    layer_specs_seq[10].add_layer_scale_reciprocal = 1;
    layer_specs_seq[10].add_layer_zero_point = 0;
    layer_specs_seq[10].skip_connection_other_layer_scale = 1;
    layer_specs_seq[10].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[10].data_layout = CHW;


    layer_specs_seq[11].layer_index = 11;
    layer_specs_seq[11].conv_layer_type = PW_CONV; 
    layer_specs_seq[11].layer_num_fils = 256;
    layer_specs_seq[11].strides = 1;
    layer_specs_seq[11].filter_size = 1;
    layer_specs_seq[11].padding_left = 0;
    layer_specs_seq[11].padding_right = 0;
    layer_specs_seq[11].padding_top = 0;
    layer_specs_seq[11].padding_bottom = 0;
    layer_specs_seq[11].layer_depth = 128;
    layer_specs_seq[11].layer_ifm_height = 28;
    layer_specs_seq[11].layer_ifm_width = 28;
    layer_specs_seq[11].layer_ofm_height = 28;
    layer_specs_seq[11].layer_ofm_width = 28;
    layer_specs_seq[11].layer_activation = RELU6;
    layer_specs_seq[11].layer_num_of_ifm_tiles_h = (28 + TILE_H - 1) / TILE_H;
    layer_specs_seq[11].layer_num_of_ifm_tiles_w = (28 + TILE_W - 1) / TILE_W;
    layer_specs_seq[11].layer_num_of_ofm_tiles_h = (28 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(28 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 11;
    layer_specs_seq[11].layer_weights_offset = 26624;
    layer_specs_seq[11].write_to_result_or_channels = 1;
    layer_specs_seq[11].write_to_tmp = 0;
    layer_specs_seq[11].followed_by = 0;
    layer_specs_seq[11].layer_ifms_zero_point = -128;
    layer_specs_seq[11].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[11].relu_threshold = 255 ;
    layer_specs_seq[11].layer_ofms_zero_point = -128;
    layer_specs_seq[11].add_layer_scale_reciprocal = 1;
    layer_specs_seq[11].add_layer_zero_point = 0;
    layer_specs_seq[11].skip_connection_other_layer_scale = 1;
    layer_specs_seq[11].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[11].data_layout = CHW;


    layer_specs_seq[12].layer_index = 12;
    layer_specs_seq[12].conv_layer_type = DW_CONV; 
    layer_specs_seq[12].layer_num_fils = 256;
    layer_specs_seq[12].strides = 1;
    layer_specs_seq[12].filter_size = 3;
    layer_specs_seq[12].padding_left = 1;
    layer_specs_seq[12].padding_right = 1;
    layer_specs_seq[12].padding_top = 1;
    layer_specs_seq[12].padding_bottom = 1;
    layer_specs_seq[12].layer_depth = 256;
    layer_specs_seq[12].layer_ifm_height = 28;
    layer_specs_seq[12].layer_ifm_width = 28;
    layer_specs_seq[12].layer_ofm_height = 28;
    layer_specs_seq[12].layer_ofm_width = 28;
    layer_specs_seq[12].layer_activation = RELU6;
    layer_specs_seq[12].layer_num_of_ifm_tiles_h = (28 + TILE_H - 1) / TILE_H;
    layer_specs_seq[12].layer_num_of_ifm_tiles_w = (28 + TILE_W - 1) / TILE_W;
    layer_specs_seq[12].layer_num_of_ofm_tiles_h = (28 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(28 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 12;
    layer_specs_seq[12].layer_weights_offset = 5632;
    layer_specs_seq[12].write_to_result_or_channels = 1;
    layer_specs_seq[12].write_to_tmp = 0;
    layer_specs_seq[12].followed_by = 0;
    layer_specs_seq[12].layer_ifms_zero_point = -128;
    layer_specs_seq[12].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[12].relu_threshold = 255 ;
    layer_specs_seq[12].layer_ofms_zero_point = -128;
    layer_specs_seq[12].add_layer_scale_reciprocal = 1;
    layer_specs_seq[12].add_layer_zero_point = 0;
    layer_specs_seq[12].skip_connection_other_layer_scale = 1;
    layer_specs_seq[12].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[12].data_layout = CHW;


    layer_specs_seq[13].layer_index = 13;
    layer_specs_seq[13].conv_layer_type = PW_CONV; 
    layer_specs_seq[13].layer_num_fils = 256;
    layer_specs_seq[13].strides = 1;
    layer_specs_seq[13].filter_size = 1;
    layer_specs_seq[13].padding_left = 0;
    layer_specs_seq[13].padding_right = 0;
    layer_specs_seq[13].padding_top = 0;
    layer_specs_seq[13].padding_bottom = 0;
    layer_specs_seq[13].layer_depth = 256;
    layer_specs_seq[13].layer_ifm_height = 28;
    layer_specs_seq[13].layer_ifm_width = 28;
    layer_specs_seq[13].layer_ofm_height = 28;
    layer_specs_seq[13].layer_ofm_width = 28;
    layer_specs_seq[13].layer_activation = RELU6;
    layer_specs_seq[13].layer_num_of_ifm_tiles_h = (28 + TILE_H - 1) / TILE_H;
    layer_specs_seq[13].layer_num_of_ifm_tiles_w = (28 + TILE_W - 1) / TILE_W;
    layer_specs_seq[13].layer_num_of_ofm_tiles_h = (28 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(28 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 13;
    layer_specs_seq[13].layer_weights_offset = 59392;
    layer_specs_seq[13].write_to_result_or_channels = 1;
    layer_specs_seq[13].write_to_tmp = 0;
    layer_specs_seq[13].followed_by = 0;
    layer_specs_seq[13].layer_ifms_zero_point = -128;
    layer_specs_seq[13].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[13].relu_threshold = 255 ;
    layer_specs_seq[13].layer_ofms_zero_point = -128;
    layer_specs_seq[13].add_layer_scale_reciprocal = 1;
    layer_specs_seq[13].add_layer_zero_point = 0;
    layer_specs_seq[13].skip_connection_other_layer_scale = 1;
    layer_specs_seq[13].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[13].data_layout = CHW;



    layer_specs_seq[15].layer_index = 15;
    layer_specs_seq[15].conv_layer_type = DW_CONV; 
    layer_specs_seq[15].layer_num_fils = 256;
    layer_specs_seq[15].strides = 2;
    layer_specs_seq[15].filter_size = 3;
    layer_specs_seq[15].padding_left = 0;
    layer_specs_seq[15].padding_right = 1;
    layer_specs_seq[15].padding_top = 0;
    layer_specs_seq[15].padding_bottom = 1;
    layer_specs_seq[15].layer_depth = 256;
    layer_specs_seq[15].layer_ifm_height = 28;
    layer_specs_seq[15].layer_ifm_width = 28;
    layer_specs_seq[15].layer_ofm_height = 14;
    layer_specs_seq[15].layer_ofm_width = 14;
    layer_specs_seq[15].layer_activation = RELU6;
    layer_specs_seq[15].layer_num_of_ifm_tiles_h = (28 + TILE_H - 1) / TILE_H;
    layer_specs_seq[15].layer_num_of_ifm_tiles_w = (28 + TILE_W - 1) / TILE_W;
    layer_specs_seq[15].layer_num_of_ofm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(14 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 15;
    layer_specs_seq[15].layer_weights_offset = 9728;
    layer_specs_seq[15].write_to_result_or_channels = 1;
    layer_specs_seq[15].write_to_tmp = 0;
    layer_specs_seq[15].followed_by = 0;
    layer_specs_seq[15].layer_ifms_zero_point = -128;
    layer_specs_seq[15].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[15].relu_threshold = 255 ;
    layer_specs_seq[15].layer_ofms_zero_point = -128;
    layer_specs_seq[15].add_layer_scale_reciprocal = 1;
    layer_specs_seq[15].add_layer_zero_point = 0;
    layer_specs_seq[15].skip_connection_other_layer_scale = 1;
    layer_specs_seq[15].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[15].data_layout = CHW;


    layer_specs_seq[16].layer_index = 16;
    layer_specs_seq[16].conv_layer_type = PW_CONV; 
    layer_specs_seq[16].layer_num_fils = 512;
    layer_specs_seq[16].strides = 1;
    layer_specs_seq[16].filter_size = 1;
    layer_specs_seq[16].padding_left = 0;
    layer_specs_seq[16].padding_right = 0;
    layer_specs_seq[16].padding_top = 0;
    layer_specs_seq[16].padding_bottom = 0;
    layer_specs_seq[16].layer_depth = 256;
    layer_specs_seq[16].layer_ifm_height = 14;
    layer_specs_seq[16].layer_ifm_width = 14;
    layer_specs_seq[16].layer_ofm_height = 14;
    layer_specs_seq[16].layer_ofm_width = 14;
    layer_specs_seq[16].layer_activation = RELU6;
    layer_specs_seq[16].layer_num_of_ifm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[16].layer_num_of_ifm_tiles_w = (14 + TILE_W - 1) / TILE_W;
    layer_specs_seq[16].layer_num_of_ofm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(14 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 16;
    layer_specs_seq[16].layer_weights_offset = 124928;
    layer_specs_seq[16].write_to_result_or_channels = 1;
    layer_specs_seq[16].write_to_tmp = 0;
    layer_specs_seq[16].followed_by = 0;
    layer_specs_seq[16].layer_ifms_zero_point = -128;
    layer_specs_seq[16].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[16].relu_threshold = 255 ;
    layer_specs_seq[16].layer_ofms_zero_point = -128;
    layer_specs_seq[16].add_layer_scale_reciprocal = 1;
    layer_specs_seq[16].add_layer_zero_point = 0;
    layer_specs_seq[16].skip_connection_other_layer_scale = 1;
    layer_specs_seq[16].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[16].data_layout = CHW;


    layer_specs_seq[17].layer_index = 17;
    layer_specs_seq[17].conv_layer_type = DW_CONV; 
    layer_specs_seq[17].layer_num_fils = 512;
    layer_specs_seq[17].strides = 1;
    layer_specs_seq[17].filter_size = 3;
    layer_specs_seq[17].padding_left = 1;
    layer_specs_seq[17].padding_right = 1;
    layer_specs_seq[17].padding_top = 1;
    layer_specs_seq[17].padding_bottom = 1;
    layer_specs_seq[17].layer_depth = 512;
    layer_specs_seq[17].layer_ifm_height = 14;
    layer_specs_seq[17].layer_ifm_width = 14;
    layer_specs_seq[17].layer_ofm_height = 14;
    layer_specs_seq[17].layer_ofm_width = 14;
    layer_specs_seq[17].layer_activation = RELU6;
    layer_specs_seq[17].layer_num_of_ifm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[17].layer_num_of_ifm_tiles_w = (14 + TILE_W - 1) / TILE_W;
    layer_specs_seq[17].layer_num_of_ofm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(14 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 17;
    layer_specs_seq[17].layer_weights_offset = 13824;
    layer_specs_seq[17].write_to_result_or_channels = 1;
    layer_specs_seq[17].write_to_tmp = 0;
    layer_specs_seq[17].followed_by = 0;
    layer_specs_seq[17].layer_ifms_zero_point = -128;
    layer_specs_seq[17].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[17].relu_threshold = 255 ;
    layer_specs_seq[17].layer_ofms_zero_point = -128;
    layer_specs_seq[17].add_layer_scale_reciprocal = 1;
    layer_specs_seq[17].add_layer_zero_point = 0;
    layer_specs_seq[17].skip_connection_other_layer_scale = 1;
    layer_specs_seq[17].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[17].data_layout = CHW;


    layer_specs_seq[18].layer_index = 18;
    layer_specs_seq[18].conv_layer_type = PW_CONV; 
    layer_specs_seq[18].layer_num_fils = 512;
    layer_specs_seq[18].strides = 1;
    layer_specs_seq[18].filter_size = 1;
    layer_specs_seq[18].padding_left = 0;
    layer_specs_seq[18].padding_right = 0;
    layer_specs_seq[18].padding_top = 0;
    layer_specs_seq[18].padding_bottom = 0;
    layer_specs_seq[18].layer_depth = 512;
    layer_specs_seq[18].layer_ifm_height = 14;
    layer_specs_seq[18].layer_ifm_width = 14;
    layer_specs_seq[18].layer_ofm_height = 14;
    layer_specs_seq[18].layer_ofm_width = 14;
    layer_specs_seq[18].layer_activation = RELU6;
    layer_specs_seq[18].layer_num_of_ifm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[18].layer_num_of_ifm_tiles_w = (14 + TILE_W - 1) / TILE_W;
    layer_specs_seq[18].layer_num_of_ofm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(14 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 18;
    layer_specs_seq[18].layer_weights_offset = 256000;
    layer_specs_seq[18].write_to_result_or_channels = 1;
    layer_specs_seq[18].write_to_tmp = 0;
    layer_specs_seq[18].followed_by = 0;
    layer_specs_seq[18].layer_ifms_zero_point = -128;
    layer_specs_seq[18].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[18].relu_threshold = 255 ;
    layer_specs_seq[18].layer_ofms_zero_point = -128;
    layer_specs_seq[18].add_layer_scale_reciprocal = 1;
    layer_specs_seq[18].add_layer_zero_point = 0;
    layer_specs_seq[18].skip_connection_other_layer_scale = 1;
    layer_specs_seq[18].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[18].data_layout = CHW;


    layer_specs_seq[19].layer_index = 19;
    layer_specs_seq[19].conv_layer_type = DW_CONV; 
    layer_specs_seq[19].layer_num_fils = 512;
    layer_specs_seq[19].strides = 1;
    layer_specs_seq[19].filter_size = 3;
    layer_specs_seq[19].padding_left = 1;
    layer_specs_seq[19].padding_right = 1;
    layer_specs_seq[19].padding_top = 1;
    layer_specs_seq[19].padding_bottom = 1;
    layer_specs_seq[19].layer_depth = 512;
    layer_specs_seq[19].layer_ifm_height = 14;
    layer_specs_seq[19].layer_ifm_width = 14;
    layer_specs_seq[19].layer_ofm_height = 14;
    layer_specs_seq[19].layer_ofm_width = 14;
    layer_specs_seq[19].layer_activation = RELU6;
    layer_specs_seq[19].layer_num_of_ifm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[19].layer_num_of_ifm_tiles_w = (14 + TILE_W - 1) / TILE_W;
    layer_specs_seq[19].layer_num_of_ofm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(14 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 19;
    layer_specs_seq[19].layer_weights_offset = 22016;
    layer_specs_seq[19].write_to_result_or_channels = 1;
    layer_specs_seq[19].write_to_tmp = 0;
    layer_specs_seq[19].followed_by = 0;
    layer_specs_seq[19].layer_ifms_zero_point = -128;
    layer_specs_seq[19].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[19].relu_threshold = 255 ;
    layer_specs_seq[19].layer_ofms_zero_point = -128;
    layer_specs_seq[19].add_layer_scale_reciprocal = 1;
    layer_specs_seq[19].add_layer_zero_point = 0;
    layer_specs_seq[19].skip_connection_other_layer_scale = 1;
    layer_specs_seq[19].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[19].data_layout = CHW;


    layer_specs_seq[20].layer_index = 20;
    layer_specs_seq[20].conv_layer_type = PW_CONV; 
    layer_specs_seq[20].layer_num_fils = 512;
    layer_specs_seq[20].strides = 1;
    layer_specs_seq[20].filter_size = 1;
    layer_specs_seq[20].padding_left = 0;
    layer_specs_seq[20].padding_right = 0;
    layer_specs_seq[20].padding_top = 0;
    layer_specs_seq[20].padding_bottom = 0;
    layer_specs_seq[20].layer_depth = 512;
    layer_specs_seq[20].layer_ifm_height = 14;
    layer_specs_seq[20].layer_ifm_width = 14;
    layer_specs_seq[20].layer_ofm_height = 14;
    layer_specs_seq[20].layer_ofm_width = 14;
    layer_specs_seq[20].layer_activation = RELU6;
    layer_specs_seq[20].layer_num_of_ifm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[20].layer_num_of_ifm_tiles_w = (14 + TILE_W - 1) / TILE_W;
    layer_specs_seq[20].layer_num_of_ofm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(14 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 20;
    layer_specs_seq[20].layer_weights_offset = 518144;
    layer_specs_seq[20].write_to_result_or_channels = 1;
    layer_specs_seq[20].write_to_tmp = 0;
    layer_specs_seq[20].followed_by = 0;
    layer_specs_seq[20].layer_ifms_zero_point = -128;
    layer_specs_seq[20].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[20].relu_threshold = 255 ;
    layer_specs_seq[20].layer_ofms_zero_point = -128;
    layer_specs_seq[20].add_layer_scale_reciprocal = 1;
    layer_specs_seq[20].add_layer_zero_point = 0;
    layer_specs_seq[20].skip_connection_other_layer_scale = 1;
    layer_specs_seq[20].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[20].data_layout = CHW;


    layer_specs_seq[21].layer_index = 21;
    layer_specs_seq[21].conv_layer_type = DW_CONV; 
    layer_specs_seq[21].layer_num_fils = 512;
    layer_specs_seq[21].strides = 1;
    layer_specs_seq[21].filter_size = 3;
    layer_specs_seq[21].padding_left = 1;
    layer_specs_seq[21].padding_right = 1;
    layer_specs_seq[21].padding_top = 1;
    layer_specs_seq[21].padding_bottom = 1;
    layer_specs_seq[21].layer_depth = 512;
    layer_specs_seq[21].layer_ifm_height = 14;
    layer_specs_seq[21].layer_ifm_width = 14;
    layer_specs_seq[21].layer_ofm_height = 14;
    layer_specs_seq[21].layer_ofm_width = 14;
    layer_specs_seq[21].layer_activation = RELU6;
    layer_specs_seq[21].layer_num_of_ifm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[21].layer_num_of_ifm_tiles_w = (14 + TILE_W - 1) / TILE_W;
    layer_specs_seq[21].layer_num_of_ofm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(14 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 21;
    layer_specs_seq[21].layer_weights_offset = 30208;
    layer_specs_seq[21].write_to_result_or_channels = 1;
    layer_specs_seq[21].write_to_tmp = 0;
    layer_specs_seq[21].followed_by = 0;
    layer_specs_seq[21].layer_ifms_zero_point = -128;
    layer_specs_seq[21].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[21].relu_threshold = 255 ;
    layer_specs_seq[21].layer_ofms_zero_point = -128;
    layer_specs_seq[21].add_layer_scale_reciprocal = 1;
    layer_specs_seq[21].add_layer_zero_point = 0;
    layer_specs_seq[21].skip_connection_other_layer_scale = 1;
    layer_specs_seq[21].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[21].data_layout = CHW;


    layer_specs_seq[22].layer_index = 22;
    layer_specs_seq[22].conv_layer_type = PW_CONV; 
    layer_specs_seq[22].layer_num_fils = 512;
    layer_specs_seq[22].strides = 1;
    layer_specs_seq[22].filter_size = 1;
    layer_specs_seq[22].padding_left = 0;
    layer_specs_seq[22].padding_right = 0;
    layer_specs_seq[22].padding_top = 0;
    layer_specs_seq[22].padding_bottom = 0;
    layer_specs_seq[22].layer_depth = 512;
    layer_specs_seq[22].layer_ifm_height = 14;
    layer_specs_seq[22].layer_ifm_width = 14;
    layer_specs_seq[22].layer_ofm_height = 14;
    layer_specs_seq[22].layer_ofm_width = 14;
    layer_specs_seq[22].layer_activation = RELU6;
    layer_specs_seq[22].layer_num_of_ifm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[22].layer_num_of_ifm_tiles_w = (14 + TILE_W - 1) / TILE_W;
    layer_specs_seq[22].layer_num_of_ofm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(14 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 22;
    layer_specs_seq[22].layer_weights_offset = 780288;
    layer_specs_seq[22].write_to_result_or_channels = 1;
    layer_specs_seq[22].write_to_tmp = 0;
    layer_specs_seq[22].followed_by = 0;
    layer_specs_seq[22].layer_ifms_zero_point = -128;
    layer_specs_seq[22].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[22].relu_threshold = 255 ;
    layer_specs_seq[22].layer_ofms_zero_point = -128;
    layer_specs_seq[22].add_layer_scale_reciprocal = 1;
    layer_specs_seq[22].add_layer_zero_point = 0;
    layer_specs_seq[22].skip_connection_other_layer_scale = 1;
    layer_specs_seq[22].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[22].data_layout = CHW;


    layer_specs_seq[23].layer_index = 23;
    layer_specs_seq[23].conv_layer_type = DW_CONV; 
    layer_specs_seq[23].layer_num_fils = 512;
    layer_specs_seq[23].strides = 1;
    layer_specs_seq[23].filter_size = 3;
    layer_specs_seq[23].padding_left = 1;
    layer_specs_seq[23].padding_right = 1;
    layer_specs_seq[23].padding_top = 1;
    layer_specs_seq[23].padding_bottom = 1;
    layer_specs_seq[23].layer_depth = 512;
    layer_specs_seq[23].layer_ifm_height = 14;
    layer_specs_seq[23].layer_ifm_width = 14;
    layer_specs_seq[23].layer_ofm_height = 14;
    layer_specs_seq[23].layer_ofm_width = 14;
    layer_specs_seq[23].layer_activation = RELU6;
    layer_specs_seq[23].layer_num_of_ifm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[23].layer_num_of_ifm_tiles_w = (14 + TILE_W - 1) / TILE_W;
    layer_specs_seq[23].layer_num_of_ofm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(14 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 23;
    layer_specs_seq[23].layer_weights_offset = 38400;
    layer_specs_seq[23].write_to_result_or_channels = 1;
    layer_specs_seq[23].write_to_tmp = 0;
    layer_specs_seq[23].followed_by = 0;
    layer_specs_seq[23].layer_ifms_zero_point = -128;
    layer_specs_seq[23].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[23].relu_threshold = 255 ;
    layer_specs_seq[23].layer_ofms_zero_point = -128;
    layer_specs_seq[23].add_layer_scale_reciprocal = 1;
    layer_specs_seq[23].add_layer_zero_point = 0;
    layer_specs_seq[23].skip_connection_other_layer_scale = 1;
    layer_specs_seq[23].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[23].data_layout = CHW;


    layer_specs_seq[24].layer_index = 24;
    layer_specs_seq[24].conv_layer_type = PW_CONV; 
    layer_specs_seq[24].layer_num_fils = 512;
    layer_specs_seq[24].strides = 1;
    layer_specs_seq[24].filter_size = 1;
    layer_specs_seq[24].padding_left = 0;
    layer_specs_seq[24].padding_right = 0;
    layer_specs_seq[24].padding_top = 0;
    layer_specs_seq[24].padding_bottom = 0;
    layer_specs_seq[24].layer_depth = 512;
    layer_specs_seq[24].layer_ifm_height = 14;
    layer_specs_seq[24].layer_ifm_width = 14;
    layer_specs_seq[24].layer_ofm_height = 14;
    layer_specs_seq[24].layer_ofm_width = 14;
    layer_specs_seq[24].layer_activation = RELU6;
    layer_specs_seq[24].layer_num_of_ifm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[24].layer_num_of_ifm_tiles_w = (14 + TILE_W - 1) / TILE_W;
    layer_specs_seq[24].layer_num_of_ofm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(14 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 24;
    layer_specs_seq[24].layer_weights_offset = 1042432;
    layer_specs_seq[24].write_to_result_or_channels = 1;
    layer_specs_seq[24].write_to_tmp = 0;
    layer_specs_seq[24].followed_by = 0;
    layer_specs_seq[24].layer_ifms_zero_point = -128;
    layer_specs_seq[24].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[24].relu_threshold = 255 ;
    layer_specs_seq[24].layer_ofms_zero_point = -128;
    layer_specs_seq[24].add_layer_scale_reciprocal = 1;
    layer_specs_seq[24].add_layer_zero_point = 0;
    layer_specs_seq[24].skip_connection_other_layer_scale = 1;
    layer_specs_seq[24].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[24].data_layout = CHW;


    layer_specs_seq[25].layer_index = 25;
    layer_specs_seq[25].conv_layer_type = DW_CONV; 
    layer_specs_seq[25].layer_num_fils = 512;
    layer_specs_seq[25].strides = 1;
    layer_specs_seq[25].filter_size = 3;
    layer_specs_seq[25].padding_left = 1;
    layer_specs_seq[25].padding_right = 1;
    layer_specs_seq[25].padding_top = 1;
    layer_specs_seq[25].padding_bottom = 1;
    layer_specs_seq[25].layer_depth = 512;
    layer_specs_seq[25].layer_ifm_height = 14;
    layer_specs_seq[25].layer_ifm_width = 14;
    layer_specs_seq[25].layer_ofm_height = 14;
    layer_specs_seq[25].layer_ofm_width = 14;
    layer_specs_seq[25].layer_activation = RELU6;
    layer_specs_seq[25].layer_num_of_ifm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[25].layer_num_of_ifm_tiles_w = (14 + TILE_W - 1) / TILE_W;
    layer_specs_seq[25].layer_num_of_ofm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(14 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 25;
    layer_specs_seq[25].layer_weights_offset = 46592;
    layer_specs_seq[25].write_to_result_or_channels = 1;
    layer_specs_seq[25].write_to_tmp = 0;
    layer_specs_seq[25].followed_by = 0;
    layer_specs_seq[25].layer_ifms_zero_point = -128;
    layer_specs_seq[25].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[25].relu_threshold = 255 ;
    layer_specs_seq[25].layer_ofms_zero_point = -128;
    layer_specs_seq[25].add_layer_scale_reciprocal = 1;
    layer_specs_seq[25].add_layer_zero_point = 0;
    layer_specs_seq[25].skip_connection_other_layer_scale = 1;
    layer_specs_seq[25].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[25].data_layout = CHW;


    layer_specs_seq[26].layer_index = 26;
    layer_specs_seq[26].conv_layer_type = PW_CONV; 
    layer_specs_seq[26].layer_num_fils = 512;
    layer_specs_seq[26].strides = 1;
    layer_specs_seq[26].filter_size = 1;
    layer_specs_seq[26].padding_left = 0;
    layer_specs_seq[26].padding_right = 0;
    layer_specs_seq[26].padding_top = 0;
    layer_specs_seq[26].padding_bottom = 0;
    layer_specs_seq[26].layer_depth = 512;
    layer_specs_seq[26].layer_ifm_height = 14;
    layer_specs_seq[26].layer_ifm_width = 14;
    layer_specs_seq[26].layer_ofm_height = 14;
    layer_specs_seq[26].layer_ofm_width = 14;
    layer_specs_seq[26].layer_activation = RELU6;
    layer_specs_seq[26].layer_num_of_ifm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[26].layer_num_of_ifm_tiles_w = (14 + TILE_W - 1) / TILE_W;
    layer_specs_seq[26].layer_num_of_ofm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(14 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 26;
    layer_specs_seq[26].layer_weights_offset = 1304576;
    layer_specs_seq[26].write_to_result_or_channels = 1;
    layer_specs_seq[26].write_to_tmp = 0;
    layer_specs_seq[26].followed_by = 0;
    layer_specs_seq[26].layer_ifms_zero_point = -128;
    layer_specs_seq[26].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[26].relu_threshold = 255 ;
    layer_specs_seq[26].layer_ofms_zero_point = -128;
    layer_specs_seq[26].add_layer_scale_reciprocal = 1;
    layer_specs_seq[26].add_layer_zero_point = 0;
    layer_specs_seq[26].skip_connection_other_layer_scale = 1;
    layer_specs_seq[26].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[26].data_layout = CHW;



    layer_specs_seq[28].layer_index = 28;
    layer_specs_seq[28].conv_layer_type = DW_CONV; 
    layer_specs_seq[28].layer_num_fils = 512;
    layer_specs_seq[28].strides = 2;
    layer_specs_seq[28].filter_size = 3;
    layer_specs_seq[28].padding_left = 0;
    layer_specs_seq[28].padding_right = 1;
    layer_specs_seq[28].padding_top = 0;
    layer_specs_seq[28].padding_bottom = 1;
    layer_specs_seq[28].layer_depth = 512;
    layer_specs_seq[28].layer_ifm_height = 14;
    layer_specs_seq[28].layer_ifm_width = 14;
    layer_specs_seq[28].layer_ofm_height = 7;
    layer_specs_seq[28].layer_ofm_width = 7;
    layer_specs_seq[28].layer_activation = RELU6;
    layer_specs_seq[28].layer_num_of_ifm_tiles_h = (14 + TILE_H - 1) / TILE_H;
    layer_specs_seq[28].layer_num_of_ifm_tiles_w = (14 + TILE_W - 1) / TILE_W;
    layer_specs_seq[28].layer_num_of_ofm_tiles_h = (7 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(7 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 28;
    layer_specs_seq[28].layer_weights_offset = 54784;
    layer_specs_seq[28].write_to_result_or_channels = 1;
    layer_specs_seq[28].write_to_tmp = 0;
    layer_specs_seq[28].followed_by = 0;
    layer_specs_seq[28].layer_ifms_zero_point = -128;
    layer_specs_seq[28].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[28].relu_threshold = 255 ;
    layer_specs_seq[28].layer_ofms_zero_point = -128;
    layer_specs_seq[28].add_layer_scale_reciprocal = 1;
    layer_specs_seq[28].add_layer_zero_point = 0;
    layer_specs_seq[28].skip_connection_other_layer_scale = 1;
    layer_specs_seq[28].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[28].data_layout = CHW;


    layer_specs_seq[29].layer_index = 29;
    layer_specs_seq[29].conv_layer_type = PW_CONV; 
    layer_specs_seq[29].layer_num_fils = 1024;
    layer_specs_seq[29].strides = 1;
    layer_specs_seq[29].filter_size = 1;
    layer_specs_seq[29].padding_left = 0;
    layer_specs_seq[29].padding_right = 0;
    layer_specs_seq[29].padding_top = 0;
    layer_specs_seq[29].padding_bottom = 0;
    layer_specs_seq[29].layer_depth = 512;
    layer_specs_seq[29].layer_ifm_height = 7;
    layer_specs_seq[29].layer_ifm_width = 7;
    layer_specs_seq[29].layer_ofm_height = 7;
    layer_specs_seq[29].layer_ofm_width = 7;
    layer_specs_seq[29].layer_activation = RELU6;
    layer_specs_seq[29].layer_num_of_ifm_tiles_h = (7 + TILE_H - 1) / TILE_H;
    layer_specs_seq[29].layer_num_of_ifm_tiles_w = (7 + TILE_W - 1) / TILE_W;
    layer_specs_seq[29].layer_num_of_ofm_tiles_h = (7 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(7 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 29;
    layer_specs_seq[29].layer_weights_offset = 1566720;
    layer_specs_seq[29].write_to_result_or_channels = 1;
    layer_specs_seq[29].write_to_tmp = 0;
    layer_specs_seq[29].followed_by = 0;
    layer_specs_seq[29].layer_ifms_zero_point = -128;
    layer_specs_seq[29].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[29].relu_threshold = 255 ;
    layer_specs_seq[29].layer_ofms_zero_point = -128;
    layer_specs_seq[29].add_layer_scale_reciprocal = 1;
    layer_specs_seq[29].add_layer_zero_point = 0;
    layer_specs_seq[29].skip_connection_other_layer_scale = 1;
    layer_specs_seq[29].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[29].data_layout = CHW;


    layer_specs_seq[30].layer_index = 30;
    layer_specs_seq[30].conv_layer_type = DW_CONV; 
    layer_specs_seq[30].layer_num_fils = 1024;
    layer_specs_seq[30].strides = 1;
    layer_specs_seq[30].filter_size = 3;
    layer_specs_seq[30].padding_left = 1;
    layer_specs_seq[30].padding_right = 1;
    layer_specs_seq[30].padding_top = 1;
    layer_specs_seq[30].padding_bottom = 1;
    layer_specs_seq[30].layer_depth = 1024;
    layer_specs_seq[30].layer_ifm_height = 7;
    layer_specs_seq[30].layer_ifm_width = 7;
    layer_specs_seq[30].layer_ofm_height = 7;
    layer_specs_seq[30].layer_ofm_width = 7;
    layer_specs_seq[30].layer_activation = RELU6;
    layer_specs_seq[30].layer_num_of_ifm_tiles_h = (7 + TILE_H - 1) / TILE_H;
    layer_specs_seq[30].layer_num_of_ifm_tiles_w = (7 + TILE_W - 1) / TILE_W;
    layer_specs_seq[30].layer_num_of_ofm_tiles_h = (7 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(7 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 30;
    layer_specs_seq[30].layer_weights_offset = 62976;
    layer_specs_seq[30].write_to_result_or_channels = 1;
    layer_specs_seq[30].write_to_tmp = 0;
    layer_specs_seq[30].followed_by = 0;
    layer_specs_seq[30].layer_ifms_zero_point = -128;
    layer_specs_seq[30].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[30].relu_threshold = 255 ;
    layer_specs_seq[30].layer_ofms_zero_point = -128;
    layer_specs_seq[30].add_layer_scale_reciprocal = 1;
    layer_specs_seq[30].add_layer_zero_point = 0;
    layer_specs_seq[30].skip_connection_other_layer_scale = 1;
    layer_specs_seq[30].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[30].data_layout = CHW;


    layer_specs_seq[31].layer_index = 31;
    layer_specs_seq[31].conv_layer_type = PW_CONV; 
    layer_specs_seq[31].layer_num_fils = 1024;
    layer_specs_seq[31].strides = 1;
    layer_specs_seq[31].filter_size = 1;
    layer_specs_seq[31].padding_left = 0;
    layer_specs_seq[31].padding_right = 0;
    layer_specs_seq[31].padding_top = 0;
    layer_specs_seq[31].padding_bottom = 0;
    layer_specs_seq[31].layer_depth = 1024;
    layer_specs_seq[31].layer_ifm_height = 7;
    layer_specs_seq[31].layer_ifm_width = 7;
    layer_specs_seq[31].layer_ofm_height = 7;
    layer_specs_seq[31].layer_ofm_width = 7;
    layer_specs_seq[31].layer_activation = RELU6;
    layer_specs_seq[31].layer_num_of_ifm_tiles_h = (7 + TILE_H - 1) / TILE_H;
    layer_specs_seq[31].layer_num_of_ifm_tiles_w = (7 + TILE_W - 1) / TILE_W;
    layer_specs_seq[31].layer_num_of_ofm_tiles_h = (7 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(7 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 31;
    layer_specs_seq[31].layer_weights_offset = 2091008;
    layer_specs_seq[31].write_to_result_or_channels = 1;
    layer_specs_seq[31].write_to_tmp = 0;
    layer_specs_seq[31].followed_by = 4;
    layer_specs_seq[31].layer_ifms_zero_point = -128;
    layer_specs_seq[31].layer_ofms_scale = 0.0235294122248888;
    layer_specs_seq[31].relu_threshold = 255 ;
    layer_specs_seq[31].layer_ofms_zero_point = -128;
    layer_specs_seq[31].add_layer_scale_reciprocal = 1;
    layer_specs_seq[31].add_layer_zero_point = 0;
    layer_specs_seq[31].skip_connection_other_layer_scale = 1;
    layer_specs_seq[31].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[31].data_layout = CHW;


layer_specs_seq[32].layer_index = 32;
                pooling_layer_specs_seq[32].ifm_depth = 1024;
                pooling_layer_specs_seq[32].ifm_height = 7;
                pooling_layer_specs_seq[32].ifm_width = 7;
                pooling_layer_specs_seq[32].ofm_depth = 1024;
                pooling_layer_specs_seq[32].ofm_height = 1;
                pooling_layer_specs_seq[32].ofm_width = 1;
                pooling_layer_specs_seq[32].full_hw = true;
                pooling_layer_specs_seq[32].fused_scale = 1.0;
                pooling_layer_specs_seq[32].ifms_zero_point = -128;
                pooling_layer_specs_seq[32].ofms_zero_point = -128;


    layer_specs_seq[33].layer_index = 33;
    layer_specs_seq[33].conv_layer_type = PW_CONV; 
    layer_specs_seq[33].layer_num_fils = 1000;
    layer_specs_seq[33].strides = 1;
    layer_specs_seq[33].filter_size = 1;
    layer_specs_seq[33].padding_left = 0;
    layer_specs_seq[33].padding_right = 0;
    layer_specs_seq[33].padding_top = 0;
    layer_specs_seq[33].padding_bottom = 0;
    layer_specs_seq[33].layer_depth = 1024;
    layer_specs_seq[33].layer_ifm_height = 1;
    layer_specs_seq[33].layer_ifm_width = 1;
    layer_specs_seq[33].layer_ofm_height = 1;
    layer_specs_seq[33].layer_ofm_width = 1;
    layer_specs_seq[33].layer_activation = 0;
    layer_specs_seq[33].layer_num_of_ifm_tiles_h = (1 + TILE_H - 1) / TILE_H;
    layer_specs_seq[33].layer_num_of_ifm_tiles_w = (1 + TILE_W - 1) / TILE_W;
    layer_specs_seq[33].layer_num_of_ofm_tiles_h = (1 + TILE_H - 1) / TILE_H;
    layer_specs_seq[(1 + TILE_W - 1) / TILE_W].layer_num_of_ofm_tiles_w = 33;
    layer_specs_seq[33].layer_weights_offset = 3139584;
    layer_specs_seq[33].write_to_result_or_channels = 1;
    layer_specs_seq[33].write_to_tmp = 1;
    layer_specs_seq[33].followed_by = 0;
    layer_specs_seq[33].layer_ifms_zero_point = -128;
    layer_specs_seq[33].layer_ofms_scale = 0.18109971284866333;
    layer_specs_seq[33].relu_threshold = 0 ;
    layer_specs_seq[33].layer_ofms_zero_point = -64;
    layer_specs_seq[33].add_layer_scale_reciprocal = 1;
    layer_specs_seq[33].add_layer_zero_point = 0;
    layer_specs_seq[33].skip_connection_other_layer_scale = 1;
    layer_specs_seq[33].skip_connection_other_layer_zero_point = 0;

    layer_specs_seq[33].data_layout = CHW;








}
#endif
