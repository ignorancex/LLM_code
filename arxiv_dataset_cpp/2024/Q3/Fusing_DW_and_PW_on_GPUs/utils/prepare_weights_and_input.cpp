#include "prepare_weights_and_input.h"

void fill_layer_input_cpu(string file_name, fms_dt layer_input[MAX_FMS_SIZE],
                          const layer_specs layer_specs_struct)
{

    int a;
    int line = 0;
    std::ifstream infile(file_name);
    assert(!infile.fail());
    while (infile >> a)
    {
        layer_input[line] = a;
        line++;
        if (line > MAX_FMS_SIZE)
        {
            cout << "fill_layer_input_cpu: The file " << file_name << " contains more entries, double check!\n";
            break;
        }
    }
    infile.close();
}

void fill_layer_input(string file_name, fms_dt layer_input[MAX_FMS_SIZE_PACKED],
                      const layer_specs layer_specs_struct)
{

    const int ifms_h = layer_specs_struct.layer_ifm_height;
    const int ifms_w = layer_specs_struct.layer_ifm_width;
    const int ifms_hw = ifms_h * ifms_w;
    const int num_of_tiles_w = layer_specs_struct.layer_num_of_ifm_tiles_w;
    const int num_of_tiles_d = layer_specs_struct.layer_depth / PACKED_ITEMS;
    const int num_of_tiles_wd = num_of_tiles_w * num_of_tiles_d;

#if DATA_TYPE == INT8_DTYPE
    int a;
#elif DATA_TYPE == FLOAT_DTYPE
    float a;
#endif
    int line = 0;
    std::ifstream infile(file_name);
    assert(!infile.fail());
    while (infile >> a)
    {
        int z = (line / ifms_hw);
        int h = ((line % ifms_hw) / ifms_w);
        int w = (line % ifms_w);

        int tile_in_z = z / PACKED_ITEMS;
        int tile_in_h = h / TILE_H;
        int tile_in_w = w / TILE_W;
        int tile_index = tile_in_h * num_of_tiles_wd + tile_in_w * num_of_tiles_d + tile_in_z;

        int in_tile_z = z % PACKED_ITEMS;
        int in_tile_h = h % TILE_H;
        int in_tile_w = w % TILE_W;
        int in_tile_index = in_tile_h * TILE_W + in_tile_w;

        int absolute_index = tile_index * TILE_HW + in_tile_index;

#if DATA_TYPE == INT8_DTYPE
        PACK_32_8(layer_input[absolute_index], (uint8_t)a, in_tile_z);
#elif DATA_TYPE == FLOAT_DTYPE
        layer_input[absolute_index] = a;
#endif

        line++;

        if (line > MAX_FMS_SIZE)
        {
            cout << "fill_layer_input: The file " << file_name << " contains more entries, double check!\n";
            break;
        }
    }
}

void fill_layer_input_fw_hzw(string file_name, fms_dt layer_input[MAX_FMS_SIZE_PACKED],
                             const layer_specs layer_specs_struct)
{
    memset(layer_input, 0, MAX_FMS_SIZE_PACKED * sizeof(fms_dt));
    const int ifms_h = layer_specs_struct.layer_ifm_height;
    const int ifms_w = layer_specs_struct.layer_ifm_width;
    const int ifms_hw = ifms_h * ifms_w;
    const int num_of_tiles_d = layer_specs_struct.layer_depth / PACKED_ITEMS;

#if DATA_TYPE == INT8_DTYPE
    int a;
#elif DATA_TYPE == FLOAT_DTYPE
    float a;
#endif
    int line = 0;
    std::ifstream infile(file_name);
    assert(!infile.fail());
    while (infile >> a)
    {
        int z = (line / ifms_hw);
        int h = ((line % ifms_hw) / ifms_w);
        int w = (line % ifms_w);

        int tile_in_z = z / PACKED_ITEMS;
        int tile_in_h = h;
        int tile_index = tile_in_h * num_of_tiles_d + tile_in_z;

        int in_tile_z = z % PACKED_ITEMS;
        int in_tile_w = w;
        int in_tile_index = in_tile_w;

        int absolute_index = tile_index * ifms_w + in_tile_index;

#if DATA_TYPE == INT8_DTYPE
        PACK_32_8(layer_input[absolute_index], (uint8_t)a, in_tile_z);
#elif DATA_TYPE == FLOAT_DTYPE
        layer_input[absolute_index] = a;
#endif

        line++;

        if (line > MAX_FMS_SIZE)
        {
            cout << "fill_layer_input_fw: The file " << file_name << " contains more entries, double check!\n";
            break;
        }
    }
}

void fill_layer_input_fw_hwz(string file_name, fms_dt layer_input[MAX_FMS_SIZE_PACKED],
                             const layer_specs layer_specs_struct)
{
    memset(layer_input, 0, MAX_FMS_SIZE_PACKED * sizeof(fms_dt));
    const int ifms_h = layer_specs_struct.layer_ifm_height;
    const int ifms_w = layer_specs_struct.layer_ifm_width;
    const int ifms_hw = ifms_h * ifms_w;
    const int layer_depth_packed = layer_specs_struct.layer_depth / PACKED_ITEMS;

#if DATA_TYPE == INT8_DTYPE
    int a;
#elif DATA_TYPE == FLOAT_DTYPE
    float a;
#endif
    int line = 0;
    std::ifstream infile(file_name);
    assert(!infile.fail());
    while (infile >> a)
    {
        int z = (line / ifms_hw);
        int h = ((line % ifms_hw) / ifms_w);
        int w = (line % ifms_w);

        int z_packed = z / PACKED_ITEMS;
        int tile_in_h = h;

        int in_packed_z = z % PACKED_ITEMS;
        int in_tile_w = w;
        int in_tile_index = in_tile_w * layer_depth_packed + z_packed;

        int absolute_index = tile_in_h * (ifms_w * layer_depth_packed) + in_tile_index;
        // if(z == 0 && h == 27 && w == 89){
        //     printf("%d >> %f", absolute_index, a);
        // }

#if DATA_TYPE == INT8_DTYPE
        PACK_32_8(layer_input[absolute_index], (uint8_t)a, in_packed_z);
#elif DATA_TYPE == FLOAT_DTYPE
        layer_input[absolute_index] = a;
#endif

        line++;

        if (line > MAX_FMS_SIZE)
        {
            cout << "fill_layer_input_fw: The file " << file_name << " contains more entries, double check!\n";
            break;
        }
    }
}

void fill_layer_input_fwh_zhw(string file_name, fms_dt layer_input[MAX_FMS_SIZE_PACKED],
                              const layer_specs layer_specs_struct)
{
    memset(layer_input, 0, MAX_FMS_SIZE_PACKED * sizeof(fms_dt));
    const int ifms_h = layer_specs_struct.layer_ifm_height;
    const int ifms_w = layer_specs_struct.layer_ifm_width;
    const int ifms_hw = ifms_h * ifms_w;
    const int layer_depth_packed = layer_specs_struct.layer_depth / PACKED_ITEMS;

#if DATA_TYPE == INT8_DTYPE
    int a;
#elif DATA_TYPE == FLOAT_DTYPE
    float a;
#endif
    int line = 0;

    std::ifstream infile(file_name);
    assert(!infile.fail());
    while (infile >> a)
    {
        int z = (line / ifms_hw);
        int h = ((line % ifms_hw) / ifms_w);
        int w = (line % ifms_w);

        int z_packed = z / PACKED_ITEMS;
        int in_packed_z = z % PACKED_ITEMS;

        int in_tile_index = h * ifms_w + w;

        int absolute_index = z_packed * ifms_hw + in_tile_index;

#if DATA_TYPE == INT8_DTYPE
        PACK_32_8(layer_input[absolute_index], (uint8_t)a, in_packed_z);
#elif DATA_TYPE == FLOAT_DTYPE
        layer_input[absolute_index] = a;
#endif

        line++;

        if (line > MAX_FMS_SIZE)
        {
            cout << "fill_layer_input_fw: The file " << file_name << " contains more entries, double check!\n";
            break;
        }
    }
}

void verify_fill_layer_input(string file_name, fms_dt ifms[MAX_FMS_SIZE_PACKED],
                             const layer_specs layer_specs_struct)
{

    ofstream myfile;
    int8_t to_print_ofms[MAX_FMS_SIZE];

    const int ifms_h = layer_specs_struct.layer_ifm_height;
    const int ifms_w = layer_specs_struct.layer_ifm_width;
    const int ifms_hw = ifms_h * ifms_w;
    const int ifms_size = ifms_hw * layer_specs_struct.layer_depth;
    const int packed_ifms_size = ifms_size / PACKED_ITEMS;
    const int num_of_tiles_w = layer_specs_struct.layer_num_of_ifm_tiles_w;
    const int num_of_tiles_d = layer_specs_struct.layer_depth / PACKED_ITEMS;
    const int num_of_tiles_wd = num_of_tiles_w * num_of_tiles_d;

    for (int i = 0; i < packed_ifms_size; i++)
    {
        int tile_indx = i / TILE_HW;
        int in_tile_index = i % TILE_HW;

        int tile_in_d = tile_indx % num_of_tiles_d;
        int tile_in_w = (tile_indx / num_of_tiles_d) % num_of_tiles_w;
        int tile_in_h = tile_indx / num_of_tiles_wd;

        int in_tile_h = in_tile_index / TILE_W;
        int in_tile_w = in_tile_index % TILE_W;

        for (int bi = 0; bi < PACKED_ITEMS; bi++)
        {
            int8_t extracted_val = EXTRACT_8_32(ifms[i], bi);
            int absolute_index = (tile_in_d * PACKED_ITEMS + bi) * ifms_hw + (tile_in_h * TILE_H + in_tile_h) * ifms_w +
                                 tile_in_w * TILE_W + in_tile_w;
            to_print_ofms[absolute_index] = extracted_val;
        }
    }
    myfile.open(file_name);
    for (int i = 0; i < ifms_size; i++)
    {
        myfile << (int)to_print_ofms[i] << "\n";
    }
    myfile.close();
}

void verify_fill_layer_input_fw(string file_name, fms_dt ifms[MAX_FMS_SIZE_PACKED],
                                const layer_specs layer_specs_struct)
{

    ofstream myfile;
    int8_t to_print_ofms[MAX_FMS_SIZE];

    const int ifms_h = layer_specs_struct.layer_ifm_height;
    const int ifms_w = layer_specs_struct.layer_ifm_width;
    const int ifms_hw = ifms_h * ifms_w;
    const int ifms_size = ifms_hw * layer_specs_struct.layer_depth;
    const int packed_ifms_size = ifms_size / PACKED_ITEMS;
    const int num_of_tiles_d = layer_specs_struct.layer_depth / PACKED_ITEMS;

    for (int i = 0; i < packed_ifms_size; i++)
    {
        int tile_indx = i / ifms_w;
        int in_tile_index = i % ifms_w;

        int tile_in_d = tile_indx % num_of_tiles_d;
        int tile_in_h = tile_indx / num_of_tiles_d;

        int in_tile_w = in_tile_index % ifms_w;

        for (int bi = 0; bi < PACKED_ITEMS; bi++)
        {
            int8_t extracted_val = EXTRACT_8_32(ifms[i], bi);
            int absolute_index = (tile_in_d * PACKED_ITEMS + bi) * ifms_hw + tile_in_h * ifms_w +
                                 in_tile_w;
            to_print_ofms[absolute_index] = extracted_val;
        }
    }
    myfile.open(file_name);
    for (int i = 0; i < ifms_size; i++)
    {
        myfile << (int)to_print_ofms[i] << "\n";
    }
    myfile.close();
}

int read_count_from_a_file(string file_name)
{
    int a;
    std::ifstream infile(file_name);
    assert(!infile.fail());
    int line_num = 0;
    infile >> a;

    return a;
}

int count_weights(string file_name)
{
    int a;
    std::ifstream infile(file_name);
    assert(!infile.fail());
    int line_num = 0;
    int packed_weight = 0;
    while (infile >> a)
    {
        line_num++;
    }
    return line_num;
}

void load_weights_cpu(string file_name,
                      weights_dt weights[])
{
    int a;
    std::ifstream infile(file_name);
    assert(!infile.fail());
    int line_num = 0;
    while (infile >> a)
    {
        weights[line_num] = (weights_dt)a;
        line_num += 1;
    }
    infile.close();
}

void load_dw_weights(string file_name,
                     weights_dt weights[],
                     layer_specs layer_specs_seq[])
{
#if DATA_TYPE == INT8_DTYPE
    int a;
#elif DATA_TYPE == FLOAT_DTYPE
    float a;
#endif
    weights_dt packed_weight;
    std::ifstream infile(file_name);
    assert(!infile.fail());
    int line_num = 0;
    while (infile >> a)
    {
        int index_in_pack = line_num % PACKED_ITEMS;
        if (index_in_pack == 0)
        {
            packed_weight = 0;
        }
#if DATA_TYPE == INT8_DTYPE
        PACK_32_8(packed_weight, (uint8_t)a, index_in_pack);
#elif DATA_TYPE == FLOAT_DTYPE
        packed_weight = a;
#endif
        if (index_in_pack == 3)
        {
            weights[line_num / PACKED_ITEMS] = packed_weight;
        }
        line_num += 1;
    }
#if DATA_TYPE == FLOAT_DTYPE && MIXED_LAYOUT
    transform_dw_layers_weights(weights, layer_specs_seq);
#endif
}

void load_dw_weights_padded_filters(string file_name,
                                    weights_dt weights[],
                                    layer_specs layer_specs_seq[])
{
#if DATA_TYPE == INT8_DTYPE
    int a;
#elif DATA_TYPE == FLOAT_DTYPE
    float a;
#endif
    std::ifstream infile(file_name);
    assert(!infile.fail());
    int line_num = 0, current_layer_index = 0;
    while (current_layer_index < MODEL_NUM_LAYERS && layer_specs_seq[current_layer_index].conv_layer_type != DW_CONV)
    {
        current_layer_index++;
    }
    layer_specs l_specs = layer_specs_seq[current_layer_index];
    int layer_filter_area_padded = least_pow_of_2_geq(l_specs.filter_size * l_specs.filter_size);
    int layer_filter_area = l_specs.filter_size * l_specs.filter_size;
    int layer_num_weights = layer_filter_area * l_specs.layer_num_fils;
    int line_num_in_layer = 0;

    while (infile >> a)
    {
        if (line_num_in_layer >= layer_num_weights)
        {
            current_layer_index++;
            while (current_layer_index < MODEL_NUM_LAYERS && layer_specs_seq[current_layer_index].conv_layer_type != DW_CONV)
            {
                current_layer_index++;
            }
            l_specs = layer_specs_seq[current_layer_index];
            layer_filter_area_padded = least_pow_of_2_geq(l_specs.filter_size * l_specs.filter_size);
            layer_filter_area = l_specs.filter_size * l_specs.filter_size;
            layer_num_weights = layer_filter_area * l_specs.layer_num_fils;
            line_num_in_layer = 0;
        }

        int in_filter_index = line_num_in_layer % layer_filter_area;
        int filter_index = (line_num_in_layer / layer_filter_area);
        int filter_index_packed = filter_index / PACKED_ITEMS;
        int index_in_pack = filter_index % PACKED_ITEMS;

        // printf("%d,\n", l_specs.layer_weights_offset / PACKED_ITEMS +
        //                                filter_index_packed * layer_filter_area_padded +
        //                                in_filter_index);
#if DATA_TYPE == INT8_DTYPE
        PACK_32_8(weights[l_specs.layer_weights_offset / PACKED_ITEMS +
                          filter_index_packed * layer_filter_area_padded + in_filter_index],
                  (uint8_t)a, index_in_pack);
#elif DATA_TYPE == FLOAT_DTYPE
        weights[l_specs.layer_weights_offset +
                filter_index * layer_filter_area_padded + in_filter_index] = a;
        // if (filter_index < 5)
        // {
        //     printf("%d, %d, %d, %d > %f\n", filter_index, layer_filter_area_padded, in_filter_index,
        //            filter_index * layer_filter_area_padded + in_filter_index, a);
        // }
#endif
        line_num += 1;
        line_num_in_layer += 1;
    }
    infile.close();
}

void load_dw_weights_cpu(string file_name,
                         weights_dt weights[])
{
#if DATA_TYPE == INT8_DTYPE
    int a;
#elif DATA_TYPE == FLOAT_DTYPE
    float a;
#endif
    std::ifstream infile(file_name);
    assert(!infile.fail());
    int line_num = 0;
    while (infile >> a)
    {
        weights[line_num] = a;
        line_num += 1;
        if (line_num >= MAX_FMS_SIZE)
        {
            cout << "load_weights_cpu: The file " << file_name << " contains more entries, double check!\n";
            break;
        }
    }
}

void load_dw_weights_cpu_padded(string file_name,
                                weights_dt weights[],
                                layer_specs layer_specs_seq[])
{
#if DATA_TYPE == INT8_DTYPE
    int a;
#elif DATA_TYPE == FLOAT_DTYPE
    float a;
#endif
    std::ifstream infile(file_name);
    assert(!infile.fail());
    int line_num = 0, current_layer_index = 0;
    while (current_layer_index < MODEL_NUM_LAYERS && layer_specs_seq[current_layer_index].conv_layer_type != DW_CONV)
    {
        current_layer_index++;
    }
    layer_specs l_specs = layer_specs_seq[current_layer_index];
    int layer_filter_area_padded = least_pow_of_2_geq(l_specs.filter_size * l_specs.filter_size);
    int layer_filter_area = l_specs.filter_size * l_specs.filter_size;
    int layer_num_weights = layer_filter_area * l_specs.layer_num_fils;
    int line_num_in_layer = 0;

    while (infile >> a)
    {
        if (line_num_in_layer >= layer_num_weights)
        {
            current_layer_index++;
            while (current_layer_index < MODEL_NUM_LAYERS && layer_specs_seq[current_layer_index].conv_layer_type != DW_CONV)
            {
                current_layer_index++;
            }
            l_specs = layer_specs_seq[current_layer_index];
            layer_filter_area_padded = least_pow_of_2_geq(l_specs.filter_size * l_specs.filter_size);
            layer_filter_area = l_specs.filter_size * l_specs.filter_size;
            layer_num_weights = layer_filter_area * l_specs.layer_num_fils;
            line_num_in_layer = 0;
        }

        int in_filter_index = line_num_in_layer % layer_filter_area;
        int filter_index = (line_num_in_layer / layer_filter_area);

        weights[l_specs.layer_weights_offset +
                filter_index * layer_filter_area_padded + in_filter_index] = a;

        // printf("%d, %d, %d,\n", line_num, l_specs.layer_weights_offset + filter_index * layer_filter_area_padded + in_filter_index, weights[l_specs.layer_weights_offset + filter_index * layer_filter_area_padded + in_filter_index]);

        line_num += 1;
        line_num_in_layer += 1;

        if (line_num >= MAX_FMS_SIZE)
        {
            cout << "load_weights_cpu: The file " << file_name << " contains more entries, double check!\n";
            break;
        }
    }
    infile.close();
}

void load_weights(string file_name,
                  weights_dt weights[],
                  layer_specs layer_specs_seq[])
{
#if DATA_TYPE == INT8_DTYPE
    int a;
#elif DATA_TYPE == FLOAT_DTYPE
    float a;
#endif
    std::ifstream infile(file_name);
    assert(!infile.fail());
    int line_num = 0;
    int packed_weight = 0;
    while (infile >> a)
    {
        int index_in_pack = line_num % PACKED_ITEMS;
        if (index_in_pack == 0)
        {
            packed_weight = 0;
        }
#if DATA_TYPE == INT8_DTYPE
        PACK_32_8(packed_weight, (uint8_t)a, index_in_pack);
        if (index_in_pack == 3)
        {
            weights[line_num / PACKED_ITEMS] = packed_weight;
        }
#elif DATA_TYPE == FLOAT_DTYPE
        weights[line_num / PACKED_ITEMS] = a;
#endif

        line_num++;
    }
#if DATA_TYPE == FLOAT_DTYPE && MIXED_LAYOUT
    transform_pw_layers_weights(weights, layer_specs_seq);
#endif
    infile.close();
}

void transform_pw_weights_fc_to_cf(weights_dt *weights, layer_specs l_specs)
{

    const int layer_weights_offset = l_specs.layer_weights_offset;
    const int filter_size = l_specs.layer_depth;
    const int layer_num_fils = l_specs.layer_num_fils;
    weights_dt tmp_buffer[filter_size * layer_num_fils];

    for (int f = 0; f < layer_num_fils; f++)
    {
        for (int d = 0; d < filter_size; d++)
        {
            tmp_buffer[d * layer_num_fils + f] =
                weights[layer_weights_offset + f * filter_size + d];
        }
    }

    for (int i = 0; i < layer_num_fils * filter_size; i++)
    {
        // printf("%f\n", tmp_buffer[i]);
        weights[layer_weights_offset + i] = tmp_buffer[i];
    }
}

void transform_pw_layers_weights(weights_dt *weights, layer_specs layer_specs_seq[])
{

    for (int i = 0; i < MODEL_NUM_LAYERS; i++)
    {
        layer_specs l_specs = layer_specs_seq[i];
        if (l_specs.conv_layer_type == PW_CONV && l_specs.data_layout == HWC)
        {
            transform_pw_weights_fc_to_cf(weights, l_specs);
        }
    }
}

void transform_dw_layers_weights(weights_dt *weights, layer_specs layer_specs_seq[])
{

    for (int i = 0; i < MODEL_NUM_LAYERS; i++)
    {
        layer_specs l_specs = layer_specs_seq[i];
        if (l_specs.conv_layer_type == DW_CONV && l_specs.data_layout == HWC)
        {
            transform_pw_weights_fc_to_cf(weights, l_specs);
        }
    }
}

void transform_dw_weights_fc_to_cf(weights_dt *weights, layer_specs l_specs)
{

    const int layer_weights_offset = l_specs.layer_weights_offset;
#if PADDED_DW_WEIGHTS
    const int filter_size = least_pow_of_2_geq(l_specs.filter_size * l_specs.filter_size);
#else
    const int filter_size = l_specs.filter_size * l_specs.filter_size;
#endif
    const int layer_num_fils = l_specs.layer_num_fils;

    for (int f = 0; f < layer_num_fils; f++)
    {
        for (int d = 0; d < filter_size; d++)
        {
            weights[layer_weights_offset + d * layer_num_fils + f] =
                weights[layer_weights_offset + f * filter_size + d];
        }
    }
}

void load_zps(string file_name,
              biases_dt fused_zps[])
{
    int a;
    std::ifstream infile(file_name);
    assert(!infile.fail());
    int line_num = 0;
    int packed_weight = 0;
    while (infile >> a)
    {
        fused_zps[line_num] = (biases_dt)a;
        line_num++;
    }
    infile.close();
}

void load_scales(string file_name, fused_scales_dt fused_scales[])
{
    float a;
    std::ifstream infile(file_name);
    assert(!infile.fail());
    int line_num = 0;
    int packed_weight = 0;
    while (infile >> a)
    {
        fused_scales[line_num] = (fused_scales_dt)a;
        line_num++;
    }
    infile.close();
}

void verify_load_weights(string file_name, weights_dt weights[], const int num_of_weights)
{

    const int packed_num_weights = num_of_weights / PACKED_ITEMS;
    std::ifstream infile(file_name);
    assert(!infile.fail());
    int line_num = 0;
    int packed_weight = 0;
    int a;
    for (int i = 0; i < packed_num_weights; i++)
    {
        for (int bi = 0; bi < PACKED_ITEMS; bi++)
        {
            infile >> a;
            int8_t extracted_val = EXTRACT_8_32(weights[i], bi);
            if (a != extracted_val)
            {
                cout << "weight mismatch at: " << i << " : should be " << a << " but it is " << (int)extracted_val << "\n";
                return;
            }
        }
    }
    infile.close();
}