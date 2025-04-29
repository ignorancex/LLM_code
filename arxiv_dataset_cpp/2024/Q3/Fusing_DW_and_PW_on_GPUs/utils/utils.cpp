#include "utils.h"

void read_settings(string file_name, Settings_struct &settings_struct)
{

    std::string delimiter = "::";
    std::ifstream infile(file_name);
    assert(!infile.fail());
    string s;
    settings_struct.num_sms = 0;
    settings_struct.bench = 0;
    while (infile >> s)
    {
        if (s.find(delimiter) != std::string::npos)
        {
            std::string token = s.substr(s.find(delimiter) + delimiter.length());
            if (s.find("IFMS_FILE") != std::string::npos || s.find("ifms_file") != std::string::npos)
            {
                token = token.replace(token.find("*M*"), 3, get_model_prefix());
                settings_struct.ifms_file_name = token;
            }
            else if (s.find("FUSION_FILE") != std::string::npos || s.find("fusion_file") != std::string::npos)
            {
                token = token.replace(token.find("*M*"), 3, get_model_prefix());
                if (DATA_TYPE == INT8_DTYPE)
                {
                    token = token.replace(token.find("*DT*"), 4, "int8");
                }
                else if (DATA_TYPE == FLOAT_DTYPE)
                {
                    token = token.replace(token.find("*DT*"), 4, "fp32");
                }
                settings_struct.fusion_file = token;
            }
            else if (s.find("DUMP_FILE") != std::string::npos || s.find("dump_file") != std::string::npos)
            {
                settings_struct.dump_file_name = token;
            }
            else if (s.find("TEST_ITERATIONS") != std::string::npos || s.find("test_iterations") != std::string::npos)
            {
                settings_struct.test_iterations = stoi(token);
            }
            else if (s.find("FIRST_LAYER") != std::string::npos || s.find("first_layer") != std::string::npos)
            {
                settings_struct.first_layer = stoi(token);
            }
            else if (s.find("NUM_LAYERS") != std::string::npos || s.find("num_layers") != std::string::npos)
            {
                settings_struct.num_layers = stoi(token);
            }
            else if (s.find("NUM_SMS") != std::string::npos || s.find("num_sms") != std::string::npos)
            {
                settings_struct.num_sms = stoi(token);
            }
            else if (s.find("RUN_FUSED") != std::string::npos || s.find("run_fused") != std::string::npos)
            {
                settings_struct.run_fused = stoi(token);
            }
            else if (s.find("RUN_UNFUSED") != std::string::npos || s.find("run_unfused") != std::string::npos)
            {
                settings_struct.run_unfused = stoi(token);
            }
            else if (s.find("BENCH") != std::string::npos || s.find("bench") != std::string::npos)
            {
                settings_struct.bench = stoi(token);
            }
        }
    }
    infile.close();
}

fusion_types get_fusion_type(string s)
{
    if (s.find("PWDW_WIDE") != std::string::npos || s.find("pwdw_wide") != std::string::npos)
    {
        return pwdw_wide;
    }
    else if (s.find("DWPW") != std::string::npos || s.find("dwpw") != std::string::npos)
    {
        return dwpw;
    }
    else if (s.find("PWDW") != std::string::npos || s.find("pwdw") != std::string::npos)
    {
        return pwdw;
    }
    else if (s.find("PWPW") != std::string::npos || s.find("pwpw") != std::string::npos)
    {
        return pwpw;
    }
}

void read_fusions_list(string file_name, Fusion_struct *layers_fusions)
{

    std::string delimiter = ",";
    std::ifstream infile(file_name);
    assert(!infile.fail());
    string s;

    for (int i = 0; i < MODEL_NUM_LAYERS; i++)
    {
        Fusion_struct fusion_struct;
        fusion_struct.first_layer_index = -1;
        layers_fusions[i] = fusion_struct;
    }

    while (infile >> s)
    {
        if (s.find(delimiter) != std::string::npos)
        {
            Fusion_struct fusion_struct;
            fusion_struct.first_layer_index = stoi(s.substr(0, s.find(delimiter)));
            std::string tmp = s.substr(s.find(delimiter) + delimiter.length());
            fusion_struct.second_layer_index = stoi(tmp.substr(0, tmp.find(delimiter)));
            fusion_struct.fusion_type = get_fusion_type(s);
            layers_fusions[fusion_struct.first_layer_index] = fusion_struct;
        }
    }

    infile.close();
}

void read_ints_from_file(string file_name,
                         int ints[])
{
    int a;
    std::ifstream infile(file_name);
    assert(!infile.fail());
    int line_num = 0;
    while (infile >> a)
    {
        ints[line_num] = a;
        line_num += 1;
    }
    infile.close();
}

bool compare_cpu_and_gpu_outputs(fms_dt *ofms_cpu, fms_dt *ofms_gpu, layer_specs layer_specs_struct)
{

    const int ofms_h = layer_specs_struct.layer_ofm_height;
    const int ofms_w = layer_specs_struct.layer_ofm_width;
    const int ifms_hw = ofms_h * ofms_w;
    const int num_of_tiles_w = layer_specs_struct.layer_num_of_ofm_tiles_w;
    const int num_of_tiles_d = layer_specs_struct.layer_num_fils / PACKED_ITEMS;
    const int num_of_tiles_wd = num_of_tiles_w * num_of_tiles_d;

    int diff_count = 0;

    for (int i = 0; i < ifms_hw * layer_specs_struct.layer_depth; i++)
    {
        int z = (i / ifms_hw);
        int h = ((i % ifms_hw) / ofms_w);
        int w = (i % ofms_w);

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
        int ofms_gpu_val = EXTRACT_8_32(ofms_gpu[absolute_index], in_tile_z);
#else
        float ofms_gpu_val = ofms_gpu[absolute_index];
#endif
        if (ofms_cpu[i] != ofms_gpu_val)
        {
            cout << "error at: " << i << ", " << absolute_index << " CPU: " << ofms_cpu[i] << " GPU: " << ofms_gpu_val << "\n";
            cout << z << " " << h << " " << w << "\n";
            diff_count++;
            if (diff_count > 100)
            {
                break;
            }
        }
    }
    return diff_count == 0;
}

bool compare_cpu_and_gpu_outputs_fw_hzw(fms_dt *ofms_cpu, fms_dt *ofms_gpu, layer_specs layer_specs_struct)
{

    const int ofms_h = layer_specs_struct.layer_ofm_height;
    const int ofms_w = layer_specs_struct.layer_ofm_width;
    const int ifms_hw = ofms_h * ofms_w;
    const int num_fils = layer_specs_struct.layer_num_fils;
    const int num_of_tiles_d = num_fils / PACKED_ITEMS;

    int diff_count = 0;

    for (int i = 0; i < ifms_hw * num_fils; i++)
    {
        int z = (i / ifms_hw);
        int h = ((i % ifms_hw) / ofms_w);
        int w = (i % ofms_w);

        int tile_in_z = z / PACKED_ITEMS;
        int tile_in_h = h;
        int tile_index = tile_in_h * num_of_tiles_d + tile_in_z;

        int in_tile_z = z % PACKED_ITEMS;
        int in_tile_w = w;
        int in_tile_index = in_tile_w;

        int absolute_index = tile_index * ofms_w + in_tile_index;
        int ofms_gpu_val = EXTRACT_8_32(ofms_gpu[absolute_index], in_tile_z);
        if (ofms_cpu[i] != ofms_gpu_val)
        {
            cout << "error at: " << i << ", " << absolute_index << " CPU: " << ofms_cpu[i] << " GPU: " << ofms_gpu_val << "\n";
            cout << z << " " << h << " " << w << "\n";
            diff_count++;
            if (diff_count > 100)
            {
                break;
            }
        }
    }
    return diff_count == 0;
}

bool compare_cpu_and_gpu_outputs_fhw_zhw(fms_dt *ofms_cpu, fms_dt *ofms_gpu, layer_specs layer_specs_struct)
{
    const int ofms_h = layer_specs_struct.layer_ofm_height;
    const int ofms_w = layer_specs_struct.layer_ofm_width;
    const int ifms_hw = ofms_h * ofms_w;
    const int num_fils = layer_specs_struct.layer_num_fils;
    const int num_of_tiles_d = num_fils / PACKED_ITEMS;

    int diff_count = 0;

    for (int i = 0; i < ifms_hw * num_fils; i++)
    {
        int z = (i / ifms_hw);
        int h = ((i % ifms_hw) / ofms_w);
        int w = (i % ofms_w);

        int tile_in_z = z / PACKED_ITEMS;
        int in_tile_z = z % PACKED_ITEMS;

        int absolute_index = tile_in_z * ifms_hw + h * ofms_w + w;
#if DATA_TYPE == INT8_DTYPE
        int ofms_gpu_val = EXTRACT_8_32(ofms_gpu[absolute_index], in_tile_z);
#else
        float ofms_gpu_val = ofms_gpu[absolute_index];
#endif
        if ((DATA_TYPE == INT8_DTYPE && ofms_cpu[i] != ofms_gpu_val) || (DATA_TYPE == FLOAT_DTYPE &&
                                                                         abs((ofms_cpu[i] - ofms_gpu_val) / ofms_gpu_val) > 0.01))
        {
            cout << "error at: " << i << ", " << absolute_index << " CPU: " << ofms_cpu[i] << " GPU: " << ofms_gpu_val << "\n";
            cout << z << " " << h << " " << w << "\n";
            diff_count++;
            if (diff_count > 100)
            {
                break;
            }
        }
    }
    return diff_count == 0;
}

void dump_outputs_hwz(string file_name, fms_dt *ofms, layer_specs layer_specs_struct)
{

    const int ofms_h = layer_specs_struct.layer_ofm_height;
    const int ofms_w = layer_specs_struct.layer_ofm_width;
    const int ifms_hw = ofms_h * ofms_w;
    const int layer_depth_packed = layer_specs_struct.layer_num_fils / PACKED_ITEMS;

    int diff_count = 0;

    ofstream myfile;
    myfile.open(file_name);

    for (int i = 0; i < ifms_hw * layer_depth_packed; i++)
    {
        int z = (i / ifms_hw);
        int h = ((i % ifms_hw) / ofms_w);
        int w = (i % ofms_w);

        int z_packed = z / PACKED_ITEMS;
        int tile_in_h = h;

        int in_packed_z = z % PACKED_ITEMS;
        int in_tile_w = w;
        int in_tile_index = in_tile_w * layer_depth_packed + z_packed;

        int absolute_index = tile_in_h * (ofms_w * layer_depth_packed) + in_tile_index;
        myfile << ofms[absolute_index];
    }

    myfile.close();
}

bool compare_cpu_and_gpu_outputs_fw_hwz(fms_dt *ofms_cpu, fms_dt *ofms_gpu, layer_specs layer_specs_struct)
{

    const int ofms_h = layer_specs_struct.layer_ofm_height;
    const int ofms_w = layer_specs_struct.layer_ofm_width;
    const int ofms_hw = ofms_h * ofms_w;
    const int layer_depth_packed = layer_specs_struct.layer_num_fils / PACKED_ITEMS;

    int diff_count = 0;

    for (int i = 0; i < ofms_hw * layer_depth_packed; i++)
    {
        int z = (i / ofms_hw);
        int h = ((i % ofms_hw) / ofms_w);
        int w = (i % ofms_w);

        int z_packed = z / PACKED_ITEMS;
        int tile_in_h = h;

        int in_packed_z = z % PACKED_ITEMS;
        int in_tile_w = w;
        int in_tile_index = in_tile_w * layer_depth_packed + z_packed;

        int absolute_index = tile_in_h * (ofms_w * layer_depth_packed) + in_tile_index;
#if DATA_TYPE == INT8_DTYPE
        int ofms_gpu_val = EXTRACT_8_32(ofms_gpu[absolute_index], in_packed_z);
#else
        float ofms_gpu_val = ofms_gpu[absolute_index];
#endif
        if (ofms_cpu[i] != ofms_gpu_val)
        {
            cout << "error at: " << i << ", " << absolute_index << " CPU: " << ofms_cpu[i] << " GPU: " << ofms_gpu_val << "\n";
            cout << z << " " << h << " " << w << "\n";
            diff_count++;
            if (diff_count > 100)
            {
                break;
            }
        }
    }
    return diff_count == 0;
}

string get_model_prefix()
{
    string model_name;
    switch (MODEL_ID)
    {
    case MOB_V1:
        model_name = "mob_v1";
        break;
    case MOB_V2:
        model_name = "mob_v2";
        break;
    case MOB_V2_0_25:
        model_name = "mob_v2_0_25";
        break;
    case MOB_V2_0_5:
        model_name = "mob_v2_0_5";
        break;
    case MOB_V2_0_75:
        model_name = "mob_v2_0_75";
        break;
    case RESNET50:
        model_name = "resnet50";
    case XCE_R:
        model_name = "xce_r";
        break;
    case GPROX_3:
        model_name = "gprox_3";
        break;
    default:
        break;
    }
    return model_name;
}

void dump_cpu_output(string file_name, fms_dt ofms[MAX_FMS_SIZE],
                     const layer_specs layer_specs_struct)
{

    int layer_size = layer_specs_struct.layer_num_fils * layer_specs_struct.layer_ofm_height *
                     layer_specs_struct.layer_ofm_width;

    ofstream myfile;
    myfile.open(file_name);
    for (int i = 0; i < layer_size; i++)
    {
        myfile << (int)ofms[i] << "\n";
    }
    myfile.close();
}

void dump_gpu_output_chw(string file_name, fms_dt *ofms_gpu, layer_specs layer_specs_struct)
{
    const int ofms_h = layer_specs_struct.layer_ofm_height;
    const int ofms_w = layer_specs_struct.layer_ofm_width;
    const int ofms_depth = layer_specs_struct.layer_depth;
    int ofms_size = ofms_depth * ofms_h * ofms_w;

    ofstream myfile;
    myfile.open(file_name);
    for (int i = 0; i < ofms_size; i++)
    {
        myfile << ofms_gpu[i] << "\n";
    }
    myfile.close();
}

void dump_gpu_output(string file_name, fms_dt *ofms_gpu, layer_specs layer_specs_struct)
{

    const int ofms_h = layer_specs_struct.layer_ofm_height;
    const int ofms_w = layer_specs_struct.layer_ofm_width;
    const int ofms_hw = ofms_h * ofms_w;
    const int num_of_tiles_w = layer_specs_struct.layer_num_of_ofm_tiles_w;
    const int num_of_tiles_d = layer_specs_struct.layer_num_fils / PACKED_ITEMS;
    const int num_of_tiles_wd = num_of_tiles_w * num_of_tiles_d;

    int diff_count = 0;

    ofstream myfile;
    myfile.open(file_name);

    for (int i = 0; i < layer_specs_struct.layer_num_fils * ofms_hw; i++)
    {
        int z = (i / ofms_hw);
        int h = ((i % ofms_hw) / ofms_w);
        int w = (i % ofms_w);

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
        int ofms_gpu_val = (int)EXTRACT_8_32(ofms_gpu[absolute_index], in_tile_z);
#else
        float ofms_gpu_val = ofms_gpu[absolute_index];
#endif
        myfile << ofms_gpu_val << "\n";
    }

    myfile.close();
}

void dump_gpu_output_fw(string file_name, fms_dt *ofms_gpu, layer_specs layer_specs_struct)
{

    const int ofms_h = layer_specs_struct.layer_ofm_height;
    const int ofms_w = layer_specs_struct.layer_ofm_width;
    const int ofms_hw = ofms_h * ofms_w;
    const int num_of_tiles_d = layer_specs_struct.layer_num_fils / PACKED_ITEMS;

    int diff_count = 0;

    ofstream myfile;
    myfile.open(file_name);

    for (int i = 0; i < layer_specs_struct.layer_num_fils * ofms_hw; i++)
    {
        int z = (i / ofms_hw);
        int h = ((i % ofms_hw) / ofms_w);
        int w = (i % ofms_w);

        int tile_in_z = z / PACKED_ITEMS;
        int tile_in_h = h;
        int tile_in_w = w / ofms_w;

        int tile_index = tile_in_h * num_of_tiles_d + tile_in_z;

        int in_tile_z = z % PACKED_ITEMS;
        int in_tile_w = w;
        int in_tile_index = in_tile_w;

        int absolute_index = tile_index * ofms_w + in_tile_index;
        int ofms_gpu_val = (int)EXTRACT_8_32(ofms_gpu[absolute_index], in_tile_z);

        myfile << ofms_gpu_val << "\n";
    }

    myfile.close();
}

int get_conv_layer_index(const int starting_layer_index, const int offset)
{
    int layer_index = -1;
    int i = 1;
    int conv_layers_so_far = 0;
    if (offset == 0)
    {
        return starting_layer_index;
    }
    for (i; conv_layers_so_far < offset && starting_layer_index + i < MODEL_NUM_LAYERS; i++)
    {
        if (conv_layers_indices[starting_layer_index + i])
        {
            conv_layers_so_far++;
            layer_index = starting_layer_index + i;
        }
    }
    return layer_index;
}

int least_pow_of_2_geq(int inp)
{
    return (int)pow(2, (int)(log2(inp) + 1));
}

int largest_pow_of_2_leq(int inp)
{
    return (int)pow(2, (int)(log2(inp)));
}

void get_pw_f_w_v2_parallelism_w(layer_specs *layer_specs_seq, int *layers_parallelism_w, const int num_layers, bool fused)
{

    for (int i = 0; i < num_layers; i++)
    {
        layer_specs l_specs = layer_specs_seq[i];
        if (l_specs.layer_index >= 0 && l_specs.layer_index < num_layers && l_specs.conv_layer_type == PW_CONV)
        {
            int compact_layer_num_filters = l_specs.layer_num_fils / PACKED_ITEMS;
            int layer_ofms_width = l_specs.layer_ofm_width;
            int parallel_w = MAX_THREADS_PER_BLOCK / (compact_layer_num_filters);
            if (fused || DATA_TYPE == FLOAT_DTYPE)
            {
                parallel_w /= 2;
            }
            if (layer_ofms_width < parallel_w)
            {
                parallel_w = layer_ofms_width;
            }
            layers_parallelism_w[i] = largest_pow_of_2_leq(parallel_w);
            if (layers_parallelism_w[i] == 0)
            {
                layers_parallelism_w[i] = 1;
            }
        }
    }
}

void get_pw_f_w_v2_parallelism_w_v2(layer_specs *layer_specs_seq, int *layers_parallelism_w, const int num_layers, bool fused)
{
    for (int i = 0; i < num_layers; i++)
    {
        layer_specs l_specs = layer_specs_seq[i];
        if (l_specs.layer_index >= 0 && l_specs.layer_index < num_layers && l_specs.conv_layer_type == PW_CONV)
        {
            int compact_layer_num_filters = l_specs.layer_num_fils / PACKED_ITEMS;
            int layer_ofms_width = l_specs.layer_ofm_width;
            int parallel_w = WARP_SIZE;
            if (layer_ofms_width < parallel_w)
            {
                parallel_w = layer_ofms_width;
            }
            layers_parallelism_w[i] = largest_pow_of_2_leq(parallel_w);
        }
    }
}