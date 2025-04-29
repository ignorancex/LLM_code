
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdlib>
#include "./headers/general_specs.h"
#include "./headers/parallalism_and_tiling.h"
#include <stdio.h>
#include <iostream>
#include "./headers/conv_kernels.h"
#include "./model_specs/layers_specs.h"
#include "./utils/prepare_weights_and_input.h"
#include <chrono>
#include "./headers/other_layers.h"

using namespace chrono;

using namespace std;

#if DSC_MODEL

int main(int argc, char **argv)
{
    layer_specs layer_specs_seq[MODEL_NUM_LAYERS];
    pooling_layer_specs pooling_layer_specs_seq[MODEL_NUM_LAYERS];
    Fusion_struct layers_fusions[MODEL_NUM_LAYERS];
    int pw_layers_parallelism_w[MODEL_NUM_LAYERS];
    int fused_pw_layers_parallelism_w[MODEL_NUM_LAYERS];
    float layers_exec_times[MODEL_NUM_LAYERS] = {0};
    float fused_layers_exec_times[MODEL_NUM_LAYERS] = {0};

    layer_specs_init(layer_specs_seq, pooling_layer_specs_seq);

    cudaError_t err = cudaSuccess;
    fms_dt *ifms, *ofms, *d_ifms, *d_ofms, *d_tmp_fms, *ifms_cpu, *ofms_cpu, *tmp_fms_cpu;
    weights_dt *weights, *d_weights, *weights_cpu,
        *dw_weights, *d_dw_weights, *dw_weights_cpu;

    biases_dt *fused_zps, *d_fused_zps;
    fused_scales_dt *fused_scales, *d_fused_scales;

    string weights_dir = get_model_prefix() + "_weights/";

    string weights_file = weights_dir + "conv_pw_weights.txt";
    string dw_weights_file = weights_dir + "dw_weights.txt";
    string weights_count_file = weights_dir + "num_of_conv_pw_weights.txt";
    string dw_weights_count_file = weights_dir + "num_of_dw_weights.txt";
    string fused_scales_file = weights_dir + "fused_scales.txt";
    string fused_zps_file = weights_dir + "fused_zps.txt";
    string fused_params_count_file = weights_dir + "num_of_conv_pw_weights.txt";
    string fused_params_offsets_file = weights_dir + "fused_params_offsets.txt";

    Settings_struct settings;
    read_settings("benchmarking_settings.cfg", settings);

    read_fusions_list(settings.fusion_file, layers_fusions);

    const int first_layer = settings.first_layer;
    const int num_layers = settings.num_layers;
    const int last_layer = get_conv_layer_index(first_layer, settings.num_layers - 1);

    string ifms_file = settings.ifms_file_name;
    ifms_file = ifms_file.replace(ifms_file.find("*L*"), 3, to_string(settings.first_layer));
    cout << ifms_file << "\n";

    string dump_file = settings.dump_file_name;
    dump_file = dump_file.replace(dump_file.find("*L*"), 3, to_string(last_layer));

    const int weights_count = read_count_from_a_file(weights_count_file);
    const int dw_weights_count = read_count_from_a_file(dw_weights_count_file);
    const int fused_params_count = read_count_from_a_file(fused_params_count_file);
    int fused_params_offsets[MODEL_NUM_LAYERS];
    read_ints_from_file(fused_params_offsets_file, fused_params_offsets);

    ifms = (fms_dt *)malloc(MAX_FMS_SIZE_PACKED * sizeof(fms_dt));
    ifms_cpu = (fms_dt *)malloc(MAX_FMS_SIZE * sizeof(fms_dt));
    ofms = (fms_dt *)malloc(MAX_FMS_SIZE_PACKED * sizeof(fms_dt));
    ofms_cpu = (fms_dt *)malloc(MAX_FMS_SIZE * sizeof(fms_dt));
    tmp_fms_cpu = (fms_dt *)malloc(MAX_TMP_FMS_SIZE * sizeof(fms_dt));
    memset(ofms, 0, MAX_FMS_SIZE_PACKED * sizeof(fms_dt));
    memset(ofms_cpu, 0, MAX_FMS_SIZE * sizeof(fms_dt));
    weights = (weights_dt *)malloc(weights_count / PACKED_ITEMS * sizeof(weights_dt));
    weights_cpu = (weights_dt *)malloc(weights_count * sizeof(weights_dt));
    dw_weights = (weights_dt *)malloc(dw_weights_count * sizeof(weights_dt));
    dw_weights_cpu = (weights_dt *)malloc(dw_weights_count * sizeof(weights_dt));
    fused_scales = (fused_scales_dt *)malloc(fused_params_count * sizeof(fused_scales_dt));
    fused_zps = (biases_dt *)malloc(fused_params_count * sizeof(biases_dt));

    float not_fused_total_time = 0.0, fused_total_time = 0.0;

#if DATA_LAYOUT == HWC
    fill_layer_input_fw_hwz(ifms_file,
                            ifms, layer_specs_seq[first_layer]);
#elif DATA_LAYOUT == HCW
    fill_layer_input_fw_hzw(ifms_file,
                            ifms, layer_specs_seq[first_layer]);
#elif DATA_LAYOUT == CHW
    fill_layer_input_fwh_zhw(ifms_file,
                             ifms, layer_specs_seq[first_layer]);
#endif

    fill_layer_input_cpu(ifms_file,
                         ifms_cpu, layer_specs_seq[first_layer]);

    load_weights(weights_file, weights, layer_specs_seq);
#if PADDED_DW_WEIGHTS
    load_dw_weights_padded_filters(dw_weights_file, dw_weights, layer_specs_seq);
#else
    load_dw_weights(dw_weights_file, dw_weights);
#endif
    verify_load_weights(weights_file,
                        weights, weights_count);

    load_scales(fused_scales_file, fused_scales);
    load_zps(fused_zps_file, fused_zps);

    load_weights_cpu(weights_file, weights_cpu);
    load_dw_weights_cpu_padded(dw_weights_file, dw_weights_cpu, layer_specs_seq);

    get_pw_f_w_v2_parallelism_w(layer_specs_seq, pw_layers_parallelism_w, MODEL_NUM_LAYERS, false);
    get_pw_f_w_v2_parallelism_w(layer_specs_seq, fused_pw_layers_parallelism_w, MODEL_NUM_LAYERS, true);

    int cpu_direction = 0, gpu_direction = 0, fused_gpu_direction = 0;

    // for (int i = 0; i < dw_weights_count / PACKED_ITEMS; i++)
    // {
    //     for (int j = 0; j < PACKED_ITEMS; j++)
    //     {
    //         if (i % 16 < 9)
    //         {
    //             printf("%d\n", EXTRACT_8_32(weights[i], j));
    //         }
    //     }
    // }

#if FW
    for (int i = 0; i < num_layers; i++)
    {
        const int current_layer_index = get_conv_layer_index(first_layer, i);
        layer_specs l_specs = layer_specs_seq[current_layer_index];
        const int layer_ofms_size = l_specs.layer_ofm_height * l_specs.layer_ofm_width * l_specs.layer_num_fils * sizeof(fms_dt);
        if (cpu_direction == 0)
        {
            convolutionCPU_fw(ifms_cpu, ofms_cpu, weights_cpu, dw_weights_cpu, fused_scales, fused_zps,
                              fused_params_offsets, l_specs);
            if (l_specs.followed_by == ADD_LAYER_ID)
            {
                cpu_add(tmp_fms_cpu, ofms_cpu, l_specs);
            }
            if (l_specs.write_to_tmp)
            {
                // printf("*************%d\n", layer_ofms_size);
                memcpy(tmp_fms_cpu, ofms_cpu, layer_ofms_size);
                // printf(">>>>>>>>>>%d, %d, \n", tmp_fms_cpu[56], ofms_cpu[56]);
            }
            if (l_specs.followed_by == AVG_POOL_LAYER_ID)
            {
                pooling_layer_specs pool_l_specs = pooling_layer_specs_seq[current_layer_index + 1];
                if (pool_l_specs.full_hw)
                {
                    cpu_avgpool_all_hw(ofms_cpu, pool_l_specs);
                }
            }
        }
        else
        {
            convolutionCPU_fw(ofms_cpu, ifms_cpu, weights_cpu, dw_weights_cpu, fused_scales, fused_zps,
                              fused_params_offsets, l_specs);
            if (l_specs.followed_by == ADD_LAYER_ID)
            {
                cpu_add(tmp_fms_cpu, ifms_cpu, l_specs);
            }
            if (l_specs.write_to_tmp)
            {
                // printf("*************%d\n", layer_ofms_size);
                memcpy(tmp_fms_cpu, ifms_cpu, layer_ofms_size);
                // printf(">>>>>>>>>>%d, %d, \n", tmp_fms_cpu[56], ifms_cpu[56]);
            }
            if (l_specs.followed_by == AVG_POOL_LAYER_ID)
            {
                pooling_layer_specs pool_l_specs = pooling_layer_specs_seq[current_layer_index + 1];
                if (pool_l_specs.full_hw)
                {
                    cpu_avgpool_all_hw(ifms_cpu, pool_l_specs);
                }
            }
        }
        cpu_direction = 1 - cpu_direction;
    }

    if (cpu_direction == 0)
    {
        dump_cpu_output(dump_file, ifms_cpu, layer_specs_seq[last_layer]);
    }
    else
    {
        dump_cpu_output(dump_file, ofms_cpu, layer_specs_seq[last_layer]);
    }
#endif
    err = cudaMalloc((void **)&d_ifms, MAX_FMS_SIZE_PACKED * sizeof(fms_dt));
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to cudaMalloc d_ifms %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_ofms, MAX_FMS_SIZE_PACKED * sizeof(fms_dt));
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to cudaMalloc d_ofms %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemset(d_ofms, 0, MAX_FMS_SIZE_PACKED * sizeof(fms_dt));

    err = cudaMalloc((void **)&d_tmp_fms, MAX_TMP_FMS_SIZE_PACKED * sizeof(fms_dt));
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to cudaMalloc d_ofms %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_weights, weights_count / PACKED_ITEMS * sizeof(weights_dt));
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to cudaMalloc d_weights %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_dw_weights, dw_weights_count * sizeof(weights_dt));
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to cudaMalloc d_weights %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_fused_scales, fused_params_count * sizeof(fused_scales_dt));
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to cudaMalloc d_fused_scales %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_fused_zps, fused_params_count * sizeof(biases_dt));
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to cudaMalloc d_fused_zps %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_ifms, ifms,
                     MAX_FMS_SIZE_PACKED * sizeof(fms_dt),
                     cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy vector d_ifms from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_weights, weights,
                     weights_count / PACKED_ITEMS * sizeof(weights_dt),
                     cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy vector d_weights from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_dw_weights, dw_weights,
                     dw_weights_count * sizeof(weights_dt),
                     cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy vector d_dw_weights from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_fused_scales, fused_scales,
                     fused_params_count * sizeof(fused_scales_dt),
                     cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy vector d_fused_scales from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_fused_zps, fused_zps,
                     fused_params_count * sizeof(biases_dt),
                     cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy vector d_fused_zps from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float fused_time_pw_dw = 0.0, fused_time_dw_pw = 0.0, not_fused_time_pw = 0.0,
          not_fused_time_dw_2 = 0.0, with_fused_time_dw_2 = 0.0, not_fused_time_dw_5 = 0.0, with_fused_time_dw_5 = 0.0;

    //**************************************************************************************************************************************
    if (settings.run_unfused)
    {
        for (int i = 0; i < settings.test_iterations; i++)
        {
            if (settings.bench == 0)
            {
                fill_layer_input_fw_hzw(ifms_file,
                                        ifms, layer_specs_seq[first_layer]);
            }
            else if (settings.bench == 1)
            {
                err = cudaMemcpy(d_ifms, ifms,
                                 MAX_FMS_SIZE_PACKED * sizeof(fms_dt),
                                 cudaMemcpyHostToDevice);

                if (err != cudaSuccess)
                {
                    fprintf(stderr,
                            "Failed to copy vector d_ifms from host to device (error code %s)!\n",
                            cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }
                cudaDeviceSynchronize();
            }
            gpu_direction = 0;
#if !TIME_LAYER_BY_LAYER
            float elapsed_time;
            cudaEvent_t start_event, stop_event;
            err = (cudaEventCreate(&start_event));
            if (err != cudaSuccess)
            {
                fprintf(stderr,
                        "Failed to cudaEventCreate start_event %s\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            err = (cudaEventCreate(&stop_event));
            if (err != cudaSuccess)
            {
                fprintf(stderr,
                        "Failed to cudaEventCreate stop_event %s\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            err = cudaEventRecord(start_event, 0);
            if (err != cudaSuccess)
            {
                fprintf(stderr,
                        "Failed to cudaEventRecord start_event %s\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
#endif
            for (int j = 0; j < num_layers; j++)
            {
                const int current_layer_index = get_conv_layer_index(first_layer, j);
                const int prev_layer_index = get_conv_layer_index(first_layer, j - 1);
                int fusion_type = -1;
                layer_specs l_specs = layer_specs_seq[current_layer_index];
                const int layer_ofms_size = l_specs.layer_ofm_height * l_specs.layer_ofm_width *
                                            l_specs.layer_num_fils * sizeof(fms_dt) / PACKED_ITEMS;
                if (layers_fusions[current_layer_index].first_layer_index != -1)
                {
                    fusion_type = layers_fusions[current_layer_index].fusion_type;
                }
                else if (layers_fusions[prev_layer_index].first_layer_index != -1)
                {
                    fusion_type = layers_fusions[prev_layer_index].fusion_type;
                }
                if (gpu_direction == 0)
                {
                    if (fusion_type == pwdw)
                    {
                        convolutionGPU_h_w_chw(d_ifms, d_ofms, d_weights, d_dw_weights, d_fused_scales, d_fused_zps,
                                               fused_params_offsets, l_specs,
                                               i, layers_exec_times[current_layer_index]);
                    }
                    else if (fusion_type == pwdw_wide)
                    {
                        convolutionGPU_h_w_chw_wide(d_ifms, d_ofms, d_weights, d_dw_weights, d_fused_scales, d_fused_zps,
                                                    fused_params_offsets, l_specs,
                                                    i, layers_exec_times[current_layer_index]);
                    }
                    else if (fusion_type == dwpw)
                    {
                        convolutionGPU_f_w_chw(d_ifms, d_ofms, d_weights, d_dw_weights, d_fused_scales, d_fused_zps,
                                               fused_params_offsets, l_specs,
                                               i, layers_exec_times[current_layer_index]);
                    }
                    else if (fusion_type == pwpw)
                    {
                        pw_convolutionGPU_f_w_v2_chw(d_ifms, d_ofms, d_weights, d_fused_scales, d_fused_zps,
                                                     l_specs,
                                                     fused_params_offsets, i, pw_layers_parallelism_w, layers_exec_times[current_layer_index]);
                    }
                    else
                    {
                        convolutionGPU_h_w_chw(d_ifms, d_ofms, d_weights, d_dw_weights, d_fused_scales, d_fused_zps,
                                               fused_params_offsets, l_specs,
                                               i, layers_exec_times[current_layer_index]);
                    }

                    if (l_specs.followed_by== ADD_LAYER_ID)
                    {
                        gpu_add(d_tmp_fms, d_ofms, l_specs, settings);
                    }
                    if (l_specs.write_to_tmp)
                    {
                        // printf("*************%d\n", layer_ofms_size);
                        cudaMemcpy(d_tmp_fms, d_ofms, layer_ofms_size, cudaMemcpyDeviceToDevice);
                        // printf(">>>>>>>>>>%d, %d, \n", tmp_fms_cpu[56], ofms_cpu[56]);
                    }
                    if (l_specs.followed_by == AVG_POOL_LAYER_ID)
                    {
                        pooling_layer_specs pool_l_specs = pooling_layer_specs_seq[current_layer_index + 1];
                        if (pool_l_specs.full_hw)
                        {
                            gpu_avgpool_all_hw(d_ofms, pool_l_specs);
                        }
                    }
                }
                else
                {
                    if (fusion_type == pwdw)
                    {
                        convolutionGPU_h_w_chw(d_ofms, d_ifms, d_weights, d_dw_weights, d_fused_scales, d_fused_zps,
                                               fused_params_offsets, l_specs,
                                               i, layers_exec_times[current_layer_index]);
                    }
                    else if (fusion_type == pwdw_wide)
                    {
                        convolutionGPU_h_w_chw_wide(d_ofms, d_ifms, d_weights, d_dw_weights, d_fused_scales, d_fused_zps,
                                                    fused_params_offsets, l_specs,
                                                    i, layers_exec_times[current_layer_index]);
                    }
                    else if (fusion_type == dwpw)
                    {
                        convolutionGPU_f_w_chw(d_ofms, d_ifms, d_weights, d_dw_weights, d_fused_scales, d_fused_zps,
                                               fused_params_offsets, l_specs,
                                               i, layers_exec_times[current_layer_index]);
                    }
                    else if (fusion_type == pwpw)
                    {
                        pw_convolutionGPU_f_w_v2_chw(d_ofms, d_ifms, d_weights, d_fused_scales, d_fused_zps,
                                                     l_specs,
                                                     fused_params_offsets, i, pw_layers_parallelism_w, layers_exec_times[current_layer_index]);
                    }
                    else
                    {
                        convolutionGPU_h_w_chw(d_ofms, d_ifms, d_weights, d_dw_weights, d_fused_scales, d_fused_zps,
                                               fused_params_offsets, l_specs,
                                               i, layers_exec_times[current_layer_index]);
                    }

                    if (l_specs.followed_by == ADD_LAYER_ID)
                    {
                        gpu_add(d_tmp_fms, d_ifms, l_specs, settings);
                    }
                    if (l_specs.write_to_tmp)
                    {
                        // printf("*************%d\n", layer_ofms_size);
                        cudaMemcpy(d_tmp_fms, d_ifms, layer_ofms_size, cudaMemcpyDeviceToDevice);
                        // printf(">>>>>>>>>>%d, %d, \n", tmp_fms_cpu[56], ofms_cpu[56]);
                    }
                    if (l_specs.followed_by == AVG_POOL_LAYER_ID)
                    {
                        pooling_layer_specs pool_l_specs = pooling_layer_specs_seq[current_layer_index + 1];
                        if (pool_l_specs.full_hw)
                        {
                            gpu_avgpool_all_hw(d_ifms, pool_l_specs);
                        }
                    }
                }
                gpu_direction = 1 - gpu_direction;
            }
#if !TIME_LAYER_BY_LAYER
            err = (cudaEventRecord(stop_event, 0));
            if (err != cudaSuccess)
            {
                fprintf(stderr,
                        "Failed to cudaEventRecord stop_event %s\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            err = (cudaEventSynchronize(stop_event));
            if (err != cudaSuccess)
            {
                fprintf(stderr,
                        "Failed to cudaEventSynchronize %s\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            err = (cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
            if (err != cudaSuccess)
            {
                fprintf(stderr,
                        "Failed to cudaEventElapsedTime %s\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            if (i >= WARMUP_ITERATIONS)
            {
                not_fused_total_time += elapsed_time;
            }
#endif
        }
        if (gpu_direction == 0) // last was 1
        {
            err = cudaMemcpy(ofms, d_ifms, MAX_FMS_SIZE_PACKED * sizeof(fms_dt), cudaMemcpyDeviceToHost);
        }
        else
        {
            err = cudaMemcpy(ofms, d_ofms, MAX_FMS_SIZE_PACKED * sizeof(fms_dt), cudaMemcpyDeviceToHost);
        }

        if (err != cudaSuccess)
        {
            fprintf(stderr,
                    "Failed to cudaMemcpyDeviceToHost %s\n",
                    cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        cudaDeviceSynchronize();
        if (cpu_direction == 0)
        {
            printf("comp1\n");
#if DATA_LAYOUT == HWC
            compare_cpu_and_gpu_outputs_fw_hwz(ifms_cpu, ofms, layer_specs_seq[last_layer]);
#elif DATA_LAYOUT == HCW
            compare_cpu_and_gpu_outputs_fw_hzw(ifms_cpu, ofms, layer_specs_seq[last_layer]);
#elif DATA_LAYOUT == CHW
            compare_cpu_and_gpu_outputs_fhw_zhw(ifms_cpu, ofms, layer_specs_seq[last_layer]);
#endif
        }
        else
        {
            printf("comp1\n");
#if DATA_LAYOUT == HWC
            compare_cpu_and_gpu_outputs_fw_hwz(ofms_cpu, ofms, layer_specs_seq[last_layer]);
#elif DATA_LAYOUT == HCW
            compare_cpu_and_gpu_outputs_fw_hzw(ofms_cpu, ofms, layer_specs_seq[last_layer]);
#elif DATA_LAYOUT == CHW
            compare_cpu_and_gpu_outputs_fhw_zhw(ofms_cpu, ofms, layer_specs_seq[last_layer]);
#endif
        }
    } // end if run_unfused
    //**************************************************************************************************************************************
    // begin fused runs
#if COMPILE_FUSED
    if (settings.run_fused)
    {
        for (int i = 0; i < settings.test_iterations; i++)
        {
            fused_gpu_direction = 0;
            if (settings.bench == 0)
            {
                fill_layer_input_fw_hwz(ifms_file,
                                        ifms, layer_specs_seq[first_layer]);
            }
            else if (settings.bench == 1)
            {
                err = cudaMemcpy(d_ifms, ifms,
                                 MAX_FMS_SIZE_PACKED * sizeof(fms_dt),
                                 cudaMemcpyHostToDevice);

                if (err != cudaSuccess)
                {
                    fprintf(stderr,
                            "Failed to copy vector d_ifms from host to device (error code %s)!\n",
                            cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }
                cudaDeviceSynchronize();
            }
            bool fusion_dest_layer = false;
#if !TIME_LAYER_BY_LAYER
            float elapsed_time;
            cudaEvent_t start_event, stop_event;
            err = (cudaEventCreate(&start_event));
            if (err != cudaSuccess)
            {
                fprintf(stderr,
                        "Failed to cudaEventCreate start_event %s\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            err = (cudaEventCreate(&stop_event));
            if (err != cudaSuccess)
            {
                fprintf(stderr,
                        "Failed to cudaEventCreate stop_event %s\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            err = cudaEventRecord(start_event, 0);
            if (err != cudaSuccess)
            {
                fprintf(stderr,
                        "Failed to cudaEventRecord start_event %s\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
#endif
            for (int j = 0; j < num_layers; j++)
            {
                if (fusion_dest_layer)
                {
                    fusion_dest_layer = false;
                    continue;
                }

                int layer_1_index = get_conv_layer_index(first_layer, j);
                int fusion_or_no_end_index = layer_1_index;
                int layer_2_index;
                bool fusion_layer = layers_fusions[layer_1_index].first_layer_index != -1 && j < num_layers - 1;
                Fusion_struct fusion_struct = layers_fusions[layer_1_index];
                if (fusion_layer)
                {
                    layer_2_index = fusion_struct.second_layer_index;
                    fusion_or_no_end_index = layer_2_index;
                    fusion_dest_layer = true;
                }
                const int layer_ofms_size = layer_specs_seq[fusion_or_no_end_index].layer_ofm_height *
                                            layer_specs_seq[fusion_or_no_end_index].layer_ofm_width *
                                            layer_specs_seq[fusion_or_no_end_index].layer_num_fils * sizeof(fms_dt) / PACKED_ITEMS;
                if (fused_gpu_direction == 0)
                {
                    if (fusion_layer)
                    {
                        if (fusion_struct.fusion_type == pwdw)
                        {
                            fused_pw_dw_convolutionGPU_h_w_chw(d_ifms, d_ofms,
                                                               d_weights,
                                                               d_dw_weights,
                                                               d_fused_scales,
                                                               d_fused_zps,
                                                               layer_specs_seq[layer_1_index],
                                                               layer_specs_seq[layer_2_index],
                                                               fused_params_offsets,
                                                               i,
                                                               fused_layers_exec_times[layer_1_index],
                                                               settings.num_sms);
                        }
                        else if (fusion_struct.fusion_type == pwdw_wide)
                        {
                            fused_pw_dw_convolutionGPU_h_w_chw_wide(d_ifms, d_ofms,
                                                                    d_weights,
                                                                    d_dw_weights,
                                                                    d_fused_scales,
                                                                    d_fused_zps,
                                                                    layer_specs_seq[layer_1_index],
                                                                    layer_specs_seq[layer_2_index],
                                                                    fused_params_offsets,
                                                                    i,
                                                                    fused_layers_exec_times[layer_1_index],
                                                                    settings.num_sms);
                        }
                        else if (fusion_struct.fusion_type == dwpw)
                        {
                            fused_dwpw_convolutionGPU_chw(d_ifms, d_ofms,
                                                          d_weights,
                                                          d_dw_weights,
                                                          d_fused_scales,
                                                          d_fused_zps,
                                                          layer_specs_seq[layer_1_index],
                                                          layer_specs_seq[layer_2_index],
                                                          fused_params_offsets,
                                                          i,
                                                          fused_layers_exec_times[layer_1_index]);
                        }
                        else if (fusion_struct.fusion_type == pwpw)
                        {
                            // if (i == 0)
                            // {
                            //     printf("fused_pw_pw_convolutionGPU\n");
                            // }
                            fused_pw_pw_convolutionGPU_chw(d_ifms, d_ofms, d_weights, d_fused_scales, d_fused_zps,
                                                           layer_specs_seq[layer_1_index], layer_specs_seq[layer_2_index],
                                                           fused_params_offsets, i, fused_pw_layers_parallelism_w, fused_layers_exec_times[layer_1_index]);
                        }
                    }
                    else
                    {
                        convolutionGPU_h_w_chw(d_ifms, d_ofms, d_weights, d_dw_weights, d_fused_scales, d_fused_zps, fused_params_offsets,
                                               layer_specs_seq[layer_1_index], i,
                                               fused_layers_exec_times[layer_1_index]);
                    }
                    if (layer_specs_seq[fusion_or_no_end_index].followed_by == ADD_LAYER_ID)
                    {
                        gpu_add(d_tmp_fms, d_ofms, layer_specs_seq[fusion_or_no_end_index], settings);
                    }
                    if (layer_specs_seq[fusion_or_no_end_index].write_to_tmp)
                    {
                        // printf("*************%d\n", layer_ofms_size);
                        cudaMemcpy(d_tmp_fms, d_ofms, layer_ofms_size, cudaMemcpyDeviceToDevice);
                        // printf(">>>>>>>>>>%d, %d, \n", tmp_fms_cpu[56], ofms_cpu[56]);
                    }
                    if (layer_specs_seq[fusion_or_no_end_index].followed_by == AVG_POOL_LAYER_ID)
                    {
                        pooling_layer_specs pool_l_specs = pooling_layer_specs_seq[fusion_or_no_end_index + 1];
                        if (pool_l_specs.full_hw)
                        {
                            gpu_avgpool_all_hw(d_ofms, pool_l_specs);
                        }
                    }
                }
                else
                {
                    if (fusion_layer)
                    {
                        if (fusion_struct.fusion_type == pwdw)
                        {
                            fused_pw_dw_convolutionGPU_h_w_chw(d_ofms, d_ifms,
                                                               d_weights,
                                                               d_dw_weights,
                                                               d_fused_scales,
                                                               d_fused_zps,
                                                               layer_specs_seq[layer_1_index],
                                                               layer_specs_seq[layer_2_index],
                                                               fused_params_offsets,
                                                               i,
                                                               fused_layers_exec_times[layer_1_index],
                                                               settings.num_sms);
                        }
                        else if (fusion_struct.fusion_type == pwdw_wide)
                        {
                            fused_pw_dw_convolutionGPU_h_w_chw_wide(d_ofms, d_ifms,
                                                                    d_weights,
                                                                    d_dw_weights,
                                                                    d_fused_scales,
                                                                    d_fused_zps,
                                                                    layer_specs_seq[layer_1_index],
                                                                    layer_specs_seq[layer_2_index],
                                                                    fused_params_offsets,
                                                                    i,
                                                                    fused_layers_exec_times[layer_1_index],
                                                                    settings.num_sms);
                        }
                        else if (fusion_struct.fusion_type == dwpw)
                        {
                            fused_dwpw_convolutionGPU_chw(d_ofms, d_ifms,
                                                          d_weights,
                                                          d_dw_weights,
                                                          d_fused_scales,
                                                          d_fused_zps,
                                                          layer_specs_seq[layer_1_index],
                                                          layer_specs_seq[layer_2_index],
                                                          fused_params_offsets,
                                                          i,
                                                          fused_layers_exec_times[layer_1_index]);
                        }
                        else if (fusion_struct.fusion_type == pwpw)
                        {
                            // if (i == 0)
                            // {
                            //     printf("fused_pw_pw_convolutionGPU\n");
                            // }
                            fused_pw_pw_convolutionGPU_chw(d_ofms, d_ifms, d_weights, d_fused_scales, d_fused_zps,
                                                           layer_specs_seq[layer_1_index], layer_specs_seq[layer_2_index],
                                                           fused_params_offsets, i, fused_pw_layers_parallelism_w, fused_layers_exec_times[layer_1_index]);
                        }
                    }
                    else
                    {
                        convolutionGPU_h_w_chw(d_ofms, d_ifms, d_weights, d_dw_weights,
                                               d_fused_scales, d_fused_zps, fused_params_offsets,
                                               layer_specs_seq[layer_1_index], i,
                                               fused_layers_exec_times[layer_1_index]);
                    }
                    if (layer_specs_seq[fusion_or_no_end_index].followed_by == ADD_LAYER_ID)
                    {
                        gpu_add(d_tmp_fms, d_ifms, layer_specs_seq[fusion_or_no_end_index], settings);
                    }
                    if (layer_specs_seq[fusion_or_no_end_index].write_to_tmp)
                    {
                        // printf("*************%d\n", layer_ofms_size);
                        cudaMemcpy(d_tmp_fms, d_ifms, layer_ofms_size, cudaMemcpyDeviceToDevice);
                        // printf(">>>>>>>>>>%d, %d, \n", tmp_fms_cpu[56], ofms_cpu[56]);
                    }
                    if (layer_specs_seq[fusion_or_no_end_index].followed_by == AVG_POOL_LAYER_ID)
                    {
                        //printf("FFFFFFFFFFFFFFFFFFFFFFFFF\n");
                        pooling_layer_specs pool_l_specs = pooling_layer_specs_seq[fusion_or_no_end_index + 1];
                        if (pool_l_specs.full_hw)
                        {
                            //printf("FFFFFFFFFFFFFFFFFFFFFFFFF\n");
                            gpu_avgpool_all_hw(d_ifms, pool_l_specs);
                        }
                    }
                }
                fused_gpu_direction = 1 - fused_gpu_direction;
            }
#if !TIME_LAYER_BY_LAYER
            err = (cudaEventRecord(stop_event, 0));
            if (err != cudaSuccess)
            {
                fprintf(stderr,
                        "Failed to cudaEventRecord stop_event %s\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            err = (cudaEventSynchronize(stop_event));
            if (err != cudaSuccess)
            {
                fprintf(stderr,
                        "Failed to cudaEventSynchronize %s\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            err = (cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
            if (err != cudaSuccess)
            {
                fprintf(stderr,
                        "Failed to cudaEventElapsedTime %s\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            if (i >= WARMUP_ITERATIONS)
            {
                fused_total_time += elapsed_time;
            }
#endif
        }

        if (fused_gpu_direction == 0) // last was 1
        {
            err = cudaMemcpy(ofms, d_ifms, MAX_FMS_SIZE_PACKED * sizeof(fms_dt), cudaMemcpyDeviceToHost);
        }
        else
        {
            err = cudaMemcpy(ofms, d_ofms, MAX_FMS_SIZE_PACKED * sizeof(fms_dt), cudaMemcpyDeviceToHost);
        }

        if (err != cudaSuccess)
        {
            fprintf(stderr,
                    "Failed to cudaMemcpyDeviceToHost %s\n",
                    cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        cudaDeviceSynchronize();
        if (cpu_direction == 0)
        {
            printf("comp2\n");
#if DATA_LAYOUT == HWC
            compare_cpu_and_gpu_outputs_fw_hwz(ifms_cpu, ofms, layer_specs_seq[last_layer]);
#elif DATA_LAYOUT == HCW
            compare_cpu_and_gpu_outputs_fw_hzw(ifms_cpu, ofms, layer_specs_seq[last_layer]);
#elif DATA_LAYOUT == CHW
            compare_cpu_and_gpu_outputs_fhw_zhw(ifms_cpu, ofms, layer_specs_seq[last_layer]);
#endif
        }
        else
        {
            printf("comp2\n");
#if DATA_LAYOUT == HWC
            compare_cpu_and_gpu_outputs_fw_hwz(ofms_cpu, ofms, layer_specs_seq[last_layer]);
#elif DATA_LAYOUT == HCW
            compare_cpu_and_gpu_outputs_fw_hzw(ofms_cpu, ofms, layer_specs_seq[last_layer]);
#elif DATA_LAYOUT == CHW
            compare_cpu_and_gpu_outputs_fhw_zhw(ofms_cpu, ofms, layer_specs_seq[last_layer]);
#endif
        }
    }
#endif
    //************************************************************************************************************

    float num_of_timed_iterations = (float)(settings.test_iterations - WARMUP_ITERATIONS);

#if TIME_LAYER_BY_LAYER
    for (int i = 0; i < MODEL_NUM_LAYERS; i++)
    {
        not_fused_total_time += layers_exec_times[i];
        fused_total_time += fused_layers_exec_times[i];
        if (layers_exec_times[i] != 0)
        {
            printf("%d: %f, %f\n", i, layers_exec_times[i] / num_of_timed_iterations,
                   fused_layers_exec_times[i] / num_of_timed_iterations);
            // printf("%f\n", fused_layers_exec_times[i] / num_of_timed_iterations);
            // printf("%f\n", layers_exec_times[i] / num_of_timed_iterations);
        }
    }
#endif

    printf("from %d to %d\n", first_layer, last_layer);
    printf("****************************\n");
    printf("average not fused time: %f\n", not_fused_total_time / num_of_timed_iterations);
    printf("****************************\n");
    printf("average fused time: %f\n", fused_total_time / num_of_timed_iterations);

    cudaFree(d_ifms);
    cudaFree(d_tmp_fms);
    cudaFree(d_ofms);
    cudaFree(d_weights);
    cudaFree(d_dw_weights);
    cudaFree(d_fused_scales);
    cudaFree(d_fused_zps);

    free(ifms);
    free(tmp_fms_cpu);
    free(ofms);
    free(weights);
    free(dw_weights);
    free(ifms_cpu);
    free(ofms_cpu);
    free(weights_cpu);
    free(dw_weights_cpu);

    return EXIT_SUCCESS;
}

#endif